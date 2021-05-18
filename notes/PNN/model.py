#!/usr/bin/env python
# coding=utf-8

#  author: tantanli
#  2021.04.21
#  pnn network

import tensorflow as tf
from tensorflow import keras


class InnerProductLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super(InnerProductLayer, self).__init__()
        self.w = None
        self.units = units

    def build(self, input_shape):
        # input shape: [batch_size, feature_num, embedding_size]
        assert len(input_shape) == 3
        self.w = self.add_weight(
            shape=(self.units, input_shape[-2]),
            initializer="random_normal",
            trainable=True,
        )  # w_shape: [units, feature_num]

    def call(self, inputs, **kwargs):
        # input shape: [batch_size, feature_num, embedding_size]
        output_tensor = tf.expand_dims(inputs, axis=1) * tf.expand_dims(self.w, axis=-1)
        # [batch_size, 1, feature_num, embedding_size] * [units, feature_num, 1]
        # [batch_size, units, feature_num, embedding_size]
        output_tensor = tf.reduce_sum(output_tensor, axis=-2)  # [batch_size, units, embedding_size]
        output_tensor = tf.reduce_sum(tf.square(output_tensor), axis=-1)  # [batch_size, units]
        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units  # [batch_size, units]


class LinearLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super(LinearLayer, self).__init__()
        self.w = None
        self.units = units

    def build(self, input_shape):
        # input shape: [batch_size, feature_num, embedding_size]
        assert len(input_shape) == 3
        concat_len = input_shape[-1] * input_shape[-2]
        self.w = self.add_weight(
            shape=(concat_len, self.units),
            initializer="random_normal",
            trainable=True,
        )  # w_shape: [feature_num * embedding_size, units]

    def call(self, inputs, **kwargs):
        # input shape: [batch_size, feature_num, embedding_size]
        embedding_size = inputs.shape[-1]
        feature_num = inputs.shape[-2]
        output_tensor = tf.reshape(inputs, [-1, embedding_size * feature_num])  # [batch_size, feature_num * embedding_size]
        output_tensor = tf.matmul(output_tensor, self.w, transpose_b=False)  # [batch_size, units]
        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units  # [batch_size, units]


def default_feature_cols_fn(params):
    feature_cols = []
    feature_names = ["pnn_feature_%d" % (d + 1) for d in range(4)]
    boundaries = [0.01 * i - 1 for i in range(200)]
    for name in feature_names:
        numeric_column = tf.feature_column.numeric_column(name)
        bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
        # categorical_column = tf.feature_column.categorical_column_with_identity(bucketized_column, len(boundaries))
        embedding_size = params.get("embedding_size", 64)
        embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
        feature_cols.append(embedding_column)
    return {"pnn_feature_columns": feature_cols}


def build_model_fn(gen_model_params, feature_cols_fn):
    def model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        pnn_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["pnn_feature_columns"]]

        ##############################################################################
        # linear
        ##############################################################################
        stack_vectors = tf.stack(pnn_inputs, axis=1)  # [batch_size, feature_num, embedding_size)]
        linear_layer = LinearLayer(params.get("linear_units", 64))
        linear_output = linear_layer(stack_vectors)  # [batch_size, linear_units]

        ##############################################################################
        # inner/outer product
        ##############################################################################
        stack_vectors = tf.stack(pnn_inputs, axis=1)  # [batch_size, feature_num, embedding_size)]
        if params.get("product_type", "inner") == "inner":
            product_layer = InnerProductLayer(params.get("product_units", 64))
        else:
            raise Exception("not ready for product type other than inner product")
        product_output = product_layer(stack_vectors)  # [batch_size, product_units]

        ##############################################################################
        # merge mlp
        ##############################################################################
        bais = tf.expand_dims(tf.ones(tf.shape(linear_output)[0]), axis=-1)  # [batch_size, 1]
        merged_tensor = \
            tf.concat([linear_output, product_output, bais], axis=-1)  # [batch_size, linear_units + product_units + 1]
        x = tf.keras.layers.ReLU()(merged_tensor)
        for i in params.get("mlp_layers", [32, 32]):
            x = tf.keras.layers.Dense(i, activation="relu")(x)
        output_tensor = x
        # logits = tf.keras.layers.Dense(1, activation="sigmoid")(output_tensor)
        logits = output_tensor
        logits_for_head = logits

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

        from tensorflow.estimator import BinaryClassHead
        binary_head = BinaryClassHead()
        return binary_head.create_estimator_spec(
            features=features,
            mode=mode,
            logits=logits_for_head,
            labels=labels,
            optimizer=optimizer,
            trainable_variables=tf.compat.v1.trainable_variables()
        )
    return model_fn


if __name__ == "__main__":
    import inputs
    train_ds = inputs.test_pnn_input_fn(batch_size=5)
    example_batch = next(iter(train_ds))[0]

    def demo(feature_column):
        feature_layer = tf.keras.layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())

    print("="*10 + "demo feature data" + "="*10)
    params = {"embedding_size": 4}

    print("label")
    print(next(iter(train_ds))[1])
    print("raw_feature")
    print(next(iter(train_ds))[0])
    print("feature_1")
    demo(default_feature_cols_fn(params)["pnn_feature_columns"][0])
    print("feature_2")
    demo(default_feature_cols_fn(params)["pnn_feature_columns"][1])

