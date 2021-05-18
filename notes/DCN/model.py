#!/usr/bin/env python
# coding=utf-8

#  author: tantanli
#  2021.04.14
#  deep and cross network

import tensorflow as tf
from tensorflow import keras

class CrossLayer(keras.layers.Layer):
    def __init__(self):
        super(CrossLayer, self).__init__()
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],), initializer="random_normal", trainable=True
        )

    def call(self, inputs, **kwargs):
        assert "first_input" in kwargs
        first_input_tensor = kwargs["first_input"]
        cross_tensor = first_input_tensor * tf.expand_dims(tf.reduce_sum(inputs * self.w, axis=-1), axis=-1)
        output_tensor = cross_tensor + self.b + inputs
        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape


def default_feature_cols_fn(params):
    deep_feature_cols = []
    cross_feature_cols = []
    feature_names = ["dcn_deep_cross_feature_%d" % (d+1) for d in range(4)]
    boundaries = [0.01 * i - 1 for i in range(200)]
    for name in feature_names:
        if "cross" in name:
            numeric_column = tf.feature_column.numeric_column(name)
            bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
            embedding_size = params.get("embedding_size", 64)
            embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
            deep_feature_cols.append(embedding_column)
        if "deep" in name:
            numeric_column = tf.feature_column.numeric_column(name)
            bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
            embedding_size = params.get("embedding_size", 64)
            embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
            cross_feature_cols.append(embedding_column)
    return {"deep_feature_columns": deep_feature_cols, "cross_feature_columns": cross_feature_cols}


def build_model_fn(gen_model_params, feature_cols_fn):
    def model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        assert "deep_feature_columns" in feature_cols
        assert "cross_feature_columns" in feature_cols
        assert "embedding_size" in params
        deep_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["deep_feature_columns"]]
        cross_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["cross_feature_columns"]]

        ##############################################################################
        # deep
        ##############################################################################
        deep_concat_vectors = tf.concat(deep_inputs, axis=1)  # [batch_size, sum_embedding)]
        x = deep_concat_vectors
        if "deep_layers" not in params:
            deep_layers = [params["embedding_size"], params["embedding_size"]]
        else:
            deep_layers = params["deep_layers"]
        for i in deep_layers:
            this_layer = tf.keras.layers.Dense(
                i,
                activation="relu",
            )
            x = this_layer(x)  # [batch_size, layer[i]]
        deep_output = x  # [batch_size, layer[-1]]

        ##############################################################################
        # cross
        ##############################################################################
        cross_inputs_concat_vectors = tf.concat(cross_inputs, axis=1)  # [batch_size, sum_embedding)]
        cross_frist_input = cross_inputs_concat_vectors
        x = cross_inputs_concat_vectors
        cross_layer_num = params.get("cross_layer_num", 3)

        for i in range(cross_layer_num):
            this_layer = CrossLayer()
            x = this_layer(x, first_input=cross_frist_input)  # [batch_size, layer[i]]
        cross_output = x

        ##############################################################################
        # merge
        ##############################################################################
        merged_tensor = tf.concat([deep_output, cross_output], axis=-1)
        logits = tf.keras.layers.Dense(1, activation="sigmoid")(merged_tensor)
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
    train_ds = inputs.test_dcn_input_fn(batch_size=5)
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
    print("feature_deep_1")
    demo(default_feature_cols_fn(params)["deep_feature_columns"][0])
    print("feature_deep_1")
    demo(default_feature_cols_fn(params)["cross_feature_columns"][0])

