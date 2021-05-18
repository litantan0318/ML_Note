#!/usr/bin/env python
# coding=utf-8

#  author: tantanli
#  2021.04.30
#  xdeepfm network
import tensorflow as tf

class CINLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CINLayer, self).__init__()
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
        assert len(inputs) == 2
        last_inputs, first_inputs = inputs
        # last_input shape: [batch_size, last_unit, embedding_size]
        # first_input shape: [batch_size, feature_num, embedding_size]
        output_tensor = tf.expand_dims(inputs, axis=1) * tf.expand_dims(self.w, axis=-1)
        # [batch_size, 1, feature_num, embedding_size] * [units, feature_num, 1]
        # [batch_size, units, feature_num, embedding_size]
        output_tensor = tf.reduce_sum(output_tensor, axis=-2)  # [batch_size, units, embedding_size]
        output_tensor = tf.reduce_sum(tf.square(output_tensor), axis=-1)  # [batch_size, units]
        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units  # [batch_size, units]


def default_feature_cols_fn(params):
    deep_feature_cols = []
    fm_feature_cols = []
    feature_names = ["deep_fm_feature_%d" % (d+1) for d in range(4)]
    boundaries = [0.01 * i - 1 for i in range(200)]
    for name in feature_names:
        if "fm" in name:
            numeric_column = tf.feature_column.numeric_column(name)
            bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
            # categorical_column = tf.feature_column.categorical_column_with_identity(bucketized_column, len(boundaries))
            embedding_size = params.get("embedding_size", 64)
            embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
            deep_feature_cols.append(embedding_column)
        if "deep" in name:
            numeric_column = tf.feature_column.numeric_column(name)
            bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
            # categorical_column = tf.feature_column.categorical_column_with_identity(bucketized_column, len(boundaries))
            embedding_size = params.get("embedding_size", 64)
            embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
            fm_feature_cols.append(embedding_column)
    return {"deep_feature_columns": deep_feature_cols, "fm_feature_columns": fm_feature_cols}


def build_model_fn(gen_model_params, feature_cols_fn):
    def model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        assert "deep_feature_columns" in feature_cols
        assert "fm_feature_columns" in feature_cols
        assert "embedding_size" in params
        deep_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["deep_feature_columns"]]
        fm_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["fm_feature_columns"]]

        # fm
        fm_stack_vectors = tf.stack(fm_inputs, axis=1)  # [batch_size * feature_num * embed_size]
        sum_and_square = tf.reduce_sum(tf.square(tf.reduce_sum(fm_stack_vectors, axis=1)), axis=1)  # batch
        square_and_sum = tf.reduce_sum(tf.reduce_sum(tf.square(fm_stack_vectors), axis=1), axis=1)  # batch
        fm_logits = 0.5 * (sum_and_square - square_and_sum)
        fm_logits_for_head = tf.expand_dims(fm_logits, axis=1)  # [batch * 1]
        # batch_logit = tf.reduce_mean(logits)

        # deep
        deep_concat_vectors = tf.concat(deep_inputs, axis=1)  # [batch_size, sum_embedding)]
        x = deep_concat_vectors
        if "layers" not in params:
            for _ in range(2):
                this_layer = tf.keras.layers.Dense(
                    params["embedding_size"],
                    activation="relu",
                )
                x = this_layer(x)  # [batch_size, node=embedding_size]
            deep_logits = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        else:
            for i in params["layers"]:
                this_layer = tf.keras.layers.Dense(
                    params["embedding_size"],
                    activation="relu",
                )
                x = this_layer(x)  # [batch_size, layer[i]]
            deep_logits = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        deep_logits_for_head = deep_logits  # [batch * 1]

        print("="*20)
        print(fm_logits_for_head)
        print(deep_logits_for_head)
        logits_for_head = fm_logits_for_head + deep_logits_for_head

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

        from tensorflow.estimator import BinaryClassHead
        binary_head = BinaryClassHead()
        # from estimator.head import _BinaryLogisticHeadWithSigmoidCrossEntropyLoss
        # binary_head = _BinaryLogisticHeadWithSigmoidCrossEntropyLoss()
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
    train_ds = inputs.test_deep_fm_input_fn(batch_size=16)
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
    print("feature_fm_1")
    demo(default_feature_cols_fn(params)["fm_feature_columns"][0])

