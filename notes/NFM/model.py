#!/usr/bin/env python
# coding=utf-8

#  author: tantanli
#  2021.04.23
#  nfm network

import tensorflow as tf
from tensorflow import keras


class BiInteractionLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super(BiInteractionLayer, self).__init__()

    def call(self, inputs, **kwargs):
        # input shape: [batch_size, feature_num, embedding_size]
        output_tensor = tf.square(tf.reduce_sum(inputs, axis=1)) - tf.reduce_sum(tf.square(inputs), axis=1)
        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]  # [batch_size, embedding_size]


def default_feature_cols_fn(params):
    feature_cols = []
    feature_names = ["nfm_feature_%d" % (d + 1) for d in range(4)]
    boundaries = [0.01 * i - 1 for i in range(200)]
    for name in feature_names:
        numeric_column = tf.feature_column.numeric_column(name)
        bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
        # categorical_column = tf.feature_column.categorical_column_with_identity(bucketized_column, len(boundaries))
        embedding_size = params.get("embedding_size", 64)
        embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
        feature_cols.append(embedding_column)
    return {"nfm_feature_columns": feature_cols}


def build_model_fn(gen_model_params, feature_cols_fn):
    def model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        nfm_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["nfm_feature_columns"]]

        ##############################################################################
        # linear
        ##############################################################################
        concat_vectors = tf.concat(nfm_inputs, axis=1)  # [batch_size, embedding_size * feature_num)]
        linear_layer = tf.keras.layers.Dense(1, activation="linear")
        linear_output = linear_layer(concat_vectors)  # [batch_size, 1]

        ##############################################################################
        #  bi interact product
        ##############################################################################
        stack_vectors = tf.stack(nfm_inputs, axis=1)  # [batch_size, feature_num, embedding_size)]
        x = stack_vectors
        bi_interact_layer = BiInteractionLayer()
        x = bi_interact_layer(x)  # [batch_size, embedding_size]
        for i in params.get("deep_layers", [64, 64]):
            x = tf.keras.layers.Dense(units=i, activation="relu")(x)
        bi_interact_output = tf.keras.layers.Dense(1, activation="linear")(x)  # [batch_size, 1]

        ##############################################################################
        # merge
        ##############################################################################

        # logits = tf.keras.layers.Dense(1, activation="sigmoid")(linear_output + bi_interact_output)
        logits = linear_output + bi_interact_output
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
    train_ds = inputs.test_nfm_input_fn(batch_size=5)
    example_batch = next(iter(train_ds))[0]

    def demo(feature_column):
        feature_layer = tf.keras.layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())

    print("="*10 + "demo feature data" + "="*10)
    params = {"embedding_size": 4}

    print("raw")
    print(next(iter(train_ds)))
    print("feature_1")
    demo(default_feature_cols_fn(params)["nfm_feature_columns"][0])
    print("feature_2")
    demo(default_feature_cols_fn(params)["nfm_feature_columns"][1])

