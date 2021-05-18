#!/usr/bin/env python
# coding=utf-8

#  author: tantanli
#  2021.04.25
#  afm network

import tensorflow as tf
from tensorflow import keras


def default_feature_cols_fn(params):
    feature_cols = []
    feature_names = ["mlr_feature_%d" % (d + 1) for d in range(4)]
    boundaries = [0.01 * i - 1 for i in range(200)]
    for name in feature_names:
        numeric_column = tf.feature_column.numeric_column(name)
        # bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
        # # categorical_column = tf.feature_column.categorical_column_with_identity(bucketized_column, len(boundaries))
        # embedding_size = params.get("embedding_size", 64)
        # embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
        # feature_cols.append(embedding_column)
        feature_cols.append(numeric_column)
    return {"mlr_feature_columns": feature_cols}


def build_model_fn(gen_model_params, feature_cols_fn):
    def model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        mlr_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["mlr_feature_columns"]]

        concat_inputs = tf.concat(mlr_inputs, axis=1)  # [batch_size, embedding_size * feature_num]
        lr_array = []
        weight_array = []
        for i in range(params.get("lr_nums", 4)):
            this_lr_output = tf.keras.layers.Dense(
                units=1,
                activation="sigmoid",
                use_bias=True
            )(concat_inputs)
            lr_array.append(this_lr_output)  # lr nums array of [batch_size, 1]

            this_weight = tf.keras.layers.Dense(
                units=1,
                activation="linear",
                use_bias=False
            )(concat_inputs)
            weight_array.append(tf.math.exp(this_weight))  # lr nums array of [batch_size, 1]
        lr_tensor = tf.squeeze(tf.stack(lr_array, axis=1))  # [batch_size, lr_nums]
        weight_tensor = tf.squeeze(tf.stack(weight_array, axis=1))  # [batch_size, lr_nums]
        weight_tensor = weight_tensor / tf.reduce_sum(weight_tensor, axis=1, keepdims=True)

        output_tensor = tf.reduce_sum(lr_tensor * weight_tensor, axis=1, keepdims=True)  # [batch_size, 1]

        logits = output_tensor
        logits_for_head = logits

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

        labels = tf.expand_dims(labels, axis=-1)
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
    train_ds = inputs.test_mlr_input_fn(batch_size=5)
    example_batch = next(iter(train_ds))[0]

    def demo(feature_column):
        feature_layer = tf.keras.layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())

    print("="*10 + "demo feature data" + "="*10)
    params = {"embedding_size": 4}

    print("raw")
    print(next(iter(train_ds))[1])
    print("feature_1")
    demo(default_feature_cols_fn(params)["mlr_feature_columns"][0])
    print("feature_2")
    demo(default_feature_cols_fn(params)["mlr_feature_columns"][1])

