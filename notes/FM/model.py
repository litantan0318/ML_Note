#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf


def default_feature_cols_fn(params):
    feature_cols = []
    feature_names = ["feature_%d" % (d+1) for d in range(4)]
    boundaries = [0.01 * i - 1 for i in range(200)]
    for name in feature_names:
        numeric_column = tf.feature_column.numeric_column(name)
        bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
        # categorical_column = tf.feature_column.categorical_column_with_identity(bucketized_column, len(boundaries))
        embedding_size = params.get("embedding_size", 64)
        embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
        feature_cols.append(embedding_column)
    return {"fm_feature_columns": feature_cols}


def build_model_fn(gen_model_params, feature_cols_fn):
    def model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        assert "fm_feature_columns" in feature_cols
        inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["fm_feature_columns"]]

        # explanation of the logit function
        # for all target = sigma(xi*wi*wj*xj)|0<i<j<n
        # (sigma(wi*xi)) ^ 2 = sigma((wi*xi)^2)|0<i<n + sigma((wi*xj))|i!=j
        #                    = sigma((wi*xi)^2)|0<i<n + 2 * sigma((wi*xj))|0<i<j<n
        # so that target = 1/2 * ((sigma(wi*xi)) ^ 2 - sigma((wi*xi)^2))

        stack_vectors = tf.stack(inputs, axis=1)  # batch * feature_num * embed_size
        sum_and_square = tf.reduce_sum(tf.square(tf.reduce_sum(stack_vectors, axis=1)), axis=1)  # batch
        square_and_sum = tf.reduce_sum(tf.reduce_sum(tf.square(stack_vectors), axis=1), axis=1)  # batch
        logits = 0.5 * (sum_and_square - square_and_sum)
        logits_for_head = tf.expand_dims(logits, axis=1)  # batch * 1
        # batch_logit = tf.reduce_mean(logits)

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
        # optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])
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
    train_ds = inputs.test_fm_input_fn(batch_size=16)
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
    demo(default_feature_cols_fn(params)["fm_feature_columns"][0])
    print("feature_2")
    demo(default_feature_cols_fn(params)["fm_feature_columns"][1])
