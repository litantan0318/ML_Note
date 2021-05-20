#!/usr/bin/env python
# coding=utf-8

#  author: tantanli
#  2021.04.25
#  afm network

import tensorflow as tf
from tensorflow import keras




def default_feature_cols_fn(params):
    feature_map = {}
    # query
    query_embedding_size = params.get("embedding_size", 64)
    query_category_column = tf.feature_column.categorical_column_with_identity("din_query_feature", num_buckets=11)
    query_embedding_column = tf.feature_column.embedding_column(query_category_column, query_embedding_size)
    feature_map["din_query_feature_columns"] = [query_embedding_column]
    # # his
    his_numeric_column = tf.feature_column.numeric_column("din_his_feature", dtype=tf.int32, shape=(params.get("his_len", 50),))
    his_embedding_size = params.get("embedding_size", 64)
    his_category_column = tf.feature_column.categorical_column_with_identity("din_his_feature", num_buckets=11)
    his_embedding_column = tf.feature_column.embedding_column(his_category_column, his_embedding_size)
    feature_map["din_his_emb_feature_columns"] = [his_embedding_column]
    feature_map["din_his_index_feature_columns"] = [his_numeric_column]
    return feature_map


def build_model_fn(gen_model_params, feature_cols_fn):

    def test_model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        # din_his_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_his_feature_columns"]]
        din_raw_query_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_query_feature_columns"]]
        # din_his_index_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_his_index_feature_columns"]]
        din_hid_emb_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_his_emb_feature_columns"]]
        # get history embeddings

        din_concat_query_inputs = tf.concat(din_raw_query_inputs, axis=1)  # [batch_size, sum(embedding)]
        din_concat_his_inputs = tf.concat(din_hid_emb_inputs, axis=1)  # [batch_size, sum(embedding)]

        logits = tf.reduce_sum(din_concat_query_inputs * din_concat_his_inputs, axis=-1)
        logits_for_head = tf.expand_dims(logits, axis=-1)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
        # optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.0, beta_2=0.0)
        # optimizer = tf.keras.optimizers.Adagrad(learning_rate=params['learning_rate'])
        optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])
        # optimizer = tf.keras.optimizers.Adadelta(learning_rate=params['learning_rate'])

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

    return test_model_fn


if __name__ == "__main__":
    import inputs
    train_ds = inputs.test_din_input_fn(batch_size=5, his_num=1, his_len_rate=1.0, noise=0.0, len_sigma=0)
    example_batch = next(iter(train_ds))[0]

    def demo(feature_column):
        feature_layer = tf.keras.layers.DenseFeatures(feature_column)
        print(feature_layer(example_batch).numpy())

    print("="*10 + "demo feature data" + "="*10)
    params = {"embedding_size": 5, "his_len": 1}

    print("raw")
    print(next(iter(train_ds)))
    print("feature_1")
    # demo(default_feature_cols_fn(params)["din_his_feature_columns"][0])
    demo(default_feature_cols_fn(params)["din_his_index_feature_columns"][0])
    demo(default_feature_cols_fn(params)["din_his_emb_feature_columns"][0])
    print("feature_2")
    demo(default_feature_cols_fn(params)["din_query_feature_columns"][0])


