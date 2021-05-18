#!/usr/bin/env python
# coding=utf-8

#  author: tantanli
#  2021.04.25
#  afm network

import tensorflow as tf
from tensorflow import keras


class CustomerizeAttentionLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomerizeAttentionLayer, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

def default_feature_cols_fn(params):
    feature_map = {}
    feature_names = ["din_query_feature", "din_his_feature"]
    # for name in feature_names:
    #     numeric_column = tf.feature_column.numeric_column(name)
    #     bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
    #     embedding_size = params.get("embedding_size", 64)
    #     embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
    #     feature_cols.append(embedding_column)
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
    def model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        # din_his_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_his_feature_columns"]]
        din_raw_query_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_query_feature_columns"]]
        din_his_index_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_his_index_feature_columns"]]
        # get history embeddings
        din_his_inputs = []
        for i in din_his_index_inputs:
            this_embedding_layer = tf.keras.layers.Embedding(11, params.get("embedding_size", 64), input_length=2)
            this_his_embeddings = this_embedding_layer(i)  # []
            din_his_inputs.append(this_his_embeddings)

        din_concat_query_inputs = tf.concat(din_raw_query_inputs, axis=1)  # [batch_size, sum(embedding)]
        # #####################  attention part  #################################
        # query_tensor = tf.keras.layers.Dense(
        #     units=params.get("key_dimension", 32),
        #     activation="relu")\
        #     (din_concat_query_inputs)  # [batch_size, query_embedding]
        # # just one head attention now
        # query_tensor = tf.expand_dims(query_tensor, axis=1)  # [batch_size, 1, query_embedding]
        # attention_outputs = []
        # attention_scores = []
        # attention_masks = []
        # for index, this_value_tensor in enumerate(din_his_inputs):
        #     # this_value_tensor: [batch_size, num_his, value_embedding]
        #     this_attention_layer = tf.keras.layers.Attention()
        #     this_key_tensor = tf.keras.layers.Dense(
        #         units=params.get("key_dimension", 32),
        #         activation="relu")\
        #         (this_value_tensor)  # [batch_size, num_his, query_embedding]
        #     inputs = [query_tensor, this_value_tensor, this_key_tensor]
        #     this_mask = tf.not_equal(din_his_index_inputs[index], 0)  # [batch_size, num_his]
        #     masks = [None, this_mask]
        #     this_attention_output, this_attention_scores = this_attention_layer(
        #         inputs,
        #         mask=masks,
        #         return_attention_scores=True
        #     )  # [batch_size, 1, value_embedding] for output
        #     this_attention_output = tf.squeeze(this_attention_output, axis=1)  # [batch_size, value_embedding]
        #     this_attention_scores = tf.squeeze(this_attention_scores, axis=1)  # [batch_size, his_num]
        #     attention_outputs.append(this_attention_output)
        #     attention_scores.append(this_attention_scores)
        #     attention_masks.append(this_mask)

        # # ####### baseline mean pooling ##################
        attention_outputs = []
        for index, this_value_tensor in enumerate(din_his_inputs):
            # this_value_tensor: [batch_size, num_his, value_embedding]
            this_attention_output = tf.reduce_mean(this_value_tensor, axis=1)
            attention_outputs.append(this_attention_output)
        # # ##################################################

        # ###### easy_attention ##################
        # query_tensor = din_concat_query_inputs  # [batch_size, query_embedding]
        # # just one head attention now
        # query_tensor = tf.expand_dims(query_tensor, axis=1)  # [batch_size, 1, query_embedding]
        # attention_outputs = []
        # attention_scores = []
        # attention_masks = []
        # for index, this_value_tensor in enumerate(din_his_inputs):
        #     # this_value_tensor: [batch_size, num_his, value_embedding]
        #     this_attention_layer = tf.keras.layers.Attention()
        #     this_key_tensor = this_value_tensor  # [batch_size, num_his, query_embedding]
        #     inputs = [query_tensor, this_value_tensor, this_key_tensor]
        #     this_mask = tf.not_equal(din_his_index_inputs[index], 0)  # [batch_size, num_his]
        #     # masks = [None, this_mask]
        #     masks = None
        #     this_attention_output, this_attention_scores = this_attention_layer(
        #         inputs,
        #         mask=masks,
        #         return_attention_scores=True
        #     )  # [batch_size, 1, value_embedding] for output
        #     this_attention_output = tf.squeeze(this_attention_output, axis=1)  # [batch_size, value_embedding]
        #     this_attention_scores = tf.squeeze(this_attention_scores, axis=1)  # [batch_size, his_num]
        #     attention_outputs.append(this_attention_output)
        #     attention_scores.append(this_attention_scores)
        #     # attention_masks.append(this_mask)
        # ################################################

        attention_tensor = tf.concat(attention_outputs, axis=1)  # [batch_size, sum(value_embedding)]


        ### mlp part
        if params.get("use_mlp", True):
            mlp_input = tf.concat([din_concat_query_inputs, attention_tensor], axis=1)  # [batch_size, sum(all_embedding)]
            x = mlp_input
            for unit in params.get("mlp_layers", [64, 64]):
                this_layer = tf.keras.layers.Dense(units=unit, activation="relu")
                x = this_layer(x)

            logits = tf.keras.layers.Dense(units=1, activation="linear")(x)
            logits_for_head = logits
        else:
            logits = tf.reduce_sum(din_concat_query_inputs * attention_tensor, axis=-1)
            logits_for_head = tf.expand_dims(logits, axis=-1)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
        # optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.0, beta_2=0.0)
        # optimizer = tf.keras.optimizers.Adagrad(learning_rate=params['learning_rate'])
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

        from tensorflow.estimator import BinaryClassHead
        binary_head = BinaryClassHead()
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #
        #     query_tensor = tf.stack(features["din_query_feature"], axis=0)  # [batch_size]
        #     his_tensor = tf.stack(features["din_his_feature"], axis=0)  #[batch_size, his_len]
        #     match = tf.equal(tf.expand_dims(query_tensor, axis=-1) - his_tensor, 0)
        #     match_score = tf.boolean_mask(attention_scores[0], match)
        #     print("#"*20)
        #     print(attention_scores[0])
        #     print(match)
        #     print(match_score)
        #     print(tf.stack(features["din_query_feature"], axis=0))
        #     prediction = {
        #         "din_query_feature": tf.stack(features["din_query_feature"], axis=0),
        #         "din_his_feature": tf.stack(features["din_his_feature"], axis=0),
        #         "attention_scores": tf.squeeze(attention_scores),
        #         "match": tf.reduce_sum(tf.cast(match, tf.int32), axis=-1),
        #         # "match_score": match_score,
        #         "logits": logits,
        #         "scores": logits,
        #         # "label": labels
        #     }
        #     return tf.estimator.EstimatorSpec(
        #         mode=mode,
        #         predictions=prediction
        #     )
        return binary_head.create_estimator_spec(
            features=features,
            mode=mode,
            logits=logits_for_head,
            labels=labels,
            optimizer=optimizer,
            trainable_variables=tf.compat.v1.trainable_variables()
        )

    def test_model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        # din_his_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_his_feature_columns"]]
        din_raw_query_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_query_feature_columns"]]
        # din_his_index_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_his_index_feature_columns"]]
        din_hid_emb_inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["din_his_emb_feature_columns"]]
        # get history embeddings

        din_concat_query_inputs = tf.concat(din_raw_query_inputs, axis=1)  # [batch_size, sum(embedding)]
        din_concat_his_inputs = tf.concat(din_hid_emb_inputs, axis=1)  # [batch_size, sum(embedding)]

        attention_tensor = din_concat_his_inputs

        logits = tf.reduce_sum(din_concat_query_inputs * attention_tensor, axis=-1)
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


