#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

def default_feature_cols_fn(params):
    field_names = ["0", "1"]
    col_dict = {}
    field_with_feature = [
        ["field0_feature_%d" % (d + 1) for d in range(2)],
        ["field1_feature_%d" % (d + 1) for d in range(2)]
    ]
    boundaries = [0.01 * i - 1 for i in range(200)]
    embedding_size = params.get("embedding_size", 64)
    for i in range(len(field_names)):
        source_feature_col = []
        for name in field_with_feature[i]:
            numeric_column = tf.feature_column.numeric_column(name)
            bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
            source_feature_col.append(bucketized_column)
        for j in range(len(field_names)):
            embedding_columns = [tf.feature_column.embedding_column(c, embedding_size) for c in source_feature_col]
            col_dict["ffm_field%d_to_%d_feature_columns" % (i, j)] = embedding_columns
    return col_dict

# def default_feature_cols_fn(params):
#     feature_cols = []
#     feature_names = ["field0_feature_%d" % (d+1) for d in range(2)]
#     boundaries = [0.01 * i - 1 for i in range(200)]
#     for name in feature_names:
#         numeric_column = tf.feature_column.numeric_column(name)
#         bucketized_column = tf.feature_column.bucketized_column(numeric_column, boundaries)
#         # categorical_column = tf.feature_column.categorical_column_with_identity(bucketized_column, len(boundaries))
#         embedding_size = params.get("embedding_size", 64)
#         embedding_column = tf.feature_column.embedding_column(bucketized_column, embedding_size)
#         feature_cols.append(embedding_column)
#     return {"ffm_field0_to_1_feature_columns": feature_cols}


def build_model_fn(gen_model_params, feature_cols_fn):
    def _get_all_vec_cross(t1, t2):
        # t1: [batch_size, feature_num_1, embedding_size]
        # t1: [batch_size, feature_num_2, embedding_size]
        cross_matrix = tf.matmul(t1, t2, transpose_b=True)  # [batch_size, feature_num_1, feature_num_2]
        energy = tf.reduce_sum(cross_matrix, axis=[1, 2])  # [batch_size]
        return energy

    def model_fn(features, labels, mode, params):
        feature_cols = feature_cols_fn(gen_model_params)
        field_names = ["0", "1"]
        cross_logits = []
        for i in range(len(field_names)):
            for j in range(len(field_names)):
                if j < i:
                    continue
                i_field_name = "ffm_field%d_to_%d_feature_columns" % (i, j)
                j_field_name = "ffm_field%d_to_%d_feature_columns" % (j, i)
                i_field_inputs = [
                    tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols[i_field_name]
                ]
                i_stack_vectors = tf.stack(i_field_inputs, axis=1)
                j_field_inputs = [
                    tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols[j_field_name]
                ]
                j_stack_vectors = tf.stack(j_field_inputs, axis=1)

                this_logit = _get_all_vec_cross(i_stack_vectors, j_stack_vectors)
                cross_logits.append(this_logit)
        # logits = tf.reduce_sum(tf.concat(cross_logits, axis=1))  # [batch]
        logits = tf.reduce_sum(tf.stack(cross_logits, axis=1), axis=1)
        logits_for_head = tf.expand_dims(logits, axis=1)  # [batch * 1]

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

        from tensorflow.estimator import BinaryClassHead
        binary_head = BinaryClassHead()

        # labels = tf.expand_dims(labels, axis=1)
        return binary_head.create_estimator_spec(
            features=features,
            mode=mode,
            logits=logits_for_head,
            labels=labels,
            optimizer=optimizer,
            trainable_variables=tf.compat.v1.trainable_variables()
        )

    # def model_fn(features, labels, mode, params):
    #     feature_cols = feature_cols_fn(gen_model_params)
    #     inputs = [tf.compat.v1.feature_column.input_layer(features, i) for i in feature_cols["ffm_field0_to_1_feature_columns"]]
    #
    #     stack_vectors = tf.stack(inputs, axis=1)  # batch * feature_num * embed_size
    #
    #     sum_and_square = tf.reduce_sum(tf.square(tf.reduce_sum(stack_vectors, axis=1)), axis=1)  # batch
    #     square_and_sum = tf.reduce_sum(tf.reduce_sum(tf.square(stack_vectors), axis=1), axis=1)  # batch
    #     logits = 0.5 * (sum_and_square - square_and_sum)
    #
    #     print("*"*20)
    #     # print(i_stack_vectors)
    #     # print(j_stack_vectors)
    #     # print(this_logit)
    #     print(tf.compat.v1.trainable_variables())
    #
    #     logits_for_head = tf.expand_dims(logits, axis=1)  # [batch * 1]
    #
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=params['beta1'])
    #     optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    #
    #     from tensorflow.estimator import BinaryClassHead
    #     binary_head = BinaryClassHead()
    #
    #     return binary_head.create_estimator_spec(
    #         features=features,
    #         mode=mode,
    #         logits=logits_for_head,
    #         labels=labels,
    #         optimizer=optimizer,
    #         trainable_variables=tf.compat.v1.trainable_variables()
    #     )

    return model_fn


if __name__ == "__main__":
    import inputs

    train_ds = inputs.test_ffm_input_fn(batch_size=4)
    example_batch = next(iter(train_ds))
    example_feature = example_batch[0]
    example_label = example_batch[1]

    def demo(feature_column):
        feature_layer = tf.keras.layers.DenseFeatures(feature_column)
        print(feature_layer(example_feature).numpy())


    print("=" * 10 + "demo feature data" + "=" * 10)
    params = {"embedding_size": 4}

    print("label")
    print(example_label)
    print("raw_feature")
    print(example_feature)
    # print("feature_1")
    # demo(default_feature_cols_fn(params)["ffm_field1_to_0_feature_columns"][0])
    print("feature_2")
    demo(default_feature_cols_fn(params)["ffm_field0_to_1_feature_columns"][1])
