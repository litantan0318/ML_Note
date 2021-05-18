import math
import tensorflow as tf
import random


def generate_category_feature_by_label(label, amplify=1.0, disturb=1.0, noise=0.0):
    if random.random() < noise:
        label = random.randint(0, 1)
    true_feature = label * amplify
    disturb_feature = random.gauss(true_feature, disturb)
    return disturb_feature


def generate_category_feature_randomly(label, amplify=1.0, disturb=1.0, noise=0.0):
    if random.random() < noise:
        label = random.randint(0, 1)
    true_feature = label * amplify
    disturb_feature = random.gauss(true_feature, disturb)
    return disturb_feature


def generate_circle_feature(r=3, noise=0.0, range_sigma=2, dimension=4):
    features = []
    this_r = 0
    for i in range(dimension):
        this_dimension = random.gauss(0, range_sigma)
        features.append(this_dimension)
        this_r += this_dimension ** 2
    if this_r < r ** 2:
        label = 1
    else:
        label = 0
    # noise
    if random.random() < noise:
        label = random.randint(0, 1)
    return features, label


def generate_attention_feature(his_num=20, noise=0.2, his_len_rate=0.5, len_sigma=5.0):
    histories = []
    query = random.randint(0, 1000)
    min_distance = 1000
    this_len = random.gauss(his_len_rate * his_num, sigma=len_sigma)
    for i in range(his_num):
        if i > this_len:
            histories.append(0)
            continue
        this_his = random.randint(0, 1000)
        this_distance = abs(this_his - query)
        min_distance = min(this_distance, min_distance)
        histories.append(this_his)
    prob = math.exp(-1 * (min_distance / 10))
    label = 1 if random.random() < prob else 0
    # noise
    if random.random() < noise:
        label = random.randint(0, 1)
    features = [query, histories]
    return features, label


def generate_easy_attention_feature(his_num=100, noise=0.2):
    histories = []
    # query = random.randint(0, 500)
    query = random.randint(0, 10)
    match = 0
    for i in range(his_num):
        # this_his = random.randint(0, 500)
        this_his = random.randint(0, 10)
        if query + 1 >= this_his >= query - 1:
            match = 1
        histories.append(this_his)
    label = match
    # noise
    if random.random() < noise:
        label = random.randint(0, 1)
    features = [query, histories]
    return features, label


def test_fm_input_fn(batch_size=128, noise=0.2, disturb=0.5):
    def rdd_generator():
        while True:
            yield_list = []
            label = float(random.randint(0, 1))
            yield_list.append(label)
            feature_list = [
                generate_category_feature_by_label(label, noise=noise, disturb=disturb),
                generate_category_feature_by_label(label, noise=noise, disturb=disturb),
                generate_category_feature_by_label(label, noise=noise, disturb=disturb),
                generate_category_feature_by_label(label, noise=noise, disturb=disturb),
            ]
            yield_list = yield_list + feature_list
            yield_str = "\t".join([str(i) for i in yield_list])
            yield yield_str

    def _parse_line(line):
        features = {}
        line_list = tf.compat.v1.string_split([line], '\t')
        line_list_dense = tf.reshape(line_list.values, (-1,))
        line_list_dense = tf.compat.v1.string_to_number(line_list_dense)
        feature_names = ["feature_%d" % i for i in range(4 + 1)]
        label = line_list_dense[0]
        for index, name in enumerate(feature_names):
            if index == 0:
                continue
            features[name] = line_list_dense[index]
        return features, label

    dataset = tf.data.Dataset.from_generator(
        rdd_generator, output_types=tf.string)

    dataset = dataset.map(_parse_line)
    dataset = dataset.batch(batch_size)

    return dataset


def test_ffm_input_fn(batch_size=128, noise=0.2, disturb=0.5):
    output_signature = [tf.TensorSpec(shape=(), dtype=tf.int32)]
    for _ in range(2):
        output_signature.append(tf.TensorSpec(shape=(), dtype=tf.float32))
    for _ in range(2):
        output_signature.append(tf.TensorSpec(shape=(), dtype=tf.float32))

    def rdd_generator():
        while True:
            yield_list = []
            label = float(random.randint(0, 1))
            yield_list.append(label)
            feature_list = []
            for _ in range(2):
                feature_list.append(
                    generate_category_feature_by_label(label, noise=noise, disturb=disturb)
                )
            for _ in range(2):
                feature_list.append(
                    generate_category_feature_by_label(label, noise=noise, disturb=disturb)
                )

            yield tuple(yield_list + feature_list)

    def _parse_line(*line):
        features = {}
        tensors = list(line)
        label = tensors.pop(0)
        field1_feature_names = ["field0_feature_%d" % (i+1) for i in range(2)]
        for index, name in enumerate(field1_feature_names):
            features[name] = line[index + 1]

        field2_feature_names = ["field1_feature_%d" % (i+1) for i in range(2)]
        for index, name in enumerate(field2_feature_names):
            features[name] = line[index + len(field1_feature_names) + 1]
        return features, label

    dataset = tf.data.Dataset.from_generator(
        rdd_generator, output_signature=tuple(output_signature))

    dataset = dataset.map(_parse_line)
    dataset = dataset.batch(batch_size)

    return dataset


def test_categorys_input_fn(batch_size=128, noise=0.2, disturb=0.5, feature_name="test_feature"):
    output_signature = [tf.TensorSpec(shape=(), dtype=tf.int32)]
    for _ in range(4):
        output_signature.append(tf.TensorSpec(shape=(), dtype=tf.float32))

    def rdd_generator():
        while True:
            yield_list = []
            label = float(random.randint(0, 1))
            yield_list.append(label)
            feature_list = []
            for _ in range(4):
                feature_list.append(
                    generate_category_feature_by_label(label, noise=noise, disturb=disturb)
                )

            yield tuple(yield_list + feature_list)

    def _parse_line(*line):
        features = {}
        tensors = list(line)
        label = tensors.pop(0)
        feature_names = [feature_name + "_" + str(i + 1) for i in range(4)]
        for index, name in enumerate(feature_names):
            features[name] = line[index + 1]
        return features, label

    dataset = tf.data.Dataset.from_generator(
        rdd_generator, output_signature=tuple(output_signature))

    dataset = dataset.map(_parse_line)
    dataset = dataset.batch(batch_size)

    return dataset


# deep_fm_feature
def test_deep_fm_input_fn(batch_size=128, noise=0.2, disturb=0.5):
    return test_categorys_input_fn(
        batch_size=batch_size,
        noise=noise,
        disturb=disturb,
        feature_name="deep_fm_feature"
    )


# dcn_deep_cross_feature
def test_dcn_input_fn(batch_size=128, noise=0.2, disturb=0.5):
    return test_categorys_input_fn(
        batch_size=batch_size,
        noise=noise,
        disturb=disturb,
        feature_name="dcn_deep_cross_feature"
    )


def test_pnn_input_fn(batch_size=128, noise=0.2, disturb=0.5):
    return test_categorys_input_fn(
        batch_size=batch_size,
        noise=noise,
        disturb=disturb,
        feature_name="pnn_feature"
    )


def test_nfm_input_fn(batch_size=128, noise=0.2, disturb=0.5):
    return test_categorys_input_fn(
        batch_size=batch_size,
        noise=noise,
        disturb=disturb,
        feature_name="nfm_feature"
    )


def test_afm_input_fn(batch_size=128, noise=0.2, disturb=0.5):
    return test_categorys_input_fn(
        batch_size=batch_size,
        noise=noise,
        disturb=disturb,
        feature_name="afm_feature"
    )


def test_mlr_input_fn(batch_size=128):
    output_signature = [tf.TensorSpec(shape=(), dtype=tf.int32)]
    for _ in range(4):
        output_signature.append(tf.TensorSpec(shape=(), dtype=tf.float32))

    def rdd_generator():
        while True:
            yield_list = []
            feature_list, label = generate_circle_feature()
            yield_list.append(label)
            yield tuple(yield_list + feature_list)

    def _parse_line(*line):
        features = {}
        tensors = list(line)
        label = tensors.pop(0)
        feature_names = ["mlr_feature" + "_" + str(i + 1) for i in range(4)]
        for index, name in enumerate(feature_names):
            features[name] = line[index + 1]
        return features, label

    dataset = tf.data.Dataset.from_generator(
        rdd_generator, output_signature=tuple(output_signature))

    dataset = dataset.map(_parse_line)
    dataset = dataset.batch(batch_size)

    return dataset


# def test_din_input_fn(batch_size=128, his_num=20, his_len_rate=0.5, noise=0.2, len_sigma=5):
def test_din_input_fn(batch_size=128, his_num=2, his_len_rate=1.0, noise=0.5, len_sigma=0.0):
    output_signature = [tf.TensorSpec(shape=(), dtype=tf.int32),
                        tf.TensorSpec(shape=(), dtype=tf.int32),
                        tf.TensorSpec(shape=(his_num,), dtype=tf.int32)]

    def rdd_generator():
        while True:
            yield_list = []
            feature_list, label = generate_easy_attention_feature(
                his_num=his_num,
                noise=noise,
                # his_len_rate=his_len_rate,
                # len_sigma=len_sigma
                )
            yield_list.append(label)
            yield tuple(yield_list + feature_list)

    def _parse_line(*line):
        features = {}
        tensors = list(line)
        label = tensors.pop(0)
        feature_names = ["din_query_feature", "din_his_feature"]
        for index, name in enumerate(feature_names):
            features[name] = line[index + 1]
        return features, label

    dataset = tf.data.Dataset.from_generator(
        rdd_generator, output_signature=tuple(output_signature))

    dataset = dataset.map(_parse_line)
    dataset = dataset.batch(batch_size)

    return dataset

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=5)

    # ds = test_din_input_fn(batch_size=4, noise=0.0, disturb=0.0)
    ds = test_din_input_fn(batch_size=4)
    count = 0
    for i in ds:
        if count > 100:
            break
        count += 1
        tf.print(i)

    # ds, label = generate_circle_feature()
    # print(ds)
    # print(sum([i**2 for i in ds]))
    # print(label)