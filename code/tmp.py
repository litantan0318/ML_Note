import tensorflow as tf

a = tf.constant(
    [
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3], [4, 5, 6]],
    ],
    dtype=tf.int32
)
b = [
        [[1, 0, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 1]],
    ]
print(tf.boolean_mask(a, tf.cast(b, tf.bool)))

