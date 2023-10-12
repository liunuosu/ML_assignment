import tensorflow as tf

def accuracy(y_pred, y):
    check_equal = tf.cast(y_pred == y, tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    return acc_val

def accuracy_class_0(y_pred, y):
    mask = tf.cast(tf.equal(y, 0), tf.float32)
    class_correct = tf.reduce_sum(tf.cast(tf.equal(y_pred, y), tf.float32) * mask)
    class_total = tf.reduce_sum(mask)
    class_accuracy = class_correct / (class_total + 1e-10)
    return class_accuracy

def accuracy_class_1(y_pred, y):
    mask = tf.cast(tf.equal(y, 1), tf.float32)
    class_correct = tf.reduce_sum(tf.cast(tf.equal(y_pred, y), tf.float32) * mask)
    class_total = tf.reduce_sum(mask)
    class_accuracy = class_correct / (class_total + 1e-10)
    return class_accuracy
