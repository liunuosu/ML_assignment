import tensorflow as tf
from utils.metrics import accuracy
from tensorflow.python.keras.layers import Dense


class FeedForward(tf.Module):
    def __init__(self, num_features, num_classes, seed=42):
        super(FeedForward, self).__init__()
        self.d1 = Dense(158, activation=tf.nn.relu)
        self.d2 = Dense(32, activation=tf.nn.relu)
        self.d3 = Dense(num_classes, activation=None)

    def forward(self, x):
        # Compute the model output
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return tf.nn.softmax(x), x  # Probabilities and logits


def get_class(y_prob):
    return tf.argmax(y_prob, axis=1)


class FeedForwardModel:
    def __init__(self, num_epochs=50, batch_size=1024, weight_seed=42, pathdir=None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_seed= weight_seed
        self.pathdir = pathdir

        self.model = None

        self.best_accuracy = 0
        self.best_epoch = 0

    def fit(self, x_train, y_train, x_validation, y_validation):

        self.model = FeedForward(num_features=x_train.shape[-1], num_classes=2, seed=self.weight_seed)

        checkpoint = tf.train.Checkpoint(model=self.model)
        manager = tf.train.CheckpointManager(checkpoint, directory=self.pathdir, max_to_keep=self.num_epochs)

        num_train_samples, n_features = x_train.shape
        classifier_opt = tf.optimizers.Adam()  # Add learning rate?

        for epoch in range(self.num_epochs):
            shuffled_ids = [i for i in range(num_train_samples)]

            for i in range(num_train_samples // self.batch_size):
                batch_ids = shuffled_ids[self.batch_size * i: self.batch_size * (i + 1)]
                #print(x_train)
                batch_features = x_train[batch_ids].astype('float32')
                #print(y_train)
                batch_labels = y_train[batch_ids].astype('float32')

                with tf.GradientTape() as tape:
                    classifier_pred, classifier_logits = self.model.forward(batch_features)

                    loss_classifier = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels.astype('float32'),
                                                                logits=classifier_logits))
                gradients = tape.gradient(loss_classifier, self.model.trainable_variables)
                classifier_opt.apply_gradients(zip(gradients, self.model.trainable_variables))

                if i == (num_train_samples // self.batch_size - 1): # == 0 and i != 0:
                    y_prob, pred_logits = self.model.forward(
                        x_train.astype('float32'))

                    train_accuracy = accuracy(get_class(y_prob),
                                              tf.argmax(y_train, axis=1))

                    y_prob_validation, pred_logits_validation = self.model.forward(
                        x_validation.astype('float32'))

                    validation_accuracy = accuracy(get_class(y_prob_validation),
                                                   tf.argmax(y_validation, axis=1))

                    print("(Training Classifier) epoch %d; training accuracy: %f; "
                          "validation accuracy: %f; batch classifier loss: %f" % (
                        epoch+1, train_accuracy, validation_accuracy, loss_classifier))

                    if validation_accuracy > self.best_accuracy + 0.0001:
                        manager.save()
                        self.best_accuracy = validation_accuracy
                        self.best_epoch = epoch+1

        checkpoint.restore(manager.latest_checkpoint)
        print("(Training Classifier) best epoch %d; best validation accuracy %f" % (
            self.best_epoch, self.best_accuracy))
        return self

    def predict(self, x_test):
        y_prob_test, pred_logits_test = self.model.forward(
            x_test.astype('float32'))
        return get_class(y_prob_test)
