from utils.dataloader import dataloader
from model.feedforward_nn import FeedForwardModel
from model.RNN import RNN_model
import time
from pathlib import Path
import datetime

now = datetime.datetime.now()
model_filename = now.strftime("%Y-%m-%d_%H-%M-%S")
pathdir = Path("weights", model_filename)
pathdir.mkdir(parents=True, exist_ok=True)

n_epoch = 100
batch_size = 512
random_state = 42
split = 0.1
#learning_rate = 0.0005

x_train, y_train, x_validation, y_validation, x_test = dataloader(validation_size=split, random_state=random_state)

# hyperparameter tuning: n_epochs, n_layers, n_neurons, learning rates, different optimizers
# validation set size? now its 0.2 but can do cross-validation
# #stratified sampling, k-fold cross validation
# implement different metrics

model = FeedForwardModel(num_epochs=n_epoch, batch_size=batch_size, pathdir=pathdir)
#model = RNN_model(num_epochs=n_epoch, batch_size=batch_size, pathdir=pathdir)
start = time.time()
model.fit(x_train, y_train, x_validation, y_validation)
end = time.time()
total = end - start
print(f"Training completed in {total} seconds!")
prediction = model.predict(x_test)
#print(prediction[0:100])

#print(x_train.shape)
#print(x_validation.shape)
#print(y_train.shape)
#print(y_validation.shape)