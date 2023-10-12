# import ridge regression from sklearn library
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np
from utils.dataloader import dataloader
from model.feedforward_nn import get_class
from sklearn.metrics import accuracy_score

split = 0.2
random_state = 42
x_train, y_train, x_validation, y_validation, x_test = dataloader(validation_size=split, random_state=random_state)

# Train the model
ridgeR = Ridge(alpha = 15)
ridgeR.fit(x_train, y_train)
y_pred = ridgeR.predict(x_validation)
# calculate mean square error
mean_squared_error_ridge = np.mean((y_pred - y_validation)**2)
print("RIDGE")
print(mean_squared_error_ridge)
print(y_pred)
print('Validation accuracy:' + str(accuracy_score((get_class(y_pred)), y_validation)))


LassoR = Lasso(alpha = 15)
LassoR.fit(x_train, y_train)
y_pred = LassoR.predict(x_validation)
# calculate mean square error
mean_squared_error_Lasso = np.mean((y_pred - y_validation)**2)
print("LASSO")
print(mean_squared_error_Lasso)
print(y_pred)
print('Validation accuracy:' + str(accuracy_score((get_class(y_pred)), y_validation)))

ENetR = ElasticNet(alpha = 20, l1_ratio= 0.5)
ENetR.fit(x_train, y_train)
y_pred = ENetR.predict(x_validation)
# calculate mean square error
mean_squared_error_ENet = np.mean((y_pred - y_validation)**2)
print("ENET")
print(mean_squared_error_ENet)
print(y_pred)
print('Validation accuracy:' + str(accuracy_score((get_class(y_pred)), y_validation)))


#print('Validation accuracy 1:' + accuracy_class_1((get_class(y_pred)), y_validation))
#print('Validation accuracy 0:' + accuracy_class_0((get_class(y_pred)), y_validation))



# get ridge coefficient and print them
#ridge_coefficient = pd.DataFrame()
#ridge_coefficient["Columns"] = x_train.columns
#ridge_coefficient['Coefficient Estimate'] = pd.Series(ridgeR.coef_)
#print(ridge_coefficient)
