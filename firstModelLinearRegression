import torch
from torch import nn # basic building blocks for graphs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

# prepare and load data
# build model
# fit the model to the data
# make predictions and evaluate model
# save and load model
# put it all together

# 1. prepare and load data
# y = mx + b
weight = 0.7 # m
bias = 0.3 # b

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1) # features
y = weight * X + bias

#print(X[:5], y[:5])

# split data into training and test sets
# course material -> traning set, Practice exam -> validation set, final exam -> test set
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split] # x up until the trainsplit y up until the trainsplit
X_test, y_test = X[train_split:], y[train_split:]

#print(len(X_train), len(y_train), len(X_test), len(y_test))

def plot_predictions(train_data = X_train, train_labels = y_train, test_data = X_test, 
                     test_labels = y_test, predictions = None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    #plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()

#plot_predictions()

# 2. build model
class LinearRegressionModel(nn.Module): # almost everything in Pytorch inherits from nn.Module
    def __init__(self):
        super().__init__() # inherit from nn.Module
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32)) # random initial weights
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32)) # random initial bias

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is input data
        return self.weights * x + self.bias # linear regression formula

# model building essentials
# torch.nn -> contains building blocks for neural networks
# torch.nn.Parameter -> what parameters should our model try to learn
# torch.nn.Module -> base class for all neural network modules, if subclass then override forward

# torch.optim -> where optimizers in PyTorch live -> helps with Gradient descent
# def forward() -> all nn.Module subclasses requires overwrite on forward. What happens in the forward computation

# Checking contents of Torch Model

# create a random seed
torch.manual_seed(42)
model_0 = LinearRegressionModel()

#print(list(model_0.parameters()))

# Making predictions using torch.inference_mode()
with torch.inference_mode():
    y_preds = model_0(X_test)

#print(y_preds)
#print(y_test)

plot_predictions(predictions = y_preds)

# 3. Fit model to data

# one way to measure how bad your model is doing is by using a loss function
# optimizer: takes account loss of model and adjusts model's parameters (weights) 

#print(list(model_0.parameters()))
#print(model_0.state_dict())

loss_fn = nn.L1Loss() # torch.mean(torch.abs(y_pred - y_test)) # MAE_Loss
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)

# building training loop in Pytorch & testing loop
epochs = 10000 # one loop through data

for epoch in range(epochs): # loop through data
    model_0.train() # set to training mode in Pytorch set # sets gradients to require gradients

    # forward pass
    y_pred = model_0(X_train)
    # calc loss
    loss = loss_fn(y_pred, y_train)
    # Optimizer zero grad
    optimizer.zero_grad()
    #Perform back propagation on the loss wiht respect to parameters of the model "think slope"
    loss.backward()
    # (perform gradient descent)
    optimizer.step()

    # Za TESTING CODE

    model_0.eval()
    with torch.inference_mode():
        #1. forward in testing still
        test_pred = model_0(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} | Test loss: {test_loss}")
        print(model_0.state_dict())

    

with torch.inference_mode(): # turns off gradient tracking ++ not needed for testing
    y_preds_new = model_0(X_test)

plot_predictions(predictions = y_preds_new)

print(f"Loss: {loss}")
print(model_0.state_dict())
print(weight, bias)

#1. Create Model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)

#2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#3. Save the model state directory
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj = model_0.state_dict(), f = MODEL_SAVE_PATH)

# To load in a saved state_dict -> instantiate a new instance of our model class

loaded_model_0 = LinearRegressionModel()

# Load saved state_dict of model 0 (new instance, updated parameters)

loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))

