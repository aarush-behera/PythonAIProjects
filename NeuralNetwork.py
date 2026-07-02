"""
# Neural Network Classification with PyTorch
# 1. Make Classification Data
import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1000 samples
n_samples = 1000
# Create Circles
X, y = make_circles(n_samples, noise = 0.03, random_state = 42)  

circles = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
#print(circles.head(10))

plt.scatter(x = X[:, 0],
            y = X[:, 1],
            c = y,
            cmap = plt.cm.RdYlBu)

#plt.show()

# Check input and output shapes
#print(X.shape, y.shape)

X_sample = X[0]
y_sample = y[0]
# you can check type by going type(X) or type(y)

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# How to make a random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#print(len(X_train), len(X_test))
#print(len(y_train), len(y_test))

#Build a model
device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10) # takes 2 feat. upscales to 5
        self.layer_2 = nn.Linear(in_features = 10, out_features = 10)
        self.layer_3 = nn.Linear(in_features = 10, out_features = 1)

    # define forward method outlining forward pass
    def forward(self, x):
        #z = self.layer_1(x)
        #z = self.layer_2(z)
        #z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
# Instantiate instance of model, send to target device

model_0 = CircleModelV1().to(device)

#print(model_0)
#print(model_0.state_dict())
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
#print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
#print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
#print(f"\n First 10 predictions:\n{untrained_preds[:10]}")
#print(f"\n First 10 Labels:\n{y_test[:10]}")

# playground.tensorflow.org <- I'll use this for personal projects

# Set up a loss and optimizer function
# regression -> MAE or MSE classification -> binary cross entropy or categorical cross entropy

loss_bce = nn.BCEWithLogitsLoss() #has sigmoid function activation built in
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)
 
# Calculate Accuracy out of 100, what does it get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    
    return acc

#logits -> pred probs -> pred labels

model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
    # logits are the output of the function that have not been
    # manipulated by an activation function yet
#print(y_logits)

# convert logits to prediction probabilities
y_preds_probs = torch.sigmoid(y_logits)
            
# Find predicted labels 
y_preds = torch.round(y_preds_probs)

# in full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
#print(y_pred_labels)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)





epochs = 1000
for epoch in range(epochs):
    model_0.train()

    #forward_pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate the loss/accuracy
    loss = loss_bce(y_logits, #nn.BCEWithLogitsLoss expects logits as one of the prameters followed y the training data 
                    y_train) 
    acc = accuracy_fn(y_true = y_train, 
                      y_pred = y_pred)
    
    # Optimizer "Zero grad"
    optimizer.zero_grad()

    # loss backward  (back propagation)
    loss.backward()

    # optimizer step (gradient descent)
    optimizer.step()

    ### LUH TESSSTT ###
    model_0.eval()
    with torch.inference_mode():
        y_test_logits = model_0(X_test).squeeze() 
        y_test_pred = torch.round(torch.sigmoid(y_test_logits))

        test_loss = loss_bce(y_test_logits, y_test) # loss order: predictions, act. val
        test_acc = accuracy_fn(y_true = y_test, y_pred = y_test_pred)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Train Loss: {loss:.5f}, Train Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("Already Exists")
else: 
    print("Downloading Helper Funcs.py... ")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_function.py", "wb") as f:
        f.write(request.content)


from helper_function import plot_predictions, plot_decision_boundary

plt.figure(figsize = (12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2) # rows, columns, index
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()

# Improve our model

"""

import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Create Data
# ==========================================

X, y = make_circles(
    n_samples=1000,
    noise=0.03,
    random_state=42
)

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# ==========================================
# Model
# ==========================================

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(2, 16)
        self.layer_2 = nn.Linear(16, 16)
        self.layer_3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)

        return x

model_0 = CircleModelV1().to(device)

# ==========================================
# Loss + Optimizer
# ==========================================

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    params=model_0.parameters(),
    lr=0.01
)

# ==========================================
# Accuracy Function
# ==========================================

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

# ==========================================
# Training Loop
# ==========================================

epochs = 3000

for epoch in range(epochs):

    # Training
    model_0.train()

    y_logits = model_0(X_train).squeeze()

    y_pred = torch.round(
        torch.sigmoid(y_logits)
    )

    loss = loss_fn(y_logits, y_train)

    acc = accuracy_fn(
        y_true=y_train,
        y_pred=y_pred
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # Testing
    model_0.eval()

    with torch.inference_mode():

        test_logits = model_0(X_test).squeeze()

        test_pred = torch.round(
            torch.sigmoid(test_logits)
        )

        test_loss = loss_fn(
            test_logits,
            y_test
        )

        test_acc = accuracy_fn(
            y_true=y_test,
            y_pred=test_pred
        )

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | "
            f"Train Loss: {loss:.5f} | "
            f"Train Acc: {acc:.2f}% | "
            f"Test Loss: {test_loss:.5f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

# ==========================================
# Final Accuracy
# ==========================================

model_0.eval()

with torch.inference_mode():

    test_logits = model_0(X_test).squeeze()

    test_pred = torch.round(
        torch.sigmoid(test_logits)
    )

    final_acc = accuracy_fn(
        y_test,
        test_pred
    )

print(f"\nFinal Test Accuracy: {final_acc:.2f}%")

from helper_function import plot_decision_boundary

plt.figure(figsize = (12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2) # rows, columns, index
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()
