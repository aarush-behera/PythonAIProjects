# My first convolutional network

import torch
from torch import nn
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from helper_function import accuracy_fn
from timeit import default_timer as timer

start_time = timer()

# Set up training data
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

image, label = train_data[0]

classes = train_data.classes

# Create batch size
BATCH_SIZE = 32

# Turn datasets into iterables
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define model
class LuhBetasBrainV0(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# Instantiate model
model_0 = LuhBetasBrainV0().to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    params=model_0.parameters(),
    lr=0.01
)

# Function to time experiments
def print_train_time(start: float, end: float, device=None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

epochs = 30

for epoch in range(epochs):
    if epoch % 10 == 0:
        print(f"\nEpoch: {epoch + 1}\n--------")

    ### TRAINING ###
    train_loss = 0

    model_0.train()

    for batch, (X, y) in enumerate(train_dataloader):

        X = X.to(device)
        y = y.to(device)

        # Forward pass
        y_pred = model_0(X)

        # Loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Zero grad
        optimizer.zero_grad()

        # Backward
        loss.backward()

        # Step
        optimizer.step()

        if batch % 400 == 0 and epoch % 10 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

    train_loss /= len(train_dataloader)

    ### TESTING ###
    test_loss = 0
    test_acc = 0

    model_0.eval()

    with torch.inference_mode():

        for X_test, y_test in test_dataloader:

            X_test = X_test.to(device)
            y_test = y_test.to(device)

            # Forward pass
            test_pred = model_0(X_test)

            # Loss
            test_loss += loss_fn(test_pred, y_test).item()

            # Accuracy
            test_acc += accuracy_fn(
                y_true=y_test,
                y_pred=test_pred.argmax(dim=1)
            )

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(
        f"Train loss: {train_loss:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"Test acc: {test_acc:.2f}%"
    )

end_time = timer()

print_train_time(
    start=start_time,
    end=end_time,
    device=device
) 



from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)
MODEL_NAME = "pytorch_cnn_model_V0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# SAVE THE MODEL SAVE DICT
#torch.save(obj = model_0.state_dict(), f = MODEL_SAVE_PATH) # state_dict is all the learned weights and biases

loaded_model_0 = LuhBetasBrainV0().to(device)
loaded_model_0.load_state_dict(f = MODEL_SAVE_PATH)
