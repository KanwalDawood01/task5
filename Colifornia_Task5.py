import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Streamlit Title
# ------------------------------
st.title("CIFAR-10 Classification with Data Augmentation")

# ------------------------------
# Transform Definitions
# ------------------------------
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Transform for validation/test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ------------------------------
# Load Dataset
# ------------------------------
batch_size = 128
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
val_size = 5000
train_size = len(dataset) - val_size
train_data, val_data = random_split(dataset, [train_size, val_size])

val_data.dataset.transform = test_transform  # apply test transform for validation set

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=batch_size)

# ------------------------------
# Define LeNet or AlexNet-style CNN
# ------------------------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.pool(self.conv1(x)))
        x = torch.tanh(self.pool(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------------------
# Training Function
# ------------------------------
def train_model():
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50

    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_acc_list.append(val_acc)

        st.write(f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, train_acc_list, val_acc_list

# ------------------------------
# Evaluation Function
# ------------------------------
def plot_confusion_matrix(model):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=test_data.classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    st.pyplot(fig)

# ------------------------------
# Train Model Button
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if st.button("Train LeNet on CIFAR-10"):
    model, train_accs, val_accs = train_model()

    # Plot Accuracy Curves
    fig, ax = plt.subplots()
    ax.plot(train_accs, label="Train Accuracy")
    ax.plot(val_accs, label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    st.pyplot(fig)

    # Show Confusion Matrix
    st.subheader("Confusion Matrix on Test Set")
    plot_confusion_matrix(model)
