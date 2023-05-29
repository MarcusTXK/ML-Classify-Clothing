# %%
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, precision_score
import random


# %%
data_dir = 'data/clothing-dataset-small-master'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train'), transform=transform)
test_set = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'validation'), transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=32, shuffle=False, num_workers=4)
# %%

resnet_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_classes = len(train_set.classes)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.0001, momentum=0.99 , weight_decay=0.00001)

# %%


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    epoch_precision = precision_score(
        all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision

# %%


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='micro')
    epoch_precision = precision_score(
        all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision


# %%
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    train_losses, train_accuracies, train_f1_scores, train_precision_scores = [], [], [], []
    test_losses, test_accuracies, test_f1_scores, test_precision_scores = [], [], [], []

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1, train_precision = train(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_f1, test_precision = evaluate(
            model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc.cpu().numpy())
        train_f1_scores.append(train_f1)
        train_precision_scores.append(train_precision)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc.cpu().numpy())
        test_f1_scores.append(test_f1)
        test_precision_scores.append(test_precision)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(
            f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} Precision: {train_precision:.4f}')
        print(
            f'Validation Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f} Precision: {test_precision:.4f}')

    return train_losses, train_accuracies, train_f1_scores, train_precision_scores, test_losses, test_accuracies, test_f1_scores, test_precision_scores


# %%
def plot_metrics(train_losses, train_accuracies, train_f1_scores, train_precision_scores, test_losses, test_accuracies, test_f1_scores, test_precision_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(20, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, test_losses, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, test_accuracies, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1_scores, label='Train')
    plt.plot(epochs, test_f1_scores, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_precision_scores, label='Train')
    plt.plot(epochs, test_precision_scores, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.show()


# %%
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet_model = resnet_model.to(device)
    num_epochs = 10

    # Hyperparameters: lr=0.01, momentum=0.9, weight_decay=1e-05

    print('Training and evaluating model...')

    train_losses, train_accuracies, train_f1_scores, train_precision_scores, test_losses, test_accuracies, test_f1_scores, test_precision_scores = train_and_evaluate(
        resnet_model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

    print('Plotting loss and accuracy...')

    plot_metrics(train_losses, train_accuracies, train_f1_scores, train_precision_scores,
                 test_losses, test_accuracies, test_f1_scores, test_precision_scores)
    
