import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, RandomCrop, RandomRotation, Normalize, Compose, RandomHorizontalFlip, ToPILImage, ColorJitter, GaussianBlur, RandomAdjustSharpness
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet50, resnet34
from torchvision.utils import save_image
import PIL
import random
import warnings
warnings.filterwarnings("ignore")

labels_to_idx = {
    "cardigans" : 0,
    "dresses" : 1,
    "men_coats" : 2,
    "men_sweaters" : 3,
    "men_trousers" : 4,
    "polo_shirts" : 5,
    "tanks" : 6,
    "women_coats" : 7,
    "women_pants" : 8,
    "women_sweaters" : 9,
    "women_tshirts" : 10,
}
class boutique_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.image_list = []
        for category in os.listdir(path):
            for file in os.listdir(f"{path}/{category}"):
                self.image_list.append((f"{path}/{category}/{file}", labels_to_idx[category]))
        self.transform = ToTensor()
        random.shuffle(self.image_list)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx][0])
        image = self.transform(image)
        return image, self.image_list[idx][1]

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class CustomRESNet50(nn.Module):
    def __init__(self):
        super(CustomRESNet50, self).__init__()
        backbone = resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.resnet50 = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(11),
            nn.Softmax(dim=1)
        )
    
    def forward(self, X):
        X = self.resnet50(X)
        X = self.classifier(X)
        return X

if __name__ == '__main__':
    k_folds = 5
    num_epochs = 20
    loss_function = nn.CrossEntropyLoss()
    random.seed(1)

    training_data = boutique_dataset('./training_data')
    testing_data  = boutique_dataset('./testing_data')

    results = {}

    torch.manual_seed(1)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(training_data)):
        print(f"fold: {fold}")

        train_subsample = SubsetRandomSampler(train_idx)
        valid_subsample = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(training_data, batch_size=16, sampler=train_subsample, num_workers=6)
        valid_loader = DataLoader(training_data, batch_size=16, sampler=valid_subsample, num_workers=6)

        model = CustomRESNet50()
        model = model.to('cuda')
        model.train()

        num_train = sum(train_idx)
        num_valid = sum(valid_idx)

        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

        for epoch in range(num_epochs):
            train_loss = 0.0
            valid_loss = 0.0
            train_correct = 0
            valid_correct = 0
            train_total = 0
            valid_total = 0
            for (inputs, labels) in train_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_total += labels.size(0)
                # _, predicted = torch.max(outputs.data, 1)
                train_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            save_path = f'./model-folds/model-fold-{fold}.pt'
            torch.save(model, save_path)
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    valid_loss += loss.item()
                    valid_total += labels.size(0)
                    valid_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            print(f"Epoch {epoch} | train_loss: {train_loss/len(train_loader):>10.5f} | train_acc: {100.0*train_correct/train_total:>3.5f}% | valid_loss: {valid_loss/len(valid_loader):>10.5f} | train_acc: {100.0*valid_correct/valid_total:>3.5f}%")

    test_loader = DataLoader(testing_data, batch_size=64, num_workers=6)
    with torch.no_grad():
        for fold in range(5):
            model = torch.load(f"model-folds/model-fold-{fold}.pt")
            model.to(device='cuda')
            model = model.to('cuda')
            model.eval()

            test_loss = 0.0
            test_size = 0
            correct = 0

            for (inputs, labels) in test_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item()
                test_size += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            print(f"Fold {fold} | Testing loss: {test_loss/len(test_loader):>10.5f} | Testing accuracy: {(correct/test_size)*100:>3.3}%")