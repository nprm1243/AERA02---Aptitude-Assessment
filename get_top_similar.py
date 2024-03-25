import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet50

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
idx_to_labels = ["cardigans", "dresses", "men_coats", "men_sweaters", "men_trousers", "polo_shirts", "tanks", "women_coats", "women_pants", "women_sweaters", "women_tshirts"]

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

classification_model_path = './model-folds/model-fold-0.pt'
classification_model = torch.load(classification_model_path)
classification_model.eval()
vectorization_model = resnet50()
vectorization_model.eval()
vectorization_model.eval()
vectorization_model.to('cuda')
totensor = ToTensor()

def get_vector(image):
    return vectorization_model(image.unsqueeze(0))

def get_top_similar(image, n_top = 5):
    classification_model.eval()
    outfit_class = classification_model(image.unsqueeze(0))
    outfit_class = torch.max(outfit_class.data, 1)[1]
    type = idx_to_labels[outfit_class]
    sub_path = f"./AISIA_BOUTIQUE_DATASET/{type}"
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    image_similarity_score = {}
    image_vector = get_vector(image)
    for file in os.listdir(sub_path):
        current_image = cv2.imread(f"{sub_path}/{file}")
        current_image = totensor(current_image)
        current_image = current_image.to('cuda')
        current_image_vector = vectorization_model(current_image.unsqueeze(0))
        image_similarity_score[f"{sub_path}/{file}"] = cos(image_vector, current_image_vector).item()
    image_similarity_score = sorted(image_similarity_score.items(), key=lambda x : x[1], reverse=True)
    return image_similarity_score[:min(len(os.listdir(sub_path)), n_top)]


# image = cv2.imread("./AISIA_BOUTIQUE_DATASET/dresses/img_5844921.jpg")
# image = totensor(image)
# image = image.to('cuda')
# tmp = get_top_similar(image, 5)
# print(tmp)