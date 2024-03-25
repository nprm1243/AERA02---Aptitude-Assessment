import os
import torch
import cv2
from torchvision.transforms import ToTensor, Resize, RandomCrop, RandomRotation, Normalize, Compose, RandomHorizontalFlip, ToPILImage, ColorJitter, GaussianBlur, RandomAdjustSharpness
from torchvision.utils import save_image

transforms = Compose([
    ToPILImage(),
    RandomRotation(degrees=10, fill = 255),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast =0.2, saturation =0.2),
    GaussianBlur(kernel_size=(3, 3), sigma=(0.01, 0.3)),
    RandomAdjustSharpness(sharpness_factor = 0, p=0.3),
    ToTensor(),
    Normalize(mean=(0.5,), std=(0.5,)),
    Resize((280, 280)),
])

PATH = "./training_data"
totensor = ToTensor()
for item in os.listdir(PATH):
    n = len(os.listdir(f"{PATH}/{item}"))
    if (n >= 200):
        continue
    needed_epoch = (200 + n - 1) // n
    SUB_PATH = f"{PATH}/{item}"
    print(f"Augmenting {SUB_PATH}")
    idx = 0
    for subitem in os.listdir(SUB_PATH):
        dir = f"{SUB_PATH}/{subitem}"
        image = cv2.imread(dir)
        image = totensor(image)
        for i in range(needed_epoch-1):
            new_image = transforms(image)
            save_image(new_image, f"{SUB_PATH}/img_augmented_{idx}.jpg")
            idx += 1

for item in os.listdir(PATH):
    SUB_PATH = f"{PATH}/{item}"
    print(item, len(os.listdir(SUB_PATH)))