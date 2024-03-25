import os
import random
import shutil

PATH = "./AISIA_BOUTIQUE_DATASET"
TRAINING_PATH = "./training_data"
TESTING_PATH = "./testing_data"
random.seed(1)

for category in os.listdir(PATH):
    SUB_PATH = f"{PATH}/{category}"
    n = min(len(os.listdir(SUB_PATH)), 320)
    n_test = int(n * 0.3)
    n_train = n - n_test
    list_of_image = os.listdir(SUB_PATH)
    random.shuffle(list_of_image)
    idx = 0
    for i in range(n_train):
        shutil.copyfile(f"{SUB_PATH}/{list_of_image[idx]}", f"{TRAINING_PATH}/{category}/{list_of_image[idx]}")
        idx += 1
    for i in range(n_test):
        shutil.copyfile(f"{SUB_PATH}/{list_of_image[idx]}", f"{TESTING_PATH}/{category}/{list_of_image[idx]}")
        idx += 1

for category in os.listdir(PATH):
    print(category, len(os.listdir(f"{TRAINING_PATH}/{category}")), len(os.listdir(f"{TESTING_PATH}/{category}")))