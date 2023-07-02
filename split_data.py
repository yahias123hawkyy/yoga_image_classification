import os
import shutil
from sklearn.model_selection import train_test_split
import constants as constant


folders = os.listdir(constant.dataset_path)


os.makedirs(constant.train_dir, exist_ok=True)
os.makedirs(constant.test_dir, exist_ok=True)

for folder in folders:
    folder_path = os.path.join(constant.dataset_path, folder)
    images = os.listdir(folder_path)
    train_images, test_images = train_test_split(
        images, test_size=0.3, random_state=42)

    train_folder_path = os.path.join(constant.train_dir, folder)
    os.makedirs(train_folder_path, exist_ok=True)
    for train_image in train_images:
        src = os.path.join(folder_path, train_image)
        dst = os.path.join(train_folder_path, train_image)
        shutil.copy(src, dst)

    test_folder_path = os.path.join(constant.test_dir, folder)
    os.makedirs(test_folder_path, exist_ok=True)
    for test_image in test_images:
        src = os.path.join(folder_path, test_image)
        dst = os.path.join(test_folder_path, test_image)
        shutil.copy(src, dst)
