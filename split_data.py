import os
import shutil
from sklearn.model_selection import train_test_split

dataset_path = 'D:/Image_classification_yoga_poses/dataset'



folders = os.listdir(dataset_path)



train_dir = 'D:/Image_classification_yoga_poses/train_dataset'
test_dir = 'D:/Image_classification_yoga_poses/test_dataset'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    images = os.listdir(folder_path)
    train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)

    # Move the train images to the train directory
    train_folder_path = os.path.join(train_dir, folder)
    os.makedirs(train_folder_path, exist_ok=True)
    for train_image in train_images:
        src = os.path.join(folder_path, train_image)
        dst = os.path.join(train_folder_path, train_image)
        shutil.copy(src, dst)

    # Move the test images to the test directory
    test_folder_path = os.path.join(test_dir, folder)
    os.makedirs(test_folder_path, exist_ok=True)
    for test_image in test_images:
        src = os.path.join(folder_path, test_image)
        dst = os.path.join(test_folder_path, test_image)
        shutil.copy(src, dst)

        