import os
import re


labels = []
images = []
category = []

i = 0
dataset_path = 'D:/Image_classification_yoga_poses/dataset'

for directory in os.listdir(dataset_path):
    category.append(directory)
    for img in os.listdir(os.path.join(dataset_path, directory)):
        if len(re.findall('.png', img.lower())) != 0 or len(re.findall('.jpg', img.lower())) != 0 or len(re.findall('.jpeg', img.lower())) != 0:
            images.append(img)
            labels.append(i)

    i = i+1

print("Total labels: ", len(labels))
print("Total images: ", len(images))
print("Total categories: ", len(category))
