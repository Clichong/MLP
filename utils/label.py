import cv2
import os
import torch
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder

root = '../caltech/caltech101/101_ObjectCategories/'

# pre label and data
image_paths = list(paths.list_images(root))
data = []
labels = []
for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    if label == 'BACKGROUND_Google':
        continue
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data.append(image)
    labels.append(label)
data = np.array(data)
labels = np.array(labels)

gle = LabelEncoder()
genre_labels = gle.fit_transform(labels)
genre_mappings = {label: index for index, label in enumerate(gle.classes_)}
print(genre_mappings)
