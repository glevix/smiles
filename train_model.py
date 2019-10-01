from lenet import LeNet
import numpy as np
import cv2
import os
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import imutils

data, labels = [], []

total_0 = 0
total_1 = 0

for path in sorted(list(paths.list_images('datasets'))):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    data.append(img_to_array(image))
    category = float(path.split(os.path.sep)[-2])
    if category > 0:
        labels.append([0., 1.])
        total_1 += 1
    else:
        labels.append([1., 0.])
        total_0 += 1

data, labels = np.array(data, dtype='float'), np.array(labels)
data = data / 255.0

weight = max(total_0, total_1) / np.array([total_0, total_1])

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels)

model = LeNet.build_bagged(5, 2000, width=28, height=28, depth=1, classes=2)
model.compile(loss=['binary_crossentropy'], optimizer='adam')

H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=weight, batch_size=64, epochs=20, verbose=1)

predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=['0', '1']))

model.save()

