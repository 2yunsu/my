from camera_ready import camera_ready
import cv2
import pandas as pd
from cnn import start_cnn

labels = pd.read_csv('labels.csv', names=['Name', 'Labels'])

while True:
    name = input("file name:")
    camera_ready(name)
    image_name = name+'.png'
    input_label = pd.DataFrame({'Name':[image_name], 'Labels': [input("Label: ")]})
    if image_name in labels[['Name']].values:
        print("file name is already exists: ")
        continue
    labels=labels.append(input_label, ignore_index=True)
    labels.to_csv("labels.csv", mode='w', index=False, header=False)
    start_learning = input("start learning?(Y/N): ")

    if start_learning=="Y"or start_learning=="y":
        start_cnn()
        break

    else:
        continue