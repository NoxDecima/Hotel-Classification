import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import shutil

csv_path = 'C:/Users/evert/Documents/Radboud University/Master/MLiP/Code/train_images/'
path = 'C:/Users/evert/Documents/Radboud University/Master/MLiP/Code/train_images/images/'
new_dataset_path = 'C:/Users/evert/Documents/Radboud University/Master/MLiP/Code/new_images/images/'

data_df = pd.read_csv(os.path.join(csv_path, "train.csv"))
output_df = pd.DataFrame(columns=['image_id', 'hotel_id'])
bathroom_df = pd.DataFrame(columns=['image_id', 'hotel_id', 'labels'])

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
bathroom_items = ['toilet', 'sink', 'toothbrush', 'hair brush']

for index, row in data_df.iterrows():
    image_path = os.path.join(path, row['image_id'])
    img = np.array(Image.open(image_path))

    output = model(img)
    results = output.pandas().xyxy[0]['name'].tolist()
    check = any(item in bathroom_items for item in results)
    if check:
        list_to_string = ', '.join([str(elem) for elem in results])
        single_bathroom_df = pd.DataFrame({'image_id':[row['image_id']], 'hotel_id':[row['hotel_id']], 'labels':list_to_string}, columns=['image_id', 'hotel_id', 'labels'])
        bathroom_df = pd.concat([bathroom_df, single_bathroom_df])
    else:
        temp_df = pd.DataFrame({'image_id':[row['image_id']], 'hotel_id':[row['hotel_id']]}, columns=['image_id', 'hotel_id'])
        output_df = pd.concat([output_df, temp_df])
        shutil.move(image_path, new_dataset_path)

output_df.to_csv('C:/Users/evert/Documents/Radboud University/Master/MLiP/Code/new_images/train.csv', index=False)
bathroom_df.to_csv('C:/Users/evert/Documents/Radboud University/Master/MLiP/Code/new_images/bathrooms.csv', index=False)

