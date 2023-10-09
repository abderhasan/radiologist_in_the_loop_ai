import pandas as pd
import shutil
import os

csv_files = ['class_1.csv', 'class_2.csv', 'class_3.csv', 'class_4.csv', 'class_5.csv']

image_path = './data/unlabeled/'

for file in csv_files:
    df = pd.read_csv(file)
    
    # Get the class number from the file name
    class_num = file.split('_')[1].split('.')[0]
    
    # Create the directories if they don't exist
    os.makedirs(f'class_{class_num}_centroid', exist_ok=True)
    os.makedirs(f'class_{class_num}_outliers', exist_ok=True)
    os.makedirs(f'class_{class_num}', exist_ok=True)
    
    for index, row in df.iterrows():
        image_name = row['image_name']
        
        # Check if the image is a centroid
        if row['is_centroid']:
            shutil.copy(image_path + image_name, f'class_{class_num}_centroid')
        # Check if the image is an outlier
        elif row['is_outlier']:
            shutil.copy(image_path + image_name, f'class_{class_num}_outliers')
        # If the image is not a centroid or an outlier
        else:
            shutil.copy(image_path + image_name, f'class_{class_num}')