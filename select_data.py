'''
@author: Abder-Rahman Ali, PhD
@email: aali25@mgh.harvard.edu
@date: 2023
'''
import pandas as pd
import shutil
import os

df = pd.read_csv('least_confidence_scores.csv')

# Select the first 50 images (i.e., those with the highest least confidence scores)
selected_images = df['image'].iloc[:50]

for image in selected_images:
    shutil.move(f'./data/unlabeled/{image}', './selected_data/')

selected_data = pd.DataFrame(columns=['image', 'folder'])

# Search for each image in the specified folders
for image in selected_images:
    if os.path.isfile(f'./original_data/data/training/A/{image}'):
        new_row = pd.DataFrame({'image': [image], 'folder': ['A']})
        selected_data = pd.concat([selected_data, new_row], ignore_index=True)
    elif os.path.isfile(f'./original_data/data/training/B/{image}'):
        new_row = pd.DataFrame({'image': [image], 'folder': ['B']})
        selected_data = pd.concat([selected_data, new_row], ignore_index=True)
    else:
        print(f"Image {image} not found in either subfolder")

selected_data.to_csv('selected_data.csv', index=False)
