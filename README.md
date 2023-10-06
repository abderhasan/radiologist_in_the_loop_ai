# Radiologist-in-the-Loop AI 

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/raw/main/imgs/radiologist_in_the_loop_cover.png" alt="Radiologist-in-the-Loop AI" width="500"/>
</div>

This is the official GitHub repository for the [Radiologist-in-the-Loop AI Tutorial](https://radiologistintheloop.ai/) at MICCAI 2023! This repository serves as a companion to the tutorial, providing resources, code examples, and materials to enhance your learning experience. 🧠💻📚

## Setup

The code in this repository is meant for performing binary classification, which involves distinguishing between two classes. In the context of the tutorial, the classification task focused on identifying whether a given ultrasound image contains the liver region or not. 

Assume your class names are `A` and `B`. In your main project folder, do the following:

1. Create a folder called `data` within your project directory.

2. Inside the `data` folder, create the following subfolders:
   - `unlabeled`: This folder will contain images without any assigned class labels.
   - `training`: This folder contains the images used for training the neural network.
   - `validation`: This folder contains the images used for neural network validation.
   - `test`: This folder contains the images used for testing the trained model.

3. Within the `training`, `validation', and `test` subfolders, create two more subfolders for each class:
   - For `training` create folders named `A` and `B`.
   - For `validation` create folders named `A` and `B`.
   - For `test` create folders named `A` and `B`.

4. Place the images that belong to each class into their respective `A` and `B` folders within the `training`, `validation`, and `test` subfolders. Those images represent the labeled data.

5. The `unlabeled` folder should contain images that have not been assigned to any specific class.

Your folder structure should now look as follows:

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/imgs/project_folder.png" alt="project_folder_structure" width="500"/>
</div>

## Uncertainty sampling (Least confidence sampling)

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/imgs/uncertainty_sampling.jpg" alt="uncertainty_sampling" width="500"/>
</div>

### Train the binary classifier

Assuming you have images in the `validation` and `test` folders, you can begin by randomly selecting (and assigning labels to) two images from the `unlabeled` folder. At this stage, ensure that one image is assigned to class `A`, and the other image is assigned to class `B`.

Now, run the `simple_classifier.py` script. Once the script finishes running, it will output the accuracy of the trained model when applied to the test dataset. Additionally, it will generate a file named `unlabeled_probabilities.csv`. This file contains the probability distribution for the unlabeled images located in your `unlabeled` folder. The format of the CSV file is as follows:

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/imgs/table.png" alt="table" width="250"/>
</div>

### Find least confidence scores

To calculate the least confidence score from the probability distribution for each prediction, run the `least_confidence_score.py` script. The primary parameters required for this script are the input and output CSV files. In the script, these are defined as follows:

```python
input_csv_file = 'unlabeled_probabilities.csv'
output_csv_file = 'least_confidence_scores.csv'
```

The results in `least_confidence_scores.csv` will be sorted in ascending order. So, the content in the CSV file would look as follows:

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/blob/main/imgs/table_2.png" alt="table" width="500"/>
</div>
