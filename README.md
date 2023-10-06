# Radiologist-in-the-Loop AI 

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/raw/main/imgs/radiologist_in_the_loop_cover.png" alt="Radiologist-in-the-Loop AI" width="500"/>
</div>

This is the official GitHub repository for the [Radiologist-in-the-Loop AI Tutorial](https://radiologistintheloop.ai/) at MICCAI 2023! This repository serves as a companion to the course, providing resources, code examples, and materials to enhance your learning experience. 🧠💻📚

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

### Training the binary classifier

Assuming you have images in the `validation` and `test` folders, you can begin by randomly selecting (and assigning labels to) two images from the `unlabeled` folder. At this stage, ensure that one image is assigned to class `A`, and the other image is assigned to class `B`.

.....
