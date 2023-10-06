# Radiologist-in-the-Loop AI 

<div align="center">
  <img src="https://github.com/abderhasan/radiologist_in_the_loop_ai/raw/main/imgs/radiologist_in_the_loop_cover.png" alt="Radiologist-in-the-Loop AI" width="500"/>
</div>

This is the official GitHub repository for the [Radiologist-in-the-Loop AI Course](https://radiologistintheloop.ai/) at MICCAI 2023! This repository serves as a companion to the course, providing resources, code examples, and materials to enhance your learning experience. 🧠💻📚

## Setup

The code in this repository is meant for performing binary classification, which involves distinguishing between two classes. In the context of the tutorial, the classification task focused on identifying whether a given ultrasound image contains the liver region or not. 

Assume your class names are `A` and `B`. In your project folder, do the following:

1. Create two classes named "A" and "B" for your project.

2. Create a folder called "data" within your project directory. This will serve as the main data storage location.

3. Inside the "data" folder, create the following subfolders:
   - "unlabeled": This folder will contain images without any assigned class labels.
   - "training": This folder is for training data.
   - "validation": This folder is for validation data.
   - "test": This folder is for test data.

4. Within the "training," "validation," and "test" subfolders, create two more subfolders for each class:
   - For "training," create folders named "A" and "B."
   - For "validation," create folders named "A" and "B."
   - For "test," create folders named "A" and "B."

5. Organize your images by placing the images that belong to each class into their respective "A" and "B" folders within the "training," "validation," and "test" subfolders.

6. The "unlabeled" folder should contain images that have not been assigned to any specific class.
