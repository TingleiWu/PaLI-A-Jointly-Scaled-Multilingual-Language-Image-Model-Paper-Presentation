# PaLI-A-Jointly-Scaled-Multilingual-Language-Image-Model-Paper-Presentation

# Outline
- [Overview](#Overview)
- [Model Architecture](#Model Architecture)
- [Data](#Data)
- [Pretraining Tasks](#Pretraining Tasks)
- [Testing](#Testing)
- [Limitations/Biases](#Limitations/Biases)
- [Critical Analysis](#Critical Analysis)
- [Link](#Link)

# Overview

> This project mainly focuses on utilizing  the technology of deep learning models to detect different emotions of people in an image. Emotion detection is a way to understand people better in social settings to detect feelings like happiness, sadness, surprise at a specific moment without actually asking them. It is useful in many areas like security, investigation and healthcare. After the algorithm was built, an emotion detection website was developed to allow users to upload images, and get the appropriate emotion of the person in the image.

# Model Architecture

# Data

Data source: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

> The data was uploaded to Google Drive from Kaggle, and it contains 35,685 emotion images in total and categorized them into 7 different categories (sad, neutral, happy, angry, disgusted, surprised, fearful). All the emotion images are saved in png format and each of them has a shape of 48x48 pixels in grayscale. However, we realized that the data labels were not all sufficiently represented: all the other emotion categories have over 4,000 image data, while for the disgusted category had about 500 images available. This can potentially cause problems later in the project. Therefore, we decided to remove the disgusted category from the data, which means only six emotions (sad, neutral, happy, angry, surprised, fearful) will be used for classification. This is done to avoid a data imbalance problem. 

# Pretraining Tasks


# Testing


# Limitations/Biases


# Critical Analysis


# Link
