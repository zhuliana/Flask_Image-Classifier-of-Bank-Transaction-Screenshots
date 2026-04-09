# Classification of Bank Transaction Screenshots for Authenticity Prediction

**Author:** Zhuliana Melva Rey | Universitas Diponegoro

## Overview
This project is an image classification system designed to uncover payment fraud in bank transactions by identifying fake Bank BRI transaction receipts. With online transaction fraud making up a large portion of cybercrime reports, there is a critical need for systems capable of distinguishing between authentic and forged payment proofs to prevent financial loss and maintain trust in banking systems.

This repository contains the code for a machine learning model built with an Artificial Neural Network (ANN) and Gray Level Co-occurrence Matrix (GLCM) feature extraction, alongside a web-based Flask application for easy, browser-based image classification. 

## Video Demo
> 

https://github.com/user-attachments/assets/4fb32b36-4901-4f7f-9542-79b7914c1fa6




## Features
* **Web-Based Image Classifier:** A user-friendly Flask application that allows users to upload a screenshot of a payment receipt and receive an instant prediction on whether it is real or fake.
* **GLCM Feature Extraction:** Analyzes the texture of the uploaded images by calculating combinations of pixel brightness. It extracts key features including Dissimilarity, Correlation, Homogeneity, Contrast, Angular Second Moment (ASM), and Energy.
* **Artificial Neural Network (ANN):** Powered by a backpropagation algorithm. The project explores different neural network architectures, including 2-hidden-layer and 3-hidden-layer models, to achieve the highest accuracy.

## Dataset
The dataset comprises 703 images collected manually from social media (Instagram, Twitter, Facebook, WhatsApp) and Google. It is well-balanced, containing:
* 351 Real Bank BRI payment receipts.
* 352 Fake Bank BRI payment receipts.

The data is split into 60% training (421 images), 20% validation (141 images), and 20% testing (141 images).

## Methodology
1. **Preprocessing Evaluation:** The project tested scenarios both with and without image preprocessing (grayscaling, sharpening, resizing). 
2. **Feature Extraction:** GLCM was applied using combinations of distances (1 and 5) and angles (0°, 45°, 90°, 135°, and All).
3. **Model Training:** Training was conducted using 100 epochs, 16 batches, and the Adam optimizer, with an early stopping mechanism if the validation loss did not decrease for 25 epochs.

## Key Results
* **Best Configuration:** Extensive testing of 40 different scenarios revealed that the model performed best **without preprocessing**, using GLCM angle "All" and distance 1 or 5.
* **High Accuracy:** The optimal model achieved an impressive **average accuracy of 96%** on non-preprocessed data, proving its effectiveness in identifying fraudulent screenshots.

## Tech Stack
* **Language:** Python
* **Machine Learning Environment:** Google Colaboratory
* **Web Framework:** Flask
* **Frontend:** HTML, CSS
