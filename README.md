# Translating American Sign Language into Text Using CNNs

Built through the **AI4ALL Ignite** Program, our project aimed to bridge communication barriers for Deaf and Hard-of-Hearing communities by training a deep learning model to interpret American Sign Language (ASL) letters. We applied Python, data preprocessing, and convolutional neural networks (CNNs) to build a functional model that recognizes ASL hand signs and outputs their corresponding letters. This initiative reflects our shared commitment to accessible and inclusive technology.

## Problem Statement 

ASL is a key language for many individuals, yet there remains a lack of accessible tools to automatically translate it into written or spoken language in real time. Our project addresses this gap by creating a model that can recognize ASL alphabet signs from static images. This technology could support communication in classrooms, public spaces, and everyday life, providing more independence and inclusion for ASL users.

## Key Results 

1. Trained a Convolutional Neural Network on over 78,000 labeled ASL images (about 3,000 images per alphabet letter).
2. Achieved strong validation accuracy across the 26 ASL alphabet classes.
3. Demonstrated successful letter classification from test images not seen during training.
4. Created a working model that can serve as a foundation for real-time ASL recognition tools.

## Methodologies 
To build our solution, we used a supervised learning approach, leveraging convolutional neural networks due to their effectiveness in image classification tasks. We:

- Preprocessed the dataset by resizing images and converting them to grayscale.
- Used data augmentation techniques to increase generalization.
- Built and trained a CNN using Keras with TensorFlow backend.
- Evaluated model performance using accuracy metrics and a confusion matrix.
- Explored model tuning strategies (filter size, epochs, dropout rate) to avoid overfitting.

## Data Sources 

- Kaggle Dataset: [Interpret Sign Language with Deep Learning](https://www.kaggle.com/code/paultimothymooney/interpret-sign-language-with-deep-learning/notebook) by Paul Mooney

## Technologies Used 

- Python  
- TensorFlow / Keras  
- NumPy  
- pandas  
- matplotlib  
- Google Colab  

## Authors 

This project was completed in collaboration with:

- Maria-Emilia Iordache  
- Tony Aguilar  
- Amina El Guenuni 
- Claire Chin  
- Ayomide Onibokun  

We are proud to have contributed this project as part of the **2025 AI4ALL Ignite** accelerator program.
