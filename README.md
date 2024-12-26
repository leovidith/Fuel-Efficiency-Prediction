# Fuel Efficiency Prediction

## Overview

The **Auto MPG dataset** is a collection of data related to fuel efficiency for different car models. This dataset is publicly available on Kaggle and is commonly used for regression tasks, aiming to predict the miles per gallon (MPG) a car can achieve based on various attributes.

The dataset contains the following columns:
- **mpg**: Miles per gallon (target variable).
- **cylinders**: Number of cylinders in the car.
- **acceleration**: Acceleration (time taken to go from 0 to 60 mph).
- **model year**: Model year of the car.
- **origin**: Origin of the car (1: USA, 2: Europe, 3: Japan).
- **horsepower**: The car's horsepower.
- **weight**: The weight of the car.
- **displacement**: The engine displacement of the car.
<img src="https://github.com/leovidith/Fuel-Efficiency-Prediction/blob/main/heatplot.png" width=700px>

For this project, the aim was to predict the fuel efficiency of a car based on features like the number of cylinders, acceleration, model year, and origin.

## Results

After training the model, the following results were obtained:

- **Final Loss (MAE)**: 2.8236
- **Final MAPE**: 11.76%

These metrics reflect the model's performance in predicting the fuel efficiency (`mpg`) for unseen data. The model is able to predict fuel efficiency with reasonable accuracy, given the complexity of the relationships among the input features.

## Agile Features

### Sprint 1: Data Collection and Preprocessing
- Collected the Auto MPG dataset from Kaggle.
- Preprocessed the data by handling missing values and converting categorical variables (like origin) into numerical values.
- Removed unnecessary columns like `car name`, `displacement`, and `horsepower` from the dataset.

### Sprint 2: Exploratory Data Analysis (EDA)
- Visualized the relationships between different features and the target variable (`mpg`).
- Created correlation matrices to understand how features relate to each other.
- Plotted the distribution of features and target variable to identify patterns and outliers.

### Sprint 3: Model Development
- Developed a neural network model using Keras with TensorFlow backend.
- Model architecture:
  - Dense layers with ReLU activation functions.
  - Batch normalization and dropout for regularization.
  - MAE loss function and Adam optimizer.
  
### Sprint 4: Model Evaluation and Tuning
- Evaluated the model using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
- Implemented early stopping, learning rate reduction, and model checkpointing to prevent overfitting and improve convergence.

### Sprint 5: Final Testing and Result Analysis
- Ran the final model on the validation dataset.
- Generated performance metrics: MAE and MAPE.
- Analyzed the model's predictions by visualizing the loss and evaluation metrics over epochs.

## Conclusion

In this project, we successfully trained a model to predict a car's fuel efficiency (MPG) based on its features such as the number of cylinders, acceleration, model year, and origin. The model performed well with an MAE of 2.8236 and a MAPE of 11.76%, showing that it can predict fuel efficiency with acceptable accuracy.

Future improvements could involve:
- Incorporating additional features such as `horsepower` and `weight` for potentially better predictions.
- Fine-tuning the model with hyperparameter optimization techniques.
- Exploring more advanced models, such as gradient boosting or deep learning models, for improved performance.
