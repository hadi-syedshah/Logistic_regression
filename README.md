# Logistic_regression
# Breast Cancer Classification using Logistic Regression

This project implements logistic regression models to classify breast cancer tumors as either Malignant (M) or Benign (B) using the Wisconsin Breast Cancer dataset.

## Dataset

The dataset ([data.csv](data.csv)) contains the following features:
- ID number
- Diagnosis (M = Malignant, B = Benign)
- 30 real-valued features computed from digitized images of cell nuclei, including:
  - Radius
  - Texture 
  - Perimeter
  - Area
  - Smoothness
  - Compactness
  - Concavity
  - Concave points
  - Symmetry
  - Fractal dimension

Each feature has three measurements:
- Mean
- Standard error (SE)
- "Worst" or largest value

## Implementation

The project includes two different implementations:

1. [sklearn_logistic.ipynb](sklearn_logistic.ipynb)
- Uses scikit-learn's LogisticRegression
- Includes data preprocessing and feature scaling
- Visualizes data distributions and correlations
- Evaluates model performance

2. [Logistic_Tensorflow.ipynb](Logistic_Tensorflow.ipynb)
- Implements logistic regression using TensorFlow/Keras
- Uses binary cross-entropy loss
- Trains model using SGD optimizer
- Includes model evaluation metrics

## Requirements

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- TensorFlow
- Seaborn
- Matplotlib

## Usage

1. Clone the repository
2. Install dependencies
3. Run either notebook:
   ```bash
   jupyter notebook sklearn_logistic.ipynb
   # or
   jupyter notebook Logistic_Tensorflow.ipynb
   ```

## Model Details

The logistic regression model:
- Input: 30 features from breast cancer data
- Output: Binary classification (0=Benign, 1=Malignant)
- Uses standard scaling for feature normalization
- Implements regularization to prevent overfitting
