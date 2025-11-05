# Crop Recommendation System using Machine Learning

## Overview
The **Crop Recommendation System** is a machine learning-based project that predicts the most suitable crop to grow based on environmental and soil conditions.  
It helps farmers make informed decisions to increase productivity by analyzing key factors such as soil nutrients, temperature, humidity, pH value, rainfall, and soil type.

## Features
- Predicts the **best crop** for given environmental conditions  
- Uses **6 Machine Learning models** and compares their performance  
- Displays **evaluation metrics** (Accuracy, Precision, Recall, F1-score)  
- Shows **Confusion Matrix** and **Model Accuracy Comparison Graph**  
- Provides an **interactive Streamlit web app** for real-time predictions  
- Includes **Soil Type feature** for enhanced accuracy  

## Machine Learning Models Used
| S.No | Model Name              |
|------|--------------------------|
| 1️  | Logistic Regression       |
| 2️  | Decision Tree Classifier  |
| 3️  | Random Forest Classifier  |
| 4️  | Support Vector Machine (SVM) |
| 5️  | K-Nearest Neighbors (KNN) |
| 6️  | Naive Bayes Classifier    |

The **Random Forest Classifier** achieved the **highest accuracy** among all models and was selected as the final model for deployment.

## Technologies Used

| Category | Technology |
|-----------|-------------|
| Programming Language | Python |
| Libraries | scikit-learn, pandas, numpy, matplotlib, seaborn, joblib |
| Web Framework | Streamlit |
| Deployment Platform | Streamlit Cloud |
| Dataset | Crop Recommendation Dataset (Kaggle) |

## Input Features
| Feature | Description |
|----------|--------------|
| **N** | Nitrogen content in soil |
| **P** | Phosphorus content in soil |
| **K** | Potassium content in soil |
| **Temperature** | Temperature in °C |
| **Humidity** | Relative humidity in % |
| **pH** | Acidity/alkalinity of the soil |
| **Rainfall** | Rainfall in mm |
| **Soil Type** | Type of soil (Clay, Loamy, Sandy, Peaty, Chalky) |

## How It Works
1. Load the dataset and preprocess it.  
2. Train six ML models and evaluate them using accuracy, precision, recall, and F1-score.  
3. Compare model performances and select the best one (Random Forest).  
4. Save the model and scaler using `joblib`.  
5. Build an interactive **Streamlit web app** for real-time predictions.

