# S10.1: Copy this code cell in 'iris_app.py' using the Sublime text editor. You have already created this ML model in the previous class(es).

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

@st.cache()
def prediction(sl,sw,pl,pw):
  predict = svc_model.predict([[sl,sw,pl,pw]])
  if predict[0] == 0:
    return "Iris-Setosa"
  elif predict[0] == 1:
    return "Iris-Verginica"
  else:
    return "Iris_Versi-color"
  
st.title("Iris flower Species Prediction")

sl = st.slider("Sepal Length",0,10)
sw = st.slider("Sepal Width",0,10)
pl = st.slider("Petal Length",0,10)
pw = st.slider("Petal Width",0,10)

button = st.button("Predict")

if button:
  predict = prediction(sl,sw,pl,pw)
  st.write(predict)
  st.write("accurcy is ",score)