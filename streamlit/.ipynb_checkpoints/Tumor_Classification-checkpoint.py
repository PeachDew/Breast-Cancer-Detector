import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Tumor Classification")

st.sidebar.header("Predicting Gender from Twitter Profile.")
st.markdown("""
# Malignant/Benign Tumore Classification using Random Forest
"""
)

#Load the models
with open("./models/rf_model.pickle", "rb") as file:
    rf_model = pickle.load(file)
    
       
    
col1, col2 = st.columns(2)
with col1:
    area = st.slider('Choose area', min_value=0.0,
                    max_value=50.0,
                    value=25.0)
    radius = st.slider('Choose radius', min_value=0.0,
                    max_value=50.0,
                    value=25.0)
    perim = st.slider('Choose perimeter', min_value=0.0,
                    max_value=50.0,
                    value=25.0)
    # Shape
    smooth = st.slider('Choose smoothness', min_value=0.0,
                    max_value=50.0,
                    value=25.0)
    # Roughness
    text = st.slider('Choose texture', min_value=0.0,
                    max_value=50.0,
                    value=25.0)
    
    
with col2:
    pred_button = st.button('Generate Prediction')
    if pred_button:
        with st.spinner('Wait for it...'):
            data = [[area, text, perim, area, smooth]]
            prediction = rf_model.predict(data)
            probabilities = rf_model.predict_proba(data)
            
            # Display the prediction
            if prediction == 0:
                st.write("The tumor is predicted to be benign.")
            else:
                st.write("The tumor is predicted to be malignant.")
            st.write("Benign Probability:", probabilities[0][0])
            st.write("Malignant Probability:", probabilities[0][1])
            
            
            

