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

st.markdown("""
# Malignant/Benign Tumor Classification
"""
)

#Load the models
with open("./models/logreg_model.pkl", "rb") as file:
    logreg_model = pickle.load(file)
with open("./models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)    
    
       
    
col1, col2 = st.columns(2)
with col1:
    area = st.slider('Area', min_value=100.0,
                    max_value=3000.0,
                    value=650.0)
    radius = st.slider('Radius', min_value=6.0,
                    max_value=30.0,
                    value=14.0)
    perim = st.slider('Perimeter', min_value=20.0,
                    max_value=200.0,
                    value=188.0)
    # Shape
    smooth = st.slider('Smoothness', 
                       help="A smooth object (low value) will be something like a sphere.",
                       min_value=0.05,
                    max_value=0.1634,
                    value=0.096)
    # Roughness
    text = st.slider('Texture', 
                     help="Imagine something like silk for a low texture value, and a carpet for high texture values.",
                     min_value=9.0,
                    max_value=40.0,
                    value=19.0)
    
    
with col2:
    pred_button = st.columns([1,4,1])[1].button('Generate')
        
    if pred_button:
        with st.spinner('Wait for it...'):
            data = [[radius, text, perim, area, smooth]]
            df = pd.DataFrame(data, columns=['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness'])
            scaled_data = scaler.transform(df)
            prediction = logreg_model.predict(scaled_data)
            probabilities = logreg_model.predict_proba(scaled_data)
            
            pred_data = {'Tumor': ['BENIGN', 'MALIGNANT'],
                    'Probability': [f'{probabilities[0][0]*100:.5f}%', f'{probabilities[0][1]*100:.5f}%']}
            st.dataframe(pred_data)
            
            if prediction == 0:
                st.markdown("The tumor is predicted to be <span style='color: #78ff7f;'>BENIGN</span>.", unsafe_allow_html=True)
            else:
                st.write("The tumor is predicted to be <span style='color: #f25c6e;'>MALIGNANT</span>.", unsafe_allow_html=True)
            
            
            
            

