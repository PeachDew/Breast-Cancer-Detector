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
# Gender/Brand Prediction 📱
This app utilizes machine learning to make predictions based on a Twitter user profile. Simply provide us with some information about a Twitter user, and we'll generate a prediction for you!
"""
)

# Load the models
#with open("./Gender_Model_Save/best_clf.pkl", "rb") as file:
#    name_model = pickle.load(file)
       
    
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
    smooth = st.slider('Choose smoothness', min_value=0.0,
                    max_value=50.0,
                    value=25.0)
    text = st.slider('Choose texture', min_value=0.0,
                    max_value=50.0,
                    value=25.0)
    
    
with col2:
    pred_button = st.button('Generate Prediction')
    if pred_button:
        with st.spinner('Wait for it...'):
            st.balloons()
            st.markdown("Yay!")
            

