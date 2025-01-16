import streamlit as st
import pandas as pd 
import pickle 
import numpy as np
from sklearn.datasets import load_diabetes

st.title(" this app is tp predict the glucose level in the blood of diabetic patient")

model_path = 'Models\model_ridge.pkl'
model_lr=pickle.load(open(model_path,'rb'))
 


#load dataset
diab= load_diabetes()
x= pd.DataFrame(diab.data,columns=diab.feature_names)

#user data
user_input ={}

for col in x.columns:
    user_input[col]= st.slider(col, x[col].min(),x[col].max())

df = pd.DataFrame(user_input, index = [0])

st.write(df)

if st.button("Predict"):
    # Make prediction
    prediction = model_lr.predict(df)[0]  # Ensure df is preprocessed correctly
    st.write(f'The predicted glucose level is: {prediction:.2f}')
