import streamlit as st
import pandas as pd
from PIL import Image
import os 
import pyiqa
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, mean_squared_error
from IPython.display import display




predicted_scores = pd.read_csv('dataset_tad66\descri.csv')
predicted_scores.rename( columns={'Unnamed: 0':'image_name'}, inplace=True )

# select model
models = ['nima', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512']
selected_model = st.selectbox('Sélectionner un modèle', models)

# select category
df_tad = pd.read_csv('dataset_tad66/filtered_tad_data.csv') 

categories = df_tad['category'].unique()
selected_category = st.selectbox('Sélectionner une catégorie', categories)

quality_choice = ['good quality', 'bad quality']
selected_quality = st.selectbox('Sélectionner le type des images', quality_choice)

# Function to check if the specific category is present in the list
def has_category(categories, target_category):
    return target_category in categories


# Filter the dataframe
filtered_df = df_tad[df_tad['category'].apply(lambda x: has_category(x, selected_category))]


good_images = filtered_df[filtered_df['round_score'] >= 5]
bad_images = filtered_df[filtered_df['round_score']< 5]


if selected_quality == 'good quality' :
    for _, row in good_images.sample(n=5).iterrows():
        try:
            image_path = f"dataset_tad66\selected_images\{str(row['image'])}"
            print(image_path)
            image = Image.open(image_path)
            predicted_score = predicted_scores.loc[predicted_scores['image_name'] == str(row['image']), selected_model + '_score'].values[0]
            print(predicted_score)
            st.image(image, caption=f"Expected quality_score :{row['score']}, Predicted_score : {round(predicted_score,2)}") 
        except:
           pass
    

else :
    for _, row in bad_images.sample(n=5).iterrows():
        image_path = f"dataset_tad66\selected_images\{str(row['image'])}"
        try :
            image = Image.open(image_path)
            print(image_path)
            predicted_score = predicted_scores.loc[predicted_scores['image_name'] == str(row['image']), selected_model + '_score'].values[0]
            st.image(image, caption=f"Expected quality_score :{row['score']}, Predicted_score : {round(predicted_score,2)}")
        
        except:
            pass