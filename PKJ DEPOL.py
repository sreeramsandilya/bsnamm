# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:38:40 2023

@author: DELL
"""

import pickle
import pandas as pd
import streamlit as st

jobs_df = pickle.load(open('JOB DATA.pkl', 'rb'))

tfidf_matrix = pickle.load(open('cosine1_sim.pkl', 'rb'))

st.set_page_config(layout="centered")

def welcome():
    return "Welcome All"


# Build the recommendation model
def get_recommendations(job_title, num_recommendations):
    job_idx = jobs_df[jobs_df['Job_Name'] == job_title].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[job_idx], tfidf_matrix)
    similar_jobs_indices = cosine_similarities.argsort()[0][-num_recommendations-1:-1]
    return jobs_df.iloc[similar_jobs_indices]

st.title('Jobs Recommender System')

option = st.selectbox('Select your Job: ', jobs_df['Job_Name'].values)

if st.button('Click here to Recommend'):
    recommendation = get_recommendations(option)
    for i, row in recommendation.iterrows():
    st.write('Job Title: ', row['Job_Name'])
    #st.write("Job Title: ", row['Job Title Cleaned'], " | Company: ", row['Company'], " | Industry: ", row['Industry'],"| Type of Role: ", row['Type of role cleaned'])
    st.write('Company: ', row['Company'])
   