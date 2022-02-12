# Imports
import csv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# Containers
sidebar = st.container()
header = st.container()
modelTraining = st.container()
results = st.container()
machine_learning = st.container()
dataset = st.container()

# Returns water potability results using the random forest classifer
def analyze_water(user_ph_level, user_hardness_level, user_solids_level, user_chloramines_level, 
    user_sulfate_level, user_conductivity_level, user_organic_carbon_level, user_trihalomethanes_level,
    user_turbidity_level):
    
    # Create DF of user entered water metric data
    user_sample_df = pd.DataFrame([user_ph_level, user_hardness_level, user_solids_level, user_chloramines_level, 
        user_sulfate_level, user_conductivity_level, user_organic_carbon_level, user_trihalomethanes_level,
        user_turbidity_level])

    # Transpose DF so that it can be evaluated by RF
    user_sample_df_formatted = user_sample_df.T
    
    # Import random forest 
    clf = load('random_forest_clf/rf_water_quality_estimator')

    with results:
        unsafe_water_prob = clf.predict_proba(user_sample_df_formatted)[0][0]
        safe_water_prob = clf.predict_proba(user_sample_df_formatted)[0][1]

        if unsafe_water_prob > safe_water_prob:
            explode = [0.1, 0]
            st.write("The water is predicted to be unsafe to drink!")
        elif safe_water_prob > unsafe_water_prob:
            explode = [0, 0.1]
            st.write("The water is predicted to be safe to drink!")
        else:
            explode = [0, 0]
        
        # Display pie chart
        labels = "Probability that water is not safe to drink", "Probablity that water is safe to drink"
        fig, ax = plt.subplots()
        ax.pie([unsafe_water_prob, safe_water_prob], explode = explode, labels = labels, autopct='%1.1f%%', shadow = True,
                startangle=90)
        ax.axis('equal')
        plt.show()
        st.pyplot(plt)

# Sidebar is used to gather user input
with sidebar: 
    st.sidebar.header("Test Your Water Potability")
    user_ph_level = st.sidebar.slider(label = "Enter PH Level", max_value = 14)
    user_hardness_level = st.sidebar.slider(label = "Enter Hardness", min_value = 75, max_value = 300)
    user_solids_level = st.sidebar.slider(label = "Enter Solids", min_value = 320, max_value = 57000)
    user_chloramines_level = st.sidebar.slider(label = "Enter Chloramines", min_value = 1, max_value = 14)
    user_sulfate_level = st.sidebar.slider(label = "Enter Sulfate", min_value = 125, max_value = 500)
    user_conductivity_level = st.sidebar.slider(label = "Enter Conductivity", min_value = 200, max_value = 760)
    user_organic_carbon_level = st.sidebar.slider(label = "Enter Organic Carbon", min_value = 2, max_value = 28)
    user_trihalomethanes_level = st.sidebar.slider(label = "Enter Trihalomethanes", min_value = 8, max_value = 125)
    user_turbidity_level = st.sidebar.slider(label = "Enter Turbidity", min_value = 1, max_value = 7)
    user_analyzie_button = st.sidebar.button(label = "Analyze Water")
    if user_analyzie_button:
        analyze_water(user_ph_level, 
        user_hardness_level, user_solids_level, user_chloramines_level, user_sulfate_level, user_conductivity_level, 
        user_organic_carbon_level, user_trihalomethanes_level, user_turbidity_level)

with header:
    st.title('Water Potability Classifier')

with modelTraining:
    st.header("Your Water Sample Results:")
    st.write("Use the sidebar to predict whether your water sample is safe to drink.")

with machine_learning:
    st.header("Machine Learning Algorithm (Random Forest):")
    st.write("Confusion matrix displaying random forest model performance on test set.")
    st.image('data/confusion_matrix.png')

with dataset:
    st.header("Dataset Analysis:")
    st.write("The machine learning model was fit using over 2000 water samples. Statistics regarding the dataset" +
             " are shown in the following sections.")
    st.subheader("Descriptive Statistics Table Regarding Dataset")
    water_data = pd.read_csv("data/water_potability.csv")
    water_data = water_data.dropna()
    st.write(water_data.describe())
    
    # Histograms
    st.subheader("Dataset Features (Histogram)")

    # PH histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('PH Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['ph'])
    ax.set(xlabel="PH",
           ylabel="Number of Water Samples")
    st.pyplot(plt)

    #Columns
    col1,col2 = st.columns(2)

    #Hardness Histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('Hardness Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['Hardness'])
    ax.set(xlabel="Hardness",
           ylabel="Number of Water Samples")
    col1.pyplot(plt)

    #Solids Histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('Solids Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['Solids'])
    ax.set(xlabel="Solids",
           ylabel="Number of Water Samples")
    col2.pyplot(plt)

    #Chloramines Histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('Chloramines Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['Chloramines'])
    ax.set(xlabel="Chloramines",
           ylabel="Number of Water Samples")
    col1.pyplot(plt)

    #Sulfate Histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('Sulfate Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['Sulfate'])
    ax.set(xlabel="Sulfate",
           ylabel="Number of Water Samples")
    col2.pyplot(plt)

    #Conductivity Histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('Conductivity Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['Conductivity'])
    ax.set(xlabel="Conductivity",
           ylabel="Number of Water Samples")
    col1.pyplot(plt)

    #Organic Carbon Histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('Organic Carbon Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['Organic_carbon'])
    ax.set(xlabel="Organic Carbon",
           ylabel="Number of Water Samples")
    col2.pyplot(plt)

    #Trihalomethanes Histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('Trihalomethanes Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['Trihalomethanes'])
    ax.set(xlabel="Trihalomethanes",
           ylabel="Number of Water Samples")
    col1.pyplot(plt)

    #Turbidity Histogram
    fig, ax = plt.subplots(figsize = (10,5))
    fig.suptitle('Turbidity Level Across Dataset Samples', fontsize=16, fontweight='bold');
    ax.hist(water_data['Turbidity'])
    ax.set(xlabel="Turbidity",
           ylabel="Number of Water Samples")
    col2.pyplot(plt)