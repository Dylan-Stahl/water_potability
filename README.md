# water_potability

## Application Need Scenario
Phoenix Water Testing Services (PWTS) offers testing services for residents and businesses in the Greater Phoenix Region and needs a faster water 
potability testing strategy. Competing water testing companies have created faster-testing solutions than PWTS. If PWTS had a testing solution 
using machine learning, the company’s testing times would be significantly reduced, making PWTS a stronger competitor. By creating a web-based machine-learning 
application that can analyze a water sample, PWTS employees can quickly process tests, increasing customer satisfaction.

## Application Description
The Water Potability Classification application is a data science-based application that uses machine learning techniques to improve PWTS testing strategies. 
Upon opening the web application, users will notice that the application is simple to use. A sidebar on the left side of the page requests water data input from 
the user to predict the potability of the water sample. Information about the machine learning algorithm is provided underneath the prediction data, along with
a dataset analysis that shows the user some statistics regarding the dataset used.
	
This application will be created using Python for the backend as it is one of the best languages to use for machine learning because of the large number of
libraries that support it. The Python libraries that will be used include Pandas, NumPy, Matplotlib, and Scikit-learn. Matplotlib will be particularly useful 
for displaying graphs on the website. The Scikit-learn library provides all the machine learning models and metrics for those models. Lastly, Jupyter Notebook 
will be used to create and train the machine learning model.
	
Streamlit is being proposed to be used for the front end. Streamlit is an app framework to be used in Python. It is compatible with the Python libraries
needed for this project and can display Matplotlib graphs nicely. Streamlit is an excellent choice for this project as it allows for a simple, easy-to-use, 
interactive user website with minimal code needed. Streamlit’s API includes widgets that can be used on the website. For example, slider input widgets can be 
used to get users’ input water samples.  

## Access to Project:
Users can run the finished web application on the latest Google Chrome, Firefox, Microsoft Edge, and Safari browsers. 
It can be accessed here: https://waterpotabilityclf.herokuapp.com/.

## How to Use:
The sidebar on the left side of the page is where the user will enter water samples. Use the sliders to adjust the values. 
Click the “Analyze Water” button below the sliders to calculate the results when all the values are entered. 
Under the “Your Water Sample Results:” section will be the prediction along with the probability that the machine learning model thinks it is correct.

## Dataset Description
The dataset that will be used to train the machine learning model (random forest classifier) can be located here and is in CSV format: 
https://www.kaggle.com/adityakadiwal/water-potability. There are 3,276 water data quality samples provided in the dataset. Unfortunately, only 2,011 
samples can be used for training and testing purposes because all samples with missing attributes will be removed.
