# Corona-Predict
A Repositroy which uses machine learning models to predict the cases for COVID - 19 in India statewise for a given number of days.
This application is deployed on heroku so that the mobile app can be used to predict the coronavirus cases.

## Model Used
I have used HOLT time series model for predicting the trend of cases of COVID - 19 from the past date for a state for the next n number of days

## REST API with Flask
Apart from building the model I have deployed the model on Herok platform as a REST API\
Link : https://corona-predd.herokuapp.com/

There are two inputs to the link which is sent to the server using GET method
1. state
2. days

Example : 
state : Tamil Nadu
days : 15

Link : https://corona-predd.herokuapp.com/predict?state=Tamil%20Nadu&days=10
