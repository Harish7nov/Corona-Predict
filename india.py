# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from datetime import timedelta
import warnings
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import itertools
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import Holt
from flask import Flask
from flask import request
import json
warnings.filterwarnings("ignore")


#Mean absolute percentage error
def mape(y2, y_pred): 
		y2, y_pred = np.array(y2), np.array(y_pred)
		return np.mean(np.abs((y2 - y_pred) / y2)) * 100

def split(ts):
		#splitting 85%/15% because of little amount of data
		train = ts.iloc[:int(ts.shape[0]*0.966)]
		test = ts.iloc[int(ts.shape[0]*0.966):]
		return(train,test)


# current_directory = os.getcwd()
# final_directory = os.path.join(current_directory, r'content')
# if not os.path.exists(final_directory):
# 	 os.makedirs(final_directory)

def pred(state, day):

	covid = pd.read_csv("covid_19_india.csv",parse_dates=[0], dayfirst=True)

	#Dropping the column
	covid.drop(["Time"],axis=1,inplace=True)
	covid.drop(["Sno"],axis=1,inplace=True)

	covid["Date"] = pd.to_datetime(covid["Date"], infer_datetime_format=True)
	
	tn_data = covid[covid["State/UnionTerritory"]==state]
	tn_data.set_index("Date", inplace = True) 
	datewise_tn= tn_data.drop(["ConfirmedIndianNational","ConfirmedForeignNational"],axis=1)
	
	datewise_tn["WeekofYear"] = datewise_tn.index.weekofyear
	week_num=[]
	weekwise_confirmed=[]
	weekwise_recovered=[]
	weekwise_deaths = []
	w=1
	count = 0
	for i in list(datewise_tn["WeekofYear"].unique()):
			weekwise_confirmed.append(datewise_tn[datewise_tn["WeekofYear"]==i]["Confirmed"].iloc[-1])
			weekwise_recovered.append(datewise_tn[datewise_tn["WeekofYear"]==i]["Cured"].iloc[-1])
			weekwise_deaths.append(datewise_tn[datewise_tn["WeekofYear"]==i]["Deaths"].iloc[-1])
			week_num.append(w)
			w=w+1
			count += 1

	plot1_x = week_num
	plot1_y = weekwise_confirmed
	

	max_tn = datewise_tn["Confirmed"].max()

	datewise_tn["Days Since"] = datewise_tn.index-datewise_tn.index[0]
	datewise_tn["Days Since"]=datewise_tn["Days Since"].dt.days

	train_ml = datewise_tn.iloc[:int(datewise_tn.shape[0]*0.95)]
	valid_ml = datewise_tn.iloc[int(datewise_tn.shape[0]*0.95):]
	model_scores = []

	lin_reg = LinearRegression(normalize=True)

	lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),
							np.array(train_ml["Confirmed"]).reshape(-1,1))

	prediction_valid_linreg= lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

	model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
	# print("RMS for LR",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))

	# plt.figure(figsize=(11,8))
	prediction_linreg = lin_reg.predict(np.array(datewise_tn["Days Since"]).reshape(-1,1))

	poly = PolynomialFeatures(degree = 8)

	train_poly = poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
	valid_poly = poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
	y = train_ml["Confirmed"]

	linreg = LinearRegression(normalize=True)
	linreg.fit(train_poly,y)

	prediction_poly = linreg.predict(valid_poly)
	rmse_poly = np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))
	model_scores.append(rmse_poly)
	# print("RMS For PR",rmse_poly)

	comp_data = poly.fit_transform(np.array(datewise_tn["Days Since"]).reshape(-1,1))
	# plt.figure(figsize=(11,6))
	predictions_poly = linreg.predict(comp_data)

	new_date=[]
	new_prediction_lr=[]
	new_prediction_poly=[]

	# Changed..
	for i in range(1,day):
			new_date.append(datewise_tn.index[-1]+timedelta(days=(i-1)))
			new_prediction_lr.append(lin_reg.predict(np.array(datewise_tn["Days Since"].max()+i).reshape(-1,1))[0][0])
			new_date_poly=poly.fit_transform(np.array(datewise_tn["Days Since"].max()+i).reshape(-1,1))
			new_prediction_poly.append(linreg.predict(new_date_poly)[0])

	pd.set_option('display.float_format', lambda x:'%f' %x)
	model_predictions = pd.DataFrame(list(zip(new_date,new_prediction_lr,new_prediction_poly)),columns = ["Dates","LRP","PRP"])
	model_predictions.head(5)

	model_train = datewise_tn.iloc[:int(datewise_tn.shape[0]*0.95)]
	valid = datewise_tn.iloc[int(datewise_tn.shape[0]*0.95):]

	model_train.head(4)

	holt = Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level = 0.3, smoothing_slope = 0.5, optimized=False)
	y_pred = valid.copy()

	y_pred["Holt"]=holt.forecast(len(valid))
	model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))

	holt_new_date= []
	holt_new_predictions =[]

	# Changed...
	for i in range(1,day):
			holt_new_date.append(datewise_tn.index[-1]+timedelta(days=(i-1)))
			holt_new_predictions.append(holt.forecast((len(valid)+i))[-1])
	model_predictions["Holts Model"]= holt_new_predictions
	model_predictions.head()

	train,test=split(datewise_tn)
	y_pred = test.copy()
	
	#Modeling
	model = ARIMA(train["Confirmed"], order=(2,2,0))
	result = model.fit()
	result.plot_predict(start=int(len(train) * 0.7), end=int(len(train) * 1.2))
	y_pred["ARIMA"]=result.forecast(steps=len(test))[0]      

	model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["ARIMA"])))

	arima_new_date= []
	arima_new_predictions =[]

	# Changed...
	for i in range(1,day):
			arima_new_date.append(datewise_tn.index[-1]+timedelta(days=i))
			arima_new_predictions.append(result.forecast((len(valid)+i))[0][-1])
	model_predictions["Arima Model"]=arima_new_predictions

	new_date=[]
	new_prediction_lr=[]
	new_prediction_poly=[]
	holt_new_date= []
	holt_new_predictions =[]
	arima_new_date= []
	arima_new_predictions =[]

	for i in range(1,day):
			new_date.append(str(datewise_tn.index[-1]+timedelta(days=i)).replace("00:00:00",'').strip())
			new_prediction_lr.append(int(lin_reg.predict(np.array(datewise_tn["Days Since"].max()+i).reshape(-1,1))[0][0]))
			new_date_poly=poly.fit_transform(np.array(datewise_tn["Days Since"].max()+i).reshape(-1,1))
			new_prediction_poly.append(int(linreg.predict(new_date_poly)[0]))
			holt_new_date.append(datewise_tn.index[-1]+timedelta(days=i))
			holt_new_predictions.append(int(holt.forecast((len(valid)+i))[-1]))
			arima_new_date.append(datewise_tn.index[-1]+timedelta(days=i))
			arima_new_predictions.append(int(result.forecast((len(valid)+i))[0][-1]))
			model_predictions = pd.DataFrame(list(zip	(new_date,new_prediction_lr,new_prediction_poly,holt_new_predictions,arima_new_predictions)),columns = ["Dates","LRP","PRP","HOLT","ARIMA"])
			
	# For predicting the death case

	model_scores_death = []

	lin_reg = LinearRegression(normalize=True)

	lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),
						np.array(train_ml["Deaths"]).reshape(-1,1))

	prediction_valid_linreg_death= lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))

	model_scores_death.append(np.sqrt(mean_squared_error(valid_ml["Deaths"],prediction_valid_linreg_death)))

	prediction_linreg_death = lin_reg.predict(np.array(datewise_tn["Days Since"]).reshape(-1,1))

	poly = PolynomialFeatures(degree = 8)

	train_poly = poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
	valid_poly = poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
	y = train_ml["Deaths"]

	linreg = LinearRegression(normalize=True)
	linreg.fit(train_poly,y)

	prediction_poly_death = linreg.predict(valid_poly)
	rmse_poly_death = np.sqrt(mean_squared_error(valid_ml["Deaths"],prediction_poly_death))
	model_scores_death.append(rmse_poly_death)

	comp_data_death = poly.fit_transform(np.array(datewise_tn["Days Since"]).reshape(-1,1))
	# plt.figure(figsize=(11,6))
	predictions_poly_death = linreg.predict(comp_data_death)
	new_date=[]
	new_prediction_lr_death=[]
	new_prediction_poly_death=[]

	# Changed...
	for i in range(1,day):
			new_date.append(datewise_tn.index[-1]+timedelta(days=(i-1)))
			new_prediction_lr_death.append(lin_reg.predict(np.array(datewise_tn["Days Since"].max()+i).reshape(-1,1))[0][0])
			new_date_poly_death=poly.fit_transform(np.array(datewise_tn["Days Since"].max()+i).reshape(-1,1))
			new_prediction_poly_death.append(linreg.predict(new_date_poly_death)[0])

	pd.set_option('display.float_format', lambda x:'%f' %x)
	model_predictions_death = pd.DataFrame(list(zip(new_date,new_prediction_lr_death,new_prediction_poly_death)),columns = 	["Dates","LRP_DEATHS","PRP_DEATHS"])
	# model_predictions_death.head(5)

	model_train_death = datewise_tn.iloc[:int(datewise_tn.shape[0]*0.95)]
	valid = datewise_tn.iloc[int(datewise_tn.shape[0]*0.95):]

	# model_train_death.head(4)

	holt = Holt(np.asarray(model_train_death["Deaths"])).fit(smoothing_level = 0.3, smoothing_slope = 0.5, optimized=False)
	y_pred_death = valid.copy()

	y_pred_death["Holt"]=holt.forecast(len(valid))
	model_scores_death.append(np.sqrt(mean_squared_error(y_pred_death["Deaths"],y_pred_death["Holt"])))

	holt_new_date= []
	holt_new_predictions_death =[]

	# changed...
	for i in range(1,day):
			holt_new_date.append(datewise_tn.index[-1]+timedelta(days=(i-1)))
			holt_new_predictions_death.append(holt.forecast((len(valid)+i))[-1])

	model_predictions_death["Holts Model"]= holt_new_predictions_death
	model_predictions_death.head()

	train,test=split(datewise_tn)
	y_pred_death = test.copy()
	
	#Modeling
	model = ARIMA(train["Deaths"], order=(2,2,0))
	result = model.fit()
	result.plot_predict(start=int(len(train) * 0.7), end=int(len(train) * 1.2))
	y_pred_death["ARIMA"]=result.forecast(steps=len(test))[0]

	model_scores_death.append(np.sqrt(mean_squared_error(y_pred_death["Deaths"],y_pred_death["ARIMA"])))

	arima_new_date= []
	arima_new_predictions_death =[]

	# Changed...
	for i in range(1,day):
			arima_new_date.append(datewise_tn.index[-1]+timedelta(days=i))
			arima_new_predictions_death.append(result.forecast((len(valid)+i))[0][-1])
	model_predictions_death["Arima Model"]=arima_new_predictions_death

	new_date=[]
	new_prediction_lr_death=[]
	new_prediction_poly_death=[]
	holt_new_date= []
	holt_new_prediction_death =[]
	arima_new_date= []
	arima_new_predictions_death =[]
	for i in range(1,day):
			new_date.append(str(datewise_tn.index[-1]+timedelta(days=i)).replace("00:00:00",'').strip())
			new_prediction_lr_death.append(int(lin_reg.predict(np.array(datewise_tn["Days Since"].max()+i).reshape(-1,1))[0]	[0]))
			new_date_poly_death=poly.fit_transform(np.array(datewise_tn["Days Since"].max()+i).reshape(-1,1))
			new_prediction_poly_death.append(int(linreg.predict(new_date_poly)[0]))
			holt_new_date.append(datewise_tn.index[-1]+timedelta(days=i))
			holt_new_predictions_death.append(int(holt.forecast((len(valid)+i))[-1]))
			arima_new_date.append(datewise_tn.index[-1]+timedelta(days=i))
			arima_new_predictions_death.append(int(result.forecast((len(valid)+i))[0][-1]))

	# model_predictions_death = pd.DataFrame(list(zip(new_date,new_prediction_lr,new_prediction_poly,holt_new_predictions,arima_new_predictions,new_prediction_lr_death,new_prediction_poly_death,holt_new_predictions_death,arima_new_predictions_death,plot1_x,plot1_y)), columns = ["Dates","LRP_CONFIRM","PRP_CONFIRM","HOLT_CONFIRM","ARIMA_CONFIRM","LRP_DEATH","PRP_DEATH","HOLT_DEATH","ARIMA_DEATH","plot1_x","plot1_y"])
	labels = ["Dates","LRP_CONFIRM","PRP_CONFIRM","HOLT_CONFIRM","ARIMA_CONFIRM","LRP_DEATH","PRP_DEATH","HOLT_DEATH","ARIMA_DEATH","plot1_x","plot1_y"]
	d = {}

	for i, j in zip(labels, [new_date,new_prediction_lr,new_prediction_poly,holt_new_predictions,arima_new_predictions,new_prediction_lr_death,new_prediction_poly_death,holt_new_predictions_death,arima_new_predictions_death,plot1_x,plot1_y]):
		d[i] = j

	model_predictions_death = d
	return json.dumps(model_predictions_death, default=myconverter)


def myconverter(obj):
	if isinstance(obj, np.integer):
		return int(obj)
	elif isinstance(obj, np.floating):
		return float(obj)
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	elif isinstance(obj, datetime.datetime):
		return obj.__str__()


app = Flask(__name__)

@app.route("/")
def index():
	return "Welcome to the Corona Virus Prediction App"

@app.route("/predict")
def home():
	return f"Works Fine : {request.args.get('state', ''), request.args.get('days', type=int)}"
# 	return pred(request.args.get('state', ''), request.args.get('days', type=int))

if __name__ == "__main__":
    app.run()
