import numpy as np 
import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error

@st.cache()
def prediction(cars_df, car_width, engine_size, horse_power, drivewheel_forward, car_company_buick):
	X = cars_df.iloc[:,:-1]
	y = cars_df['price']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	lr = LinearRegression()
	lr.fit(X_train, y_train)
	score = lr.score(X_train, y_train)

	price = lr.predict([[car_width, engine_size, horse_power, drivewheel_forward, car_company_buick]])
	price = price[0]

	y_test_pred = lr.predict(X_test)

	test_r2_score = r2_score(y_test, y_test_pred)
	test_msle = mean_squared_log_error(y_test, y_test_pred)
	test_mae = mean_absolute_error(y_test, y_test_pred)
	test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

	return price, score, test_r2_score, test_msle, test_mae, test_rmse

def app(cars_df):
	st.markdown("<p style='color:blue;font-size:25px'> This app uses <b> LinearRegression </b> To predict the price of a car based on your input")
	st.subheader('Select the values')
	car_wid = st.slider('carwidth', float(cars_df['carwidth'].min()), float(cars_df['carwidth'].max()))
	eng_size = st.slider('enginesize', int(cars_df['enginesize'].min()), int(cars_df['enginesize'].max()))
	horse_pow = st.slider('horsepower', int(cars_df['horsepower'].min()), int(cars_df['horsepower'].max()))
	dr_fwd = st.radio('Is it a forward drive wheel car?', ('Yes', 'No'))
	if dr_fwd == 'No':
		dr_fwd = 0
	else:
		dr_fwd = 1

	car_bui = st.radio('Is the car manufactured by Buick?', ('Yes', 'No'))
	if car_bui == 'No':
		car_bui = 0

	else:
		car_bui = 1

	if st.button('Predict'):
		st.subheader('Prediction Results')
		price, score, car_r2_score, car_msle, car_mae, car_rmse = prediction(cars_df, car_wid, eng_size, horse_pow, dr_fwd, car_bui)
		st.success(f'The predicted price of the car is {int(price)}')
		st.info(f'Accuracy score of model is {score:.3f}')
		st.info(f'R2 score is{car_r2_score:.3f}')
		st.info(f'MSLE is {car_msle:.3f}')
		st.info(f'MAE is {car_mae:.3f}')
		st.info(f'RMSE is {car_rmse:.3f}')