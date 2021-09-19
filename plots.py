import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns 

def app(cars_df):
	st.header('Visualise Data')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.header('Scatter Plot')
	features_list = st.multiselect('Select the x-axis value', ('carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick'))
	for i in features_list:
		st.subheader(f'Scatter plot between{i} and price')
		plt.figure(figsize=(9,9))
		sns.scatterplot(x=i, y='price', data=cars_df)
		st.pyplot()

	st.subheader('Visualisation Sector')
	plot_types = st.multiselect('Select the plot', ('Histogram','Boxplot','Heatmap'))
	if 'Histogram' in plot_types:
		st.subheader('Histogram')
		columns = st.selectbox('Select the columns to create Histogram', ('carwidth', 'enginesize', 'horsepower'))
		plt.figure(figsize=(9,9))
		plt.hist(cars_df[columns], bins='sturges', edgecolor='black')
		st.pyplot()

	if 'Boxplot' in plot_types:
		st.subheader('Boxplot')
		columns = st.selectbox('Select columns to create Boxplot', ('carwidth', 'enginesize', 'horsepower'))
		plt.figure(figsize=(9,9))
		sns.boxplot(cars_df[columns])
		st.pyplot()

	if 'Heatmap' in plot_types:
		st.subheader('Heatmap')
		plt.figure(figsize=(9,9))
		ax = sns.heatmap(cars_df.corr(), annot=True)
		bottom, top = ax.get_ylim()
		ax.set_ylim(bottom+0.5, top-0.5)
		st.pyplot()