import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class HousingData:

	def __init__(self,data_path):
		self.data_path = data_path
		self.housing_data = pd.read_csv(self.data_path)
		self.housing_data_with_id = self.housing_data.reset_index()

	def describe(self):
		return self.housing_data.describe()

	def plot_median_house_value(self):
		self.housing_data.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,
		s=self.housing_data['population']/100,label='population',c='median_house_value',
		cmap=plt.get_cmap('jet'),colorbar=True,
		)
		plt.legend()
		plt.show()

	def corr_plot_between(self,main_corr, other_attribs=None):
		if isinstance(other_attribs,str):
			self.housing_data.plot(kind='scatter',x=main_corr,y=other_attribs,alpha=0.1)
			plt.show()
			return 0
		else:
			attributes = [main_corr]+other_attribs
			pd.plotting.scatter_matrix(self.housing_data[attributes],figsize=(12,8))
			plt.show()
			return 0

	def lon_lat_scatter_plot(self,alpha=1):
		self.housing_data.plot(kind='scatter',x='longitude',y='latitude',alpha=alpha)
		plt.show()
		

	def corr_matrix_for(self,correlate_with_text):
		correlation_matrix = self.housing_data.corr()
		return correlation_matrix[correlate_with_text].sort_values(ascending=False)



if __name__ == '__main__':
	HOUSING_PATH = os.path.relpath('housing/housing.csv')
	housing = HousingData(HOUSING_PATH)
	print(housing.describe())
	print(housing.corr_matrix_for('median_house_value'))
	housing.plot_median_house_value()
	housing.corr_plot_between('median_house_value',['median_income','total_rooms','housing_median_age'])
	housing.corr_plot_between('median_income','median_house_value')
	housing.lon_lat_scatter_plot(alpha=0.1)



