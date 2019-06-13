from explore_data import HousingData
from sklearn.model_selection import StratifiedShuffleSplit
import os
import numpy as np 

class Split(HousingData):

	def __init__(self,data_path):
		super().__init__(data_path)

	def split_data(self,split_ratio,seed=42):
		np.random.seed(seed)
		shuffled_indicies = np.random.permutation(len(self.housing_data))
		test_set_size = int(len(self.housing_data)*split_ratio)
		test_indicies = shuffled_indicies[:test_set_size]
		train_indicies = shuffled_indicies[test_set_size:]
		return self.housing_data.iloc[test_indicies],self.housing_data.iloc[train_indicies]

	def strat_split_data(self,test_ratio,seed=42):
		split = StratifiedShuffleSplit(n_splits=1,test_size=test_ratio,random_state=seed)
		self.housing_data['income_cat'] = np.ceil(self.housing_data['median_income']/1.5)
		self.housing_data['income_cat'].where(self.housing_data['income_cat'] < 
											  5,5.0, inplace=True)
		for train_index, test_index in split.split(self.housing_data,self.housing_data['income_cat']):
			strat_train_set = self.housing_data.loc[train_index]
			strat_test_set = self.housing_data.loc[test_index]

		for set in (strat_train_set,strat_test_set):
			set.drop(['income_cat'],axis=1,inplace=True)

		return strat_train_set, strat_test_set


if __name__ == '__main__':
	HOUSING_PATH = os.path.relpath('housing/housing.csv')
	split = Split(HOUSING_PATH)
	housing_test, housing_train = split.split_data(split_ratio=0.2)
	print(housing_train)