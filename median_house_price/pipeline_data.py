from custom_transformers import CombinedAttributesAdder, DataFrameSelector, CustomBinarizer
from sklearn.pipeline import FeatureUnion, Pipeline 
from sklearn.preprocessing import StandardScaler
from split_data import Split
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
import os

class TransformationPipeline(Split):

	def __init__(self,data_path):
		super().__init__(data_path)

	def prepare_data(self):
		self.strat_train_set, self.strat_test_set = self.strat_split_data(0.2)
		self.housing_predictors = self.strat_train_set.drop('median_house_value',axis=1)
		self.housing_labels = self.strat_train_set['median_house_value'].copy()
		self.housing_num = self.housing_predictors.drop('ocean_proximity',axis=1)

	def transform_pipeline(self):
		self.prepare_data()
		num_attribs = list(self.housing_num)
		cat_attribs = ['ocean_proximity']

		num_pipeline = Pipeline([
								 ('selector', DataFrameSelector(num_attribs)),
								 ('imputer',Imputer(strategy='median')),
								 ('attribs_adder',CombinedAttributesAdder()),
								 ('std_scaler',StandardScaler()),
								 ])

		cat_pipeline = Pipeline([
								 ('selector', DataFrameSelector(cat_attribs)),
								 ('label_binarizer',CustomBinarizer()),
								 ])

		full_pipeline = FeatureUnion(transformer_list = [
									('num_pipeline',num_pipeline),
									('cat_pipeline',cat_pipeline)
									])

		print(full_pipeline)

		return full_pipeline.fit_transform(self.housing_predictors)

if __name__ == '__main__':
	HOUSING_PATH = os.path.relpath('housing/housing.csv')
	tfp = TransformationPipeline(HOUSING_PATH)
	housing_prepared = tfp.transform_pipeline()
	print(housing_prepared.shape)


