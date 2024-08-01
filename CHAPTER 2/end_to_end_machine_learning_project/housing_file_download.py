import constants as const
import tarfile 
import urllib.request
import os
import pandas as pd
import matplotlib.pyplot as plt
# Download data
def fetch_housing_data(housing_url=const.HOUSING_URL, housing_path=const.HOUSING_PATH):
	os.makedirs(housing_path, exist_ok=True)
	tgz_path = os.path.join(housing_path, 'housing.tgz')
	urllib.request.urlretrieve(housing_url, tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()
	
fetch_housing_data()

# Load data with pandas
def load_housing_data(housing_path=const.HOUSING_PATH):
	csv_path = os.path.join(housing_path, 'housing.csv')
	return pd.read_csv(csv_path)