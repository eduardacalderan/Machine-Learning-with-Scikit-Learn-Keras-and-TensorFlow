from zlib import crc32
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import housing_file_download
from sklearn.model_selection import StratifiedShuffleSplit
import os

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
  path = os.path.join(r'CHAPTER 2/end_to_end_machine_learning_project/images/', fig_id + "." + fig_extension)
  print("Saving figure", fig_id)
  if tight_layout:
      plt.tight_layout()
  plt.savefig(path, format=fig_extension, dpi=resolution)

housing = housing_file_download.load_housing_data()
print(housing.head())

housing.hist(bins=50, figsize=(20,15)) # hist will plot a histogram for each numeric attribute
plt.show()

# Test suit
def split_train_test(data, test_ratio):
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]

  return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))

def test_set_check(indentifier, test_ratio):
  return crc32(np.int64(indentifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
  ids = data[id_column]
  in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
  return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()
plt.show()

# stratified sampling based on income category - amostragem estratificada com base na categoria da renda
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]

# análise das proporções da categoria de renda no conjunto de testes 
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# removing "income_cat" for the data come back to original state
for set_ in (strat_train_set, strat_test_set):
  set_.drop("income_cat", axis=1, inplace=True)

# Copiando  o conjunto de treinamento para manipular sem prejudicar ele
housing = strat_test_set.copy() 

# VISUALIZANDO DADOS GEOGRÁFICOS
# diagrama de dispersão
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) # definir alpha para 0.1 facilita a visualização dos locais em que há alta densidade de pontos de dados
save_fig("better_visualization_plot")
print('end')

# Vendo os preços dos imóveis
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()

save_fig("legend_visualization_plot")
