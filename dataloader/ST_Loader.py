import glob
from utils.utils import *
import matplotlib.pyplot as plt

file_paths = glob.glob(f"../output/sc_heart/simulation_results/*/RCTD_simu_cell_level.h5ad")

print(file_paths)

x_all, y_all = get_all_simulation_data(file_paths, 0, ['S100A4', 'VIM', 'CD44', 'SPARC', 'KLF6', 'MDK'])

X_train, X_test, y_train, y_test = get_train_and_test_data(x_all,y_all,123)
# print(x_all)
# print(y_all)

from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor

# create model instance
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(X_train, y_train)

# make predictions
preds = my_model.predict(X_test)

print(f"{preds[:20]} {y_test[:20]}")