import glob
from dataloader.utils.utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import json
from imblearn.combine import SMOTEENN
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
from sklearn.metrics import accuracy_score

import sys
log_file = open("regression_results.log","a+")
sys.stdout = log_file


def rounded_pred(y_pred):
    pred_round = np.round(y_pred)
    pred_round = np.abs(np.where(pred_round<0, 0, pred_round))

    return pred_round


parser = argparse.ArgumentParser()
parser.add_argument('-c', "--celltype",)

args = parser.parse_args()
celltype = args.celltype
celltype_marker_mapping = json.loads(open("celltype_markerGene_map.json").read())
markers = celltype_marker_mapping[celltype]

seed = 123

print(f"------------- RUNNING Regression for : {celltype} ------------")
# read file and preprocess it

print("$$$$ READING THE FILE $$$$")
file_paths = glob.glob(f"./output/sc_heart/simulation_results/*/RCTD_simu_cell_level.h5ad")
x_all, y_all = get_all_simulation_data(file_paths,celltype,markers)

X,y = ([],[])
for idx, y_train_individual in enumerate(y_all):
    if y_train_individual > 10: 
        continue
    X.append(x_all[idx])
    y.append(y_all[idx])

X_train, X_test, y_train, y_test = get_train_and_test_data(X,y,123)
print("$$$$ READING Done $$$$")

# balance the data
print("$$$$ BALANCING THE DATA $$$$")
X,y = X_train, y_train
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
print("$$$$ BALANCING Done! $$$$")


# train classifier
# define model
model = XGBRegressor(objective ='reg:linear', 
                  n_estimators = 10000, seed = 123) 
# define model evaluation method
model.fit(X_resampled, y_resampled)
print("#### Training Done ####")

# train accuracy
y_pred = model.predict(X_train)

y_pred_round = rounded_pred(y_pred)
accuracy = accuracy_score(y_train, y_pred_round)
print('Train Accuracy: %.2f%%' % (accuracy * 100.0))

# train rmse
rmse = np.sqrt(MSE(y_train, model.predict(X_train))) 
print(f'Train rmse: {rmse}')


# test accuracy
y_pred = model.predict(X_test)

y_pred_round = rounded_pred(y_pred)
accuracy = accuracy_score(y_test, y_pred_round)
print('Test Accuracy: %.2f%%' % (accuracy * 100.0))

# test rmse
rmse = np.sqrt(MSE(y_test, model.predict(X_test))) 
print(f'Test rmse: {rmse}')