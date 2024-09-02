import glob
from dataloader.utils.utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import json
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score

import sys
log_file = open("classification_results.log","a+")
sys.stdout = log_file

parser = argparse.ArgumentParser()
parser.add_argument('-c', "--celltype",)

args = parser.parse_args()
celltype = args.celltype
celltype_marker_mapping = json.loads(open("celltype_markerGene_map.json").read())
markers = celltype_marker_mapping[celltype]

seed = 123

print(f"------------- RUNNING Classification for : {celltype} ------------")

# read file and preprocess it

print("$$$$ READING THE FILE $$$$")
file_paths = glob.glob(f"./output/sc_heart/simulation_results/*/RCTD_simu_cell_level.h5ad")
x_all, y_all = get_all_simulation_data(file_paths,celltype,markers)

y_all_classifier = []
for y in y_all:
    if y==0:
        y_all_classifier.append(0)
    else:
        y_all_classifier.append(1)
y_all = y_all_classifier

X_train, X_test, y_train, y_test = get_train_and_test_data(x_all,y_all,123)
print("$$$$ READING Done $$$$")

# balance the data
print("$$$$ BALANCING THE DATA $$$$")
X,y = X_train, y_train
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
print("$$$$ BALANCING Done! $$$$")


# train classifier
# define model
print("#### Training Classifier ####")
model = XGBClassifier() 
# define model evaluation method
model.fit(X_resampled, y_resampled)
print("#### Training Done ####")

# train accuracy
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print('Train Accuracy: %.2f%%' % (accuracy * 100.0))


# test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy: %.2f%%' % (accuracy * 100.0))