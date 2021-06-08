#Reading data & prepearing
import os
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_validate

from src.constants import *
from src.functions import *
os.chdir(".."); os.chdir("..")

data = pd.read_csv(os.path.join('data','result_better_features.csv'))
data = add_trial_num(data)
data,feature_names = average_l_r(data,feature_names)
data['subject_trial'] = data[subject_feature_name].astype(str) +"_"+ data[trial_feature_name].astype(str)
groups = data[subject_feature_name]
subject_trial = data['subject_trial']
num_cols = len(feature_names)



logo = LeaveOneGroupOut()



### non-NN methods ##############
X = data[feature_names].copy()
y = data[target_feature_name].copy()

#best_params = run_optuna_xgboost(X,y,groups)
best_params = {'n_estimators': 89, 'max_depth': 9, 'reg_alpha': 4, 'reg_lambda': 4, 'min_child_weight': 3, 'gamma': 1, 'learning_rate': 0.006034574087944653, 'subsample': 0.674615682511142, 'colsample_bytree': 0.5}
scores = cross_validate(xgboost.XGBClassifier(**best_params), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
print('auc XGBost optimized',sum(scores['test_score'])/len(scores['test_score']))
#auc XGBost optimized 0.727

scores = cross_validate(xgboost.XGBClassifier(), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
print('auc XGBost',sum(scores['test_score'])/len(scores['test_score']))
#0.71

scores = cross_validate(RandomForestClassifier(), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
print('auc random forest',sum(scores['test_score'])/len(scores['test_score']))
#0.7
########################
