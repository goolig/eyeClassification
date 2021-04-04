import xgboost
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_validate
import random
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from src.main.hyper_params_XGBOOST import run_optuna_xgboost

seed= 0
random.seed(seed)
os.chdir(".."); os.chdir("..")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

feature_names = ['median_pupil_size_1', 'median_pupil_size_2','poly_pupil_size_1', 'poly_pupil_size_2', 'max_sac_1', 'max_sac_2',
       'count_sac_1', 'count_sac_2', 'diff_pos_x_1', 'diff_pos_x_2', 'diff_pos_y_1', 'diff_pos_y_2']
target_feature_name = 'load'


def add_trial_num(data):
    times = -1
    curr_user = -1
    curr_trial = 0
    trial_number = []
    for i, row in data.iterrows():
        if row['subject'] != curr_user:
            curr_user = row['subject']
            times = -1
            curr_trial = 0
        if row['times'] < times:
            curr_trial += 1
        times = row['times']
        trial_number.append(curr_trial)
    data['trial'] = trial_number
    return data

data = pd.read_csv(os.path.join('data','result_correct.csv'))
data = add_trial_num(data)

data['subject_trial'] = data['subject'].astype(str) +"_"+ data['trial'].astype(str)
groups = data['subject']
subject_trial = data['subject_trial']

logo = LeaveOneGroupOut()

X = data[feature_names]
y = data[target_feature_name]

# #best_params = run_optuna_xgboost(X,y,groups)
# best_params = {'n_estimators': 89, 'max_depth': 9, 'reg_alpha': 4, 'reg_lambda': 4, 'min_child_weight': 3, 'gamma': 1, 'learning_rate': 0.006034574087944653, 'subsample': 0.674615682511142, 'colsample_bytree': 0.5}
# scores = cross_validate(xgboost.XGBClassifier(**best_params), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
# print('auc XGBost optimized',sum(scores['test_score'])/len(scores['test_score']))
# #auc XGBost optimized 0.727
#
#
#
# scores = cross_validate(xgboost.XGBClassifier(), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
# print('auc XGBost',sum(scores['test_score'])/len(scores['test_score']))
# #0.71
#
# scores = cross_validate(RandomForestClassifier(), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
# print('auc random forest',sum(scores['test_score'])/len(scores['test_score']))
# #0.7
#
#
# from sktime.classification.shapelet_based import MrSEQLClassifier
# clf = MrSEQLClassifier()
# clf.fit(X, y)


from sktime.forecasting.model_selection import (
    CutoffSplitter,
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
window_length=1
cv = SlidingWindowSplitter(window_length=window_length, start_with_window=True,fh=0)
y_out = []
for subj_tri in set(subject_trial):
    #cv.get_n_splits(y[groups==subj])
    curr_y = y[subject_trial==subj_tri]
    curr_X = X[subject_trial == subj_tri]
    for i, (w, f) in enumerate(cv.split(curr_y)):
        y_out.append(curr_y.iloc[f[0]])

print(len(y_out))
print(len(y))
