from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers

import numpy as np
import xgboost
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_validate
import random
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras.layers import concatenate

seed= 0
random.seed(seed)
os.chdir(".."); os.chdir("..")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

feature_names = ['psr', 'psl', 'fix_r_dur', 'fix_l_dur', 'fix_r_disx', 'fix_l_disx', 'fix_r_disy', 'fix_l_disy', 'fix_r_c', 'fix_l_c', 'sac_r_dur', 'sac_l_dur', 'sac_l_ampl', 'sac_r_ampl', 'sac_l_pv', 'sac_r_pv', 'sac_l_x', 'sac_l_y', 'sac_r_x', 'sac_r_y', 'sac_r_c', 'sac_l_c', 'psr_poly_1', 'psr_poly_2', 'psr_poly_3', 'psl_poly_1', 'psl_poly_2', 'psl_poly_3']
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
        if row['time'] < times:
            curr_trial += 1
        times = row['time']
        trial_number.append(curr_trial)
    data['trial'] = trial_number
    return data

data = pd.read_csv('/Users/or.brezel/Documents/HagitMac/Desktop/ATC_exp4_Jan2019/Matlab/data/result_better_features.csv')
data = add_trial_num(data)

data['subject_trial'] = data['subject'].astype(str) +"_"+ data['trial'].astype(str)
groups = data['subject']
subject_trial = data['subject_trial']

logo = LeaveOneGroupOut()


#### non-NN methods ##############
X = data[feature_names]
y = data[target_feature_name]

# #best_params = run_optuna_xgboost(X,y,groups)
# best_params = {'n_estimators': 89, 'max_depth': 9, 'reg_alpha': 4, 'reg_lambda': 4, 'min_child_weight': 3, 'gamma': 1, 'learning_rate': 0.006034574087944653, 'subsample': 0.674615682511142, 'colsample_bytree': 0.5}
# scores = cross_validate(xgboost.XGBClassifier(**best_params), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
# print('auc XGBost optimized',sum(scores['test_score'])/len(scores['test_score']))
# #auc XGBost optimized 0.727

# scores = cross_validate(xgboost.XGBClassifier(), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
# print('auc XGBost',sum(scores['test_score'])/len(scores['test_score']))
# #0.71

# scores = cross_validate(RandomForestClassifier(), X, y, cv=logo.split(X, y,groups),scoring='roc_auc')
# print('auc random forest',sum(scores['test_score'])/len(scores['test_score']))
# #0.7
#########################


num_cols = len(feature_names)


# convert an array of values into a dataset matrix
def create_dataset(dataset, output, target_att, subject, window_size=2,padding_const = 0 ):
    for end_idx in range(len(dataset)):
        start_idx = end_idx - window_size + 1
        padding = 0-start_idx #TODO: Can use padding for that
        start_idx = max(0,start_idx)
        for i,c in enumerate(feature_names):
            a = []
            if padding>0:
                a = [padding_const] * padding
            a = np.concatenate((a,dataset[c].values[start_idx:end_idx+1]))
            output[i].append(a)
        subject.append(dataset.subject.values[end_idx])
        target_att.append(dataset[target_feature_name].values[end_idx])
    return output


# for sub_trial in data['subject_trial']:
#     create_dataset(data.loc[data['subject_trial']==sub_trial])

window_size = 20

patient = []
target_att=[]
output = [[] for x in range(len(feature_names))]
for sub_trial in data.subject_trial.unique():
    curr_subj_trial_data = data.loc[data.subject_trial==sub_trial,:]
    output = create_dataset(curr_subj_trial_data, output, target_att, patient, window_size=window_size)
    print(len(output[0]))

X_nn = output
X = data[feature_names]
y = data[target_feature_name]

lstm_size = 16

from tensorflow import keras
inputs = [keras.Input((window_size, 1)) for x in feature_names]
# merge_one = layers.Concatenate(axis=-1)(inputs)


# Add 2 bidirectional LSTMs
blstm = [layers.LSTM(lstm_size)(x) for x in inputs] # input_shape=(lookback, num_cols)
# x = layers.Bidirectional(layers.LSTM(4))(x)
# Add a classifier
output_layer = layers.concatenate(inputs=blstm, axis=1)

outputs = layers.Dense(1, activation="sigmoid")(output_layer)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(optimizer="Adam", loss="binary_crossentropy")


results = {'auc':[]}
for train_idx,test_idx in logo.split(output[0], target_att,patient):
    print(train_idx,test_idx)
    train_data = [np.array(x)[train_idx] for x in X_nn]
    y_train = np.array(target_att)[train_idx]
    test_data = [np.array(x)[test_idx] for x in X_nn]
    y_test = np.array(target_att)[test_idx]

    model.fit(train_data, y_train)
    preds = model.predict(test_data)#[:,0]
    curr_auc = roc_auc_score(y_test, preds)
    print(curr_auc)
    results['auc'].append(curr_auc)
print(pd.DataFrame(results).mean())


#TODO: check that the number of test instances are the same - compare head-to-head with XGBoost
# Try BLSTM instead of LSTM
# Stack more LSTM layers and other parameters

print('num instances in NN data',len(output[0]))
print('num instances in original data',len(data))