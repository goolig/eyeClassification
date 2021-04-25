from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing

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

# feature_names = ['median_pupil_size_1', 'median_pupil_size_2','poly_pupil_size_1', 'poly_pupil_size_2', 'max_sac_1', 'max_sac_2',
#        'count_sac_1', 'count_sac_2', 'diff_pos_x_1', 'diff_pos_x_2', 'diff_pos_y_1', 'diff_pos_y_2']
feature_names = ['psr', 'psl', 'fix_r_dur', 'fix_l_dur',
       'fix_r_disx', 'fix_l_disx', 'fix_r_disy', 'fix_l_disy', 'fix_r_c',
       'fix_l_c', 'sac_r_dur', 'sac_l_dur', 'sac_l_ampl', 'sac_r_ampl',
       'sac_l_pv', 'sac_r_pv', 'sac_l_x', 'sac_l_y', 'sac_r_x', 'sac_r_y',
       'sac_r_c', 'sac_l_c', 'psr_poly_1', 'psr_poly_2', 'psr_poly_3',
       'psl_poly_1', 'psl_poly_2', 'psl_poly_3']

target_feature_name = 'load'
time_feature_name = 'time'#'times'
subject_feature_name = 'subject'
trial_feature_name = 'trial'

def add_trial_num(data):
    times = -1
    curr_user = -1
    curr_trial = 0
    trial_number = []
    for i, row in data.iterrows():
        if row[subject_feature_name] != curr_user:
            curr_user = row[subject_feature_name]
            times = -1
            curr_trial = 0
        if row[time_feature_name] < times:
            curr_trial += 1
        times = row[time_feature_name]
        trial_number.append(curr_trial)
    data[trial_feature_name] = trial_number
    return data

# convert an array of values into a dataset matrix
def create_dataset(dataset, output, target_att, subject, window_size=2,padding_const = -9999 ):
    for end_idx in range(len(dataset)):
        start_idx = end_idx - window_size + 1
        padding = 0-start_idx
        start_idx = max(0,start_idx)
        for i,c in enumerate(feature_names):
            a = []
            if padding>0:
                a = [padding_const] * padding
            a = np.array(np.concatenate((a,dataset[c].values[start_idx:end_idx+1])))
            output[i].append(a)
        subject.append(dataset.subject.values[end_idx])
        target_att.append(dataset[target_feature_name].values[end_idx])
    return output

def create_data_wrapper(input_data):
    patient = []
    y = []
    X = [[] for x in range(len(feature_names))]
    for sub_trial in input_data.subject_trial.unique():
        curr_subj_trial_data = input_data.loc[input_data.subject_trial == sub_trial, :]
        X = create_dataset(curr_subj_trial_data, X, y, patient, window_size=window_size)
    return [np.array(x) for x in X],np.array(y)


#Reading data & prepearing
data = pd.read_csv(os.path.join('data','result_better_features.csv'))
data = add_trial_num(data)

data['subject_trial'] = data[subject_feature_name].astype(str) +"_"+ data[trial_feature_name].astype(str)
groups = data[subject_feature_name]
subject_trial = data['subject_trial']

num_cols = len(feature_names)

logo = LeaveOneGroupOut()


#### non-NN methods ##############
# X = data[feature_names].copy()
# y = data[target_feature_name].copy()

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


window_size = 20
#as the window size increase the performance imrpove. this must be due to some leaking. need to further check

X = data[feature_names]
y = data[target_feature_name]
print('done prepearing data, number of features',len(feature_names))

def create_model(lstm_size = 16):
    inputs = [keras.Input((window_size, 1)) for x in feature_names]
    blstm = [layers.LSTM(lstm_size)(x) for x in inputs]
    output_layer = layers.concatenate(inputs=blstm, axis=1)
    # Final layer
    outputs = layers.Dense(1, activation="sigmoid")(output_layer)
    #Creating model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="Adam", loss="binary_crossentropy")
    return model


results = {'auc':[]}
print('number of subjects',len(set(data[subject_feature_name])))


for train_idx,test_idx in logo.split(data, data[target_feature_name], data[subject_feature_name]):
    scaler = StandardScaler() #scaling 0-1

    train = data.iloc[train_idx,:].copy()
    train[feature_names] = scaler.fit_transform(train[feature_names])
    X_train,y_train = create_data_wrapper(train)

    test = data.iloc[test_idx, :].copy()
    test[feature_names] = scaler.transform(test[feature_names])
    X_test, y_test = create_data_wrapper(test)


    print('creating model')
    model = create_model()
    print('fitting model')
    model.fit(X_train, y_train,verbose=1)
    preds = model.predict(X_test)
    curr_auc = roc_auc_score(y_test, preds)
    print(curr_auc)
    results['auc'].append(curr_auc)
print(pd.DataFrame(results).mean())


# print('num instances in NN data',len(output[0]))
print('num instances in original data',len(data))
print('average trial length',len(data)/len(data.subject_trial.unique()),'instances')

print('window size',window_size)