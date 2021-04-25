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

from src.main.hyper_params_XGBOOST import run_optuna_xgboost

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

data = pd.read_csv(os.path.join('data','result_better_features.csv'))
data = add_trial_num(data)

data['subject_trial'] = data[subject_feature_name].astype(str) +"_"+ data[trial_feature_name].astype(str)
groups = data[subject_feature_name]
subject_trial = data['subject_trial']

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

scaler = StandardScaler()
data[feature_names] = scaler.fit_transform(data[feature_names]) #TODO: do only after train\test split
num_cols = len(feature_names)


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
            a = np.concatenate((a,dataset[c].values[start_idx:end_idx+1]))
            output[i].append(a)
        subject.append(dataset.subject.values[end_idx])
        target_att.append(dataset[target_feature_name].values[end_idx])
    return output


# for sub_trial in data['subject_trial']:
#     create_dataset(data.loc[data['subject_trial']==sub_trial])

window_size = 10 #about  4 per second. 375 per trial, 90 seconds per trial
#as the window size increase the performance imrpove. this must be due to some leaking. need to further check

patient = []
y_nn=[]
output = [[] for x in range(len(feature_names))]
for sub_trial in data.subject_trial.unique():
    curr_subj_trial_data = data.loc[data.subject_trial==sub_trial,:]
    output = create_dataset(curr_subj_trial_data, output, y_nn, patient, window_size=window_size)

X_nn = output
X = data[feature_names]
y = data[target_feature_name]
print('done prepearing data, number of features',len(feature_names))

def create_model(lstm_size = 32,mask_value=-9999,num_lstm=1,do=0.2):
    inputs = [keras.Input((window_size, 1)) for x in feature_names]
    mask = [layers.Masking(mask_value=mask_value)(x) for x in inputs]



    blstm = layers.Concatenate(axis=-1)(mask)



    for i in range(num_lstm):
        blstm = layers.Bidirectional(layers.LSTM(lstm_size,return_sequences = True))(blstm)
        blstm = layers.Dropout(do)(blstm)
    blstm = layers.Bidirectional(layers.LSTM(lstm_size))(blstm)

    output_layer = blstm


    #Old code
    # blstm = layers.LSTM(lstm_size)(merge_mask)
    # blstm = [layers.LSTM(lstm_size)(x) for x in mask]  # input_shape=(lookback, num_cols)
    # output_layer = layers.concatenate(inputs=blstm, axis=1)


    # Final layer
    outputs = layers.Dense(1, activation="sigmoid")(output_layer)

    #Creating model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="Adam", loss="binary_crossentropy")
    return model



results = {'auc':[]}
print('number of subjects',len(set(patient)))
for train_idx,test_idx in logo.split(output[0], y_nn, patient):
    train_data = [np.array(x)[train_idx] for x in X_nn]
    y_train = np.array(y_nn)[train_idx]
    test_data = [np.array(x)[test_idx] for x in X_nn]
    y_test = np.array(y_nn)[test_idx]


    print('creating model')
    model = create_model()
    print('fitting model')
    model.fit(train_data, y_train,verbose=1)
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
print('average trial length',len(data)/len(data.subject_trial.unique()),'instances')

print('window size',window_size)