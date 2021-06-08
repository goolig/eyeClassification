import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import xgboost
from sklearn.model_selection import LeaveOneGroupOut
import random
import pandas as pd
import os
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from src.constants import *
from src.functions import *
from src.main.hyper_params_XGBOOST import run_optuna_xgboost

seed= 0
random.seed(seed)
os.chdir(".."); os.chdir("..")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



#Reading data & prepearing
data = pd.read_csv(os.path.join('data','result_better_features.csv'))
data = add_trial_num(data)
data,feature_names = average_l_r(data,feature_names)
data['subject_trial'] = data[subject_feature_name].astype(str) +"_"+ data[trial_feature_name].astype(str)
groups = data[subject_feature_name]
subject_trial = data['subject_trial']
num_cols = len(feature_names)



logo = LeaveOneGroupOut()


X = data[feature_names]
y = data[target_feature_name]
print('done prepearing data, number of features',len(feature_names))

def create_model(lstm_size = 16, mask_value=-9999, num_lstm=3, do=0.5, model_interactions_lstm=False):
    #model_interactions_lstm=True : 0.828. bi directional
    #model_interactions_lstm=False: 0.879. uni directional
    #model_interactions_lstm=False: 0.895. bi directional
    inputs = [keras.Input((window_size, 1)) for x in feature_names]
    mask = [layers.Masking(mask_value=mask_value)(x) for x in inputs]
    if not model_interactions_lstm:
        to_concat = [layers.Bidirectional(layers.LSTM(lstm_size))(x) for x in mask]
    else:
        to_concat=mask
    concat = layers.Concatenate(axis=-1)(to_concat)
    blstm = layers.Dropout(do)(concat)
    if model_interactions_lstm:
        # Adding blstm
        for i in range(num_lstm):
            blstm = layers.Bidirectional(layers.LSTM(lstm_size,return_sequences = True))(blstm)
            blstm = layers.Dropout(do)(blstm)
        blstm = layers.Bidirectional(layers.LSTM(lstm_size))(blstm)
        blstm = layers.Dropout(do)(blstm)
    # Final layer
    outputs = layers.Dense(1, activation="sigmoid")(blstm)
    #Creating model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,) ,loss="binary_crossentropy",metrics=['AUC'])
    return model

def create_model_pairs(lstm_size = 16, mask_value=-9999, do=0.5,add_att=True):
    #Bidirectional - 0.905
    #Biderctional + attention layers.Attention()([blstm,blstm]) 0.914
    #with window = 50 0.922
    #window = 75 0.939
 #   window_size = 100 0.94
    # windows 125 0.948
    #window size 150 0.964
    #window size 175 0.964
#window size = 200 0.938
    #Window 250 0.937 very long runtime
    inputs = [keras.Input((window_size, 1)) for x in feature_names]
    mask = [layers.Masking(mask_value=mask_value)(x) for x in inputs]
    assert len(inputs)%2 ==0,'input is not given in pairs'

    lstms = []
    print('num lstm',range(int(len(inputs)/2)))

    if False:
        for i in range(int(len(inputs) / 2)):
            con = layers.Concatenate(axis=-1)(mask[i * 2:i * 2 + 2])
            lstms.append(layers.Bidirectional(layers.LSTM(lstm_size,return_sequences=False))(con))
    else:
        for i in range(len(inputs)):
            lstms.append(layers.Bidirectional(layers.LSTM(lstm_size, return_sequences=False))(mask[i]))
    concat = layers.Concatenate(axis=-1)(lstms)
    concat = layers.Flatten()(concat)
    blstm = layers.Dropout(do)(concat)

    if add_att:
        blstm = layers.Attention()([blstm,blstm])
        #blstm = layers.Concatenate(axis=-1)([a,b])
        blstm = layers.Dropout(do)(blstm)

    # Final layer
    outputs = layers.Dense(1, activation="sigmoid")(blstm)
    #Creating model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,) ,loss="binary_crossentropy",metrics=['AUC'])
    return model


results = {'auc':[],'aupr':[],'accuracy':[],'precision':[],'recall':[],'subject':[],'window_size':[],'test_type':[]}
print('number of subjects',len(set(data[subject_feature_name])))

for window_size in [25,50]: #,100,150
    curr_subject_id=0
    for train_idx,test_idx in logo.split(data, data[target_feature_name], data[subject_feature_name]):
        scaler = StandardScaler() #scaling 0-1

        train = data.iloc[train_idx,:].copy()
        train[feature_names] = scaler.fit_transform(train[feature_names])
        X_train,y_train = create_data_wrapper(train,window_size,feature_names)

        test = data.iloc[test_idx, :].copy()
        test[feature_names] = scaler.transform(test[feature_names])



        print('creating model')
        model = create_model_pairs()
        print('fitting model')

        mon = 'val_auc'
        verb = 0

        assert 'auc' in mon,'must change mode in following two lines'
        reduce_lr = ReduceLROnPlateau(monitor=mon, factor=0.1, patience=0,
                                      min_lr=0.0001,min_delta=0.01,cooldown=1,mode='max')
        early_stop = EarlyStopping(monitor=mon, min_delta=0.0001, patience=3,
                                   restore_best_weights=True,mode='max')

        model.fit(X_train, y_train, verbose=1, epochs=50, callbacks=[early_stop,reduce_lr], validation_split=0.15) #,batch_size=512

        X_test, y_test = create_data_wrapper(test,window_size, feature_names,test=True)
        X_test_pos, y_test_pos = create_data_wrapper(test[test['load']==1],window_size,feature_names, test=True)
        X_test_neg, y_test_neg = create_data_wrapper(test[test['load']==0], window_size,feature_names,test=True)

        for test_name, X_test, y_test in [('all',X_test,y_test),('pos',X_test_pos,y_test_pos),('neg',X_test_neg,y_test_neg)]:
            pred_scores = model.predict(X_test)
            t = 0.5
            preds = [1 if x[0]>t else 0 for x in pred_scores]
            try:
                curr_auc = roc_auc_score(y_test, pred_scores)
            except:
                curr_auc=-1
            try:
                curr_aupr = average_precision_score(y_test, pred_scores)
            except:
                curr_aupr=-1
            curr_acc = accuracy_score(y_test,preds)
            curr_prec = precision_score(y_test,preds)
            curr_recall = recall_score(y_test,preds)
            pd.DataFrame({'y':y_test,'preds':[x[0] for x in pred_scores]}).to_csv(test_name+'_'+str(window_size) + '_' + str(curr_subject_id) + '_preds.csv', index=False)
            print(curr_auc)
            results['auc'].append(curr_auc)
            results['aupr'].append(curr_aupr)
            results['accuracy'].append(curr_acc)
            results['recall'].append(curr_recall)
            results['precision'].append(curr_prec)
            results['window_size'].append(window_size)
            results['subject'].append(curr_subject_id)
            results['test_type'].append(test_name)

        curr_subject_id+=1
    print(pd.DataFrame(results).mean())
    pd.DataFrame(results).to_csv('results.csv')


# print('num instances in NN data',len(output[0]))
print('num instances in original data',len(data))
print('average trial length',len(data)/len(data.subject_trial.unique()),'instances')

print('window size',window_size)