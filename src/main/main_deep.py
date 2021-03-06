import tensorflow as tf
from tensorflow.python.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, BatchNormalization, \
    Activation, GlobalAveragePooling1D

from const import aug_test, neuralnet_to_tabular_representation
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

from src.constants import feature_names, feature_names_raw
from src.functions import *
from src.main.hyper_params_XGBOOST import run_optuna_xgboost

random_state = np.random.RandomState(0)
seed= 0
random.seed(seed)
os.chdir(".."); os.chdir("..")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



# X = data[feature_names]
# y = data[target_feature_name]
# print('done prepearing data, number of features',len(feature_names))

def create_model_conv(conv_num = 64,kernel_size=5, mask_value=-9999, do=0.5):

    inputs = [keras.Input((window_size, 1)) for x in feature_names]
    mask = [layers.Masking(mask_value=mask_value)(x) for x in inputs]
    to_concat=mask
    concat = layers.Concatenate(axis=-1)(to_concat)

    layer = Conv1D(filters=conv_num, kernel_size=kernel_size,input_shape=(len(inputs),len(feature_names)))(concat)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.ReLU()(layer)

    layer = Conv1D(filters=int(conv_num/2), kernel_size=kernel_size)(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.ReLU()(layer)
    #layer = MaxPooling1D(3)(layer)
    layer = Conv1D(filters=(conv_num/4), kernel_size=kernel_size)(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.ReLU()(layer)
    layer = Conv1D(filters=(conv_num/8), kernel_size=kernel_size)(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.ReLU()(layer)

    #layer = Conv1D(filters=(conv_num/8), kernel_size=kernel_size, activation='relu')(layer)
    #layer = GlobalAveragePooling1D()(layer)
    layer = Dropout(do)(layer)
    layer = Flatten()(layer)
    #layer = Dense(10,activation='relu')(layer)
    outputs = layers.Dense(1, activation="sigmoid")(layer)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,) ,loss="binary_crossentropy",metrics=['AUC'])
    return model

def create_model(lstm_size = 32, mask_value=-9999, num_lstm=1, do=0.5, model_interactions_lstm=False,add_att=True):
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
        for i in range(num_lstm-1):
            blstm = layers.Bidirectional(layers.LSTM(lstm_size,return_sequences = True))(blstm)
            blstm = layers.Dropout(do)(blstm)
        blstm = layers.Bidirectional(layers.LSTM(lstm_size))(blstm)
        blstm = layers.Dropout(do)(blstm)
    if add_att:
        blstm = layers.Attention(causal=True)([blstm,blstm])
        #blstm = layers.Concatenate(axis=-1)([a,b])
        blstm = layers.Dropout(do)(blstm)

    # Final layer
    outputs = layers.Dense(1, activation="sigmoid")(blstm)
    #Creating model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,) ,loss="binary_crossentropy",metrics=['AUC'])
    return model

def create_model_pairs(lstm_size = 16, mask_value=-9999, do=0.5,add_att=True,eye='both',num_lstm=1):
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
    #print('num lstm',range(int(len(inputs)/2)))

    if eye=='both':
        for i in range(int(len(inputs) / 2)):
            con = layers.Concatenate(axis=-1)(mask[i * 2:i * 2 + 2])
            for j in range(num_lstm):
                if j!=num_lstm-1:
                    con = layers.Bidirectional(layers.LSTM(lstm_size, return_sequences=True))(con)
                else:
                    con = layers.Bidirectional(layers.LSTM(lstm_size,return_sequences=False))(con)
            lstms.append(con)
    else:
        for i in range(len(inputs)):
            curr_lstm = mask[i]
            for j in range(num_lstm):
                if j!=num_lstm-1:
                    curr_lstm = layers.Bidirectional(layers.LSTM(lstm_size, return_sequences=True))(curr_lstm)
                else:
                    curr_lstm=layers.Bidirectional(layers.LSTM(lstm_size, return_sequences=False))(curr_lstm)
            lstms.append(curr_lstm)
    concat = layers.Concatenate(axis=-1)(lstms)
    concat = layers.Flatten()(concat)
    blstm = layers.Dropout(do)(concat)

    if add_att:
        blstm = layers.Attention(causal=True)([blstm,blstm])
        #blstm = layers.Concatenate(axis=-1)([a,b])
        blstm = layers.Dropout(do)(blstm)

    # Final layer
    outputs = layers.Dense(1, activation="sigmoid")(blstm)
    #Creating model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,) ,loss="binary_crossentropy",metrics=['AUC'])
    return model

#Reading data & prepearing
feature_names_original = feature_names_raw #feature_names
results = {'auc': [], 'aupr': [], 'accuracy': [], 'precision': [], 'recall': [], 'eval_id': [], 'window_size': [],
           'test_type': [], 'file_name':[], 'eye':[],'algo':[]}

max_number_of_evaluation_groups=3

missing_const = -9999

for file_name in ['result_raw2.csv']: #raw_reduced.csv 'result_better_features.csv', 'result_old_exp.csv' 'result_old_exp2.csv','result_better_features.csv',
    #for eye in ['avg','r','l','both']: #
    for eye in ['avg']:  #for attention check
        data = pd.read_csv(os.path.join('data',file_name))
        data = add_trial_num(data)
        data,feature_names = average_l_r(data,feature_names_original)
        data['subject_trial'] = data[subject_feature_name].astype(str) +"_"+ data[trial_feature_name].astype(str)
        groups = data[subject_feature_name]
        subject_trial = data['subject_trial']
        num_cols = len(feature_names)

        logo = LeaveOneGroupOut()

        print('number of subjects',len(set(data[subject_feature_name])))
        data[feature_names] = data[feature_names].replace('-1',missing_const)
        data[feature_names] = data[feature_names].replace(-1, missing_const)
        data[feature_names] = data[feature_names].fillna(missing_const)

        #for window_size in [10,20,40,60,80,100,120,140,160,180]: #,200
        for window_size in [40]: #for attention check
            eval_id=0
            for train_idx,test_idx in logo.split(data, data[target_feature_name], data[subject_feature_name]%max_number_of_evaluation_groups):
                scaler = StandardScaler() #scaling 0-1

                train = data.iloc[train_idx,:].copy()
                train[feature_names] = scaler.fit_transform(train[feature_names])
                X_train,y_train = create_data_wrapper(train,window_size,feature_names,padding_const=missing_const)
                X_train_xgb = neuralnet_to_tabular_representation(X_train)

                test = data.iloc[test_idx, :].copy()
                test[feature_names] = scaler.transform(test[feature_names])


                xgb = xgboost.XGBClassifier()
                xgb.fit(X_train_xgb, y_train)

                #print('creating model')
                model = create_model_conv(mask_value=missing_const)
                #model = create_model(model_interactions_lstm=False,lstm_size=32,num_lstm=2,mask_value=missing_const)
                #print('fitting model')

                mon = 'val_auc'
                verb = 0

                assert 'auc' in mon,'must change mode in following two lines'
                reduce_lr = ReduceLROnPlateau(monitor=mon, factor=0.1, patience=0,
                                              min_lr=0.0001,min_delta=0.01,cooldown=1,mode='max')
                early_stop = EarlyStopping(monitor=mon, min_delta=0.0001, patience=3,
                                           restore_best_weights=True,mode='max')
                print('fitting')
                model.fit(X_train, y_train, verbose=0, epochs=50, callbacks=[early_stop,reduce_lr], validation_split=0.15) #,batch_size=512
                print('done fitting')
                X_test, y_test = create_data_wrapper(test,window_size, feature_names,test=True,padding_const=missing_const)
                X_test_pos, y_test_pos = create_data_wrapper(test[test['load']==1],window_size,feature_names, test=True,padding_const=missing_const)
                X_test_neg, y_test_neg = create_data_wrapper(test[test['load']==0], window_size,feature_names,test=True,padding_const=missing_const)
                X_test_aug, y_test_aug = create_data_wrapper(aug_test(test,random_state), window_size,feature_names,test=True,padding_const=missing_const)




                for test_name, X_test, y_test in [('all',X_test,y_test),('pos',X_test_pos,y_test_pos),('neg',X_test_neg,y_test_neg),('aug',X_test_aug,y_test_aug)]:
                    X_test_xgb = neuralnet_to_tabular_representation(X_test)
                    pred_scores = model.predict(X_test).flatten()
                    pred_scores_xgb = xgb.predict_proba(X_test_xgb)[:,-1]
                    t = 0.5
                    preds = [1 if x>t else 0 for x in pred_scores]
                    pred_xgb = [1 if x > t else 0 for x in pred_scores_xgb]
                    for algo_name, preds, pred_scores in [('NN',preds,pred_scores),('XGBoost',pred_xgb,pred_scores_xgb)]:
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
                        pd.DataFrame({'y':y_test,'preds':[x for x in pred_scores]}).\
                            to_csv(os.path.join('results', file_name +'_' +algo_name +'_' + eye + '_' + test_name +'_' + str(window_size) + '_' + str(eval_id) + '_preds.csv'), index=False)
                        #print(curr_auc)
                        results['auc'].append(curr_auc)
                        results['aupr'].append(curr_aupr)
                        results['accuracy'].append(curr_acc)
                        results['recall'].append(curr_recall)
                        results['precision'].append(curr_prec)
                        results['window_size'].append(window_size)
                        results['eval_id'].append(eval_id)
                        results['test_type'].append(test_name)
                        results['eye'].append(eye)
                        results['file_name'].append(file_name)
                        results['algo'].append(algo_name)
                eval_id+=1
                print('done eval round', eval_id)
            print('done window',window_size)
        print('done eye',eye)
    res = pd.DataFrame(results)
    res = res[res.test_type=='aug']
    print('NN',res[res.algo=='NN'].mean())
    print('XGBoost',res[res.algo == 'XGBoost'].mean())
    pd.DataFrame(results).to_csv(os.path.join('results','results.csv'))


    # print('num instances in NN data',len(output[0]))
    print('num instances in original data',len(data))
    print('average trial length',len(data)/len(data.subject_trial.unique()),'instances')

    print('done file',file_name)