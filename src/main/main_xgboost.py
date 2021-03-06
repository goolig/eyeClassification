#Reading data & prepearing
from const import aug_test
import os
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from src.constants import *
from src.functions import *
os.chdir(".."); os.chdir("..")

results = {'auc': [], 'aupr': [], 'accuracy': [], 'precision': [], 'recall': [], 'eval_id': [], 'window_size': [],
           'test_type': [], 'file_name': [], 'eye': []}
max_number_of_evaluation_groups = 3
feature_names_original = feature_names_raw #feature_names
xgboost_parmas = {'n_estimators': 875, 'max_depth': 22, 'reg_alpha': 2, 'reg_lambda': 5, 'min_child_weight': 38, 'gamma': 0, 'learning_rate': 0.01540293696409555, 'subsample': 0.19692882420211885, 'colsample_bytree': 0.45999999999999996}

def neuralnet_to_tabular_representation(arr):
    return np.array([[curr_win_pos for feature in range(len(arr)) for curr_win_pos in arr[feature][pat]] for pat in range(len(arr[0]))])
random_state = np.random.RandomState(0)


if False:
    data = pd.read_csv(os.path.join('data', 'result_old_exp.csv'))
    data = add_trial_num(data)
    data, feature_names = average_l_r(data, feature_names_original)
    data['subject_trial'] = data[subject_feature_name].astype(str) + "_" + data[trial_feature_name].astype(str)
    groups = data[subject_feature_name]
    subject_trial = data['subject_trial']
    num_cols = len(feature_names)

    X_train, y_train = create_data_wrapper(data, 50, feature_names)
    X_train = neuralnet_to_tabular_representation(X_train)

    print('starting params')
    print(set(subject_trial))
    run_optuna_xgboost(X_train, y_train, [int(x.split('_')[0]) % 3 for x in subject_trial])


if True:
    for file_name in ['raw_reduced.csv']:  # 'result_better_features.csv', 'result_old_exp.csv'

        data = pd.read_csv(os.path.join('data', file_name))
        data = add_trial_num(data)
        data, feature_names = average_l_r(data, feature_names_original)
        data['subject_trial'] = data[subject_feature_name].astype(str) + "_" + data[trial_feature_name].astype(str)
        groups = data[subject_feature_name]
        subject_trial = data['subject_trial']
        num_cols = len(feature_names)

        logo = LeaveOneGroupOut()
        print('number of subjects',len(set(data[subject_feature_name])))
        eval_id=0

        data[feature_names] = data[feature_names].replace('-1',None)
        data[feature_names] = data[feature_names].replace(-1, None)
        for train_idx, test_idx in logo.split(data, data[target_feature_name],
                                              data[subject_feature_name] % max_number_of_evaluation_groups):

            #for window size
            for window_size in [40]:#[10,20,40,60,80,100,120,140,160,180]:
                train = data.iloc[train_idx, :].copy()
                #train[feature_names] = scaler.fit_transform(train[feature_names])
                X_train, y_train = create_data_wrapper(train, window_size, feature_names)

                test = data.iloc[test_idx, :].copy()
                #test[feature_names] = scaler.transform(test[feature_names])
                model = xgboost.XGBClassifier()

                X_train = neuralnet_to_tabular_representation(X_train)
                print('training')
                model.fit(X_train, y_train)

                X_test, y_test = create_data_wrapper(test, window_size, feature_names, test=True,padding_const=None)
                X_test = neuralnet_to_tabular_representation(X_test)

                X_test_pos, y_test_pos = create_data_wrapper(test[test['load']==1],window_size,feature_names, test=True,padding_const=None)
                X_test_pos = neuralnet_to_tabular_representation(X_test_pos)

                X_test_neg, y_test_neg = create_data_wrapper(test[test['load']==0], window_size,feature_names,test=True,padding_const=None)
                X_test_neg = neuralnet_to_tabular_representation(X_test_neg)

                X_test_aug, y_test_aug = create_data_wrapper(aug_test(test,random_state), window_size,feature_names,test=True,padding_const=None)
                X_test_aug = neuralnet_to_tabular_representation(X_test_aug)


                for test_name, X_test, y_test in [('all',X_test,y_test),('pos',X_test_pos,y_test_pos),('neg',X_test_neg,y_test_neg),('aug',X_test_aug,y_test_aug)]:

                    pred_scores = model.predict_proba(X_test)[:,-1]
                    t = 0.5
                    preds = [1 if x > t else 0 for x in pred_scores]
                    try:
                        curr_auc = roc_auc_score(y_test, pred_scores)
                    except:
                        curr_auc = -1
                    try:
                        curr_aupr = average_precision_score(y_test, pred_scores)
                    except:
                        curr_aupr = -1
                    curr_acc = accuracy_score(y_test, preds)
                    curr_prec = precision_score(y_test, preds)
                    curr_recall = recall_score(y_test, preds)
                    pd.DataFrame({'y': y_test, 'preds': [x for x in pred_scores]}). \
                        to_csv(os.path.join('results', file_name + '_' + '_' + str(window_size) + '_' + str(eval_id) + str(test_name) + '_preds.csv'), index=False)
                    # print(curr_auc)
                    results['auc'].append(curr_auc)
                    results['aupr'].append(curr_aupr)
                    results['accuracy'].append(curr_acc)
                    results['recall'].append(curr_recall)
                    results['precision'].append(curr_prec)
                    results['window_size'].append(window_size)
                    results['eval_id'].append(eval_id)
                    results['test_type'].append(test_name)
                    results['eye'].append('both')
                    results['file_name'].append(file_name)
                eval_id += 1
        res = pd.DataFrame(results)
        print(res[res.test_type=='aug'].mean())
        pd.DataFrame(results).to_csv(os.path.join('results','results.csv'))
