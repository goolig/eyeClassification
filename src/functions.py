import numpy as np

from src.constants import subject_feature_name, time_feature_name, trial_feature_name, target_feature_name


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
def average_l_r(data,feature_names):
    #average of both eyes as asked by Hagit.
    new_cols = []
    for i in range(int(len(feature_names) / 2)):
        f1 = feature_names[i*2]
        f2 = feature_names[i*2+1]
        new_f_name = f1 + '_avg_eyes'
        data[new_f_name] = (data[f1]+data[f2])/2.0
        del data[f1]
        del data[f2]
        new_cols.append(new_f_name)
    return data,new_cols

def create_dataset(dataset, output, target_att, subject,feature_names, window_size=2,padding_const = -9999 ):
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


def create_data_wrapper(input_data,window_size,feature_names,test=False):
    patient = []
    y = []
    X = [[] for x in range(len(feature_names))]
    if not test:
        for sub_trial in input_data.subject_trial.unique():
            curr_subj_trial_data = input_data.loc[input_data.subject_trial == sub_trial, :]
            X = create_dataset(curr_subj_trial_data, X, y, patient,feature_names, window_size=window_size)
    else:
        curr_subj_trial_data = input_data
        X = create_dataset(curr_subj_trial_data, X, y, patient,feature_names, window_size=window_size)
    return [np.array(x) for x in X],np.array(y)
