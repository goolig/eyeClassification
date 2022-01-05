import pandas as pd
import numpy as np
def aug_test(test_data,random_state):
    ans = None
    for subject_trial in set(test_data.subject_trial):
        portion = random_state.random()
        curr_data = test_data[test_data['subject_trial']==subject_trial]
        curr_data = curr_data.sample(frac=portion,random_state=random_state).sort_index()
        if ans is None:
            ans = curr_data
        else:
            ans = pd.concat([ans,curr_data])
    print('test data size',len(test_data),'after augmentation',len(ans))
    return ans


def neuralnet_to_tabular_representation(arr):
    return np.array([[curr_win_pos for feature in range(len(arr)) for curr_win_pos in arr[feature][pat]] for pat in range(len(arr[0]))])
