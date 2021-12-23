import pandas as pd
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
