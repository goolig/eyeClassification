# feature_names = ['median_pupil_size_1', 'median_pupil_size_2','poly_pupil_size_1', 'poly_pupil_size_2', 'max_sac_1', 'max_sac_2',
#        'count_sac_1', 'count_sac_2', 'diff_pos_x_1', 'diff_pos_x_2', 'diff_pos_y_1', 'diff_pos_y_2']
feature_names = [
                'psr', 'psl',
                'fix_r_dur', 'fix_l_dur',
                'fix_r_disx', 'fix_l_disx',
                'fix_r_disy', 'fix_l_disy',
                'fix_r_c', 'fix_l_c',
                'sac_r_dur', 'sac_l_dur',
                'sac_r_ampl', 'sac_l_ampl',
                'sac_r_pv','sac_l_pv',
                'sac_r_x', 'sac_l_x',
                'sac_r_y', 'sac_l_y',
                'sac_r_c', 'sac_l_c',
                'psr_poly_1','psl_poly_1',
                'psr_poly_2', 'psl_poly_2',
                'psr_poly_3', 'psl_poly_3'
]

feature_names_raw = [
    'xpr', 'xpl',
    'psr', 'psl',
    'ypr', 'ypl'
]

target_feature_name = 'load'
time_feature_name = 'time'#'times'
subject_feature_name = 'subject'
trial_feature_name = 'trial'

