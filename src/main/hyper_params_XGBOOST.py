
import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from xgboost import XGBClassifier

def objective(trial: Trial, X, y, groups) -> float:
    # joblib.dump(study, 'study.pkl')


    param = {
        "n_estimators": trial.suggest_int('n_estimators', 0, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 25),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 50),
        #'scale_pos_weight ': trial.suggest_int('scale_pos_weight', 1, 100),
        'gamma': trial.suggest_int('gamma', 0, 5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
        'subsample': trial.suggest_loguniform('subsample', 0.001, 1),

        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),

        'nthread': -1
    }
    model = XGBClassifier(**param)
    # model.fit(train_X, train_y)
    logo = LeaveOneGroupOut()

    scores = cross_validate(model,X, y,
                            cv=logo.split(X, y, groups), scoring='roc_auc')
    return sum(scores['test_score'])/len(scores['test_score'])

def run_optuna_xgboost(X, Y, groups,n_trials=10):
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, X, Y,groups), n_trials=n_trials)
    return study.best_trial.params
