import joblib
import numpy as np
from utils import custom_f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier


class Esemble:
    def __init__(self, method, X_train, X_val, y_train, y_val, num_rounds, sampling_name):
        self.method = method
        self.X_train = X_train
        self.X_test = X_val
        self.y_train = y_train
        self.y_test = y_val
        self.num_rounds = num_rounds
        self.name = sampling_name

    def save_dict_to_txt(self, file_path, dictionary):
        with open(file_path, 'w') as f:
            for key, value in dictionary.items():
                f.write(f'{key}: {value}\n')

    def DecisionTree(self, params):
        bst = DecisionTreeClassifier(**params)
        bst.fit(self.X_train, self.y_train)
        y_pred = bst.predict(self.X_test)
        accuracy = custom_f1_score(self.y_test, y_pred, average='weighted')
        print("DT F1-Score:", accuracy)
        return accuracy

    def lightGBM(self, params):
        model = LGBMClassifier(**params, n_estimators=self.num_rounds)
        bst = MultiOutputClassifier(model)
        bst.fit(self.X_train, self.y_train)
        y_pred = bst.predict(self.X_test)
        accuracy = custom_f1_score(self.y_test, y_pred, average='weighted')
        print("lightGBM F1-Score:", accuracy)
        return accuracy

    def XGBoost(self, params):
        model = XGBClassifier(**params, n_estimators=self.num_rounds)
        bst = MultiOutputClassifier(model)
        bst.fit(self.X_train, self.y_train)
        y_pred = bst.predict(self.X_test)
        accuracy = custom_f1_score(self.y_test, y_pred, average='weighted')
        print("XGBoost F1-Score:", accuracy)
        return accuracy

    def CatBoost(self, params):
        model = CatBoostClassifier(**params, n_estimators=self.num_rounds)
        bst = MultiOutputClassifier(model)
        bst.fit(self.X_train, self.y_train)
        y_pred = bst.predict(self.X_test)
        accuracy = custom_f1_score(self.y_test, y_pred, average='weighted')
        print("CatBoost F1-Score:", accuracy)
        return accuracy

    def objective(self, trial):
        if self.method == 0:
            params = {
                'criterion': 'entropy',
                'max_features': trial.suggest_float('max_features', 0.5, 0.99),
                'max_depth': trial.suggest_int('max_depth', 5, 200),
            }
            accuracy = self.DecisionTree(params)

        if self.method == 1:
            params = {
                'boosting': 'gbdt',
                'data_sample_strategy': 'goss',
                'num_leaves': 30,
                'force_col_wise': True,
                'device': 'cpu',
                'objective': 'multiclass',
                'num_class': 4,

                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                # 'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            }
            accuracy = self.lightGBM(params)

        if self.method == 2:
            params = {
                'device': 'cuda',
                'num_class': 4,
                'objective': 'multi:softprob',
                'booster': 'gbtree',

                'eta': trial.suggest_float('eta', 0.01, 0.1),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                # 'max_leaves': trial.suggest_int('max_leaves', 31, 256)
            }
            accuracy = self.XGBoost(params)

        if self.method == 3:
            params = {
                'task_type': 'CPU',
                'classes_count': 4,
                'loss_function': 'MultiClass',
                'grow_policy': 'Depthwise',

                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'depth': trial.suggest_int('depth', 5, 16),
                # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            }
            accuracy = self.CatBoost(params)

        return accuracy

    def save_best_model(self, best_params):
        if self.method == 0:
            best_params.update({
                'criterion': 'entropy',
            })

            bst = DecisionTreeClassifier(**best_params)
            bst.fit(self.X_train, self.y_train)
            joblib.dump(bst, f'File/dt_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/dt_{self.name}_params.txt', best_params)
            print(f'{custom_f1_score(self.y_test, bst.predict(self.X_test), average="weighted"):.4f}')
            print("Model saved!")

        if self.method == 1:
            best_params.update({
                'boosting': 'gbdt',
                'data_sample_strategy': 'goss',
                'num_leaves': 30,
                'force_col_wise': True,
                'device': 'cpu',
                'objective': 'multiclass',
                'num_class': 4,
            })

            bst = LGBMClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)])
            joblib.dump(bst, f'File/lgb_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/lgb_{self.name}_params.txt', best_params)
            print(f'{custom_f1_score(self.y_test, bst.predict(self.X_test), average="weighted"):.4f}')
            print("Model saved!")

        if self.method == 2:
            best_params.update({
                'device': 'cuda',
                'num_class': 4,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'booster': 'gbtree',
            })

            bst = XGBClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
            joblib.dump(bst, f'File/xgb_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/xgb_{self.name}_params.txt', best_params)
            print(f'{custom_f1_score(self.y_test, bst.predict(self.X_test), average="weighted"):.4f}')
            print("Model saved!")

        if self.method == 3:
            best_params.update({
                'task_type': 'CPU',
                'classes_count': 4,
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'grow_policy': 'Depthwise',
            })

            bst = CatBoostClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
            joblib.dump(bst, f'File/cat_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/cat_{self.name}_params.txt', best_params)
            print(f'{custom_f1_score(self.y_test, bst.predict(self.X_test), average="weighted"):.4f}')
            print("Model saved!")
