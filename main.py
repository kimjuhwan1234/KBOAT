import os
import h5py
import optuna
import warnings
import pandas as pd
from Module.Esemble import *

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    directory = 'Database'
    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    data_dict={}
    for i in h5_files:
        with h5py.File(f'Database/{i}', 'r') as f:
            data_dict[i[:-3]] = f[i[:-3]][:]
        print(data_dict[i[:-3]].shape)


    for i in range(0, 4):
        # {DT=0, lightGBM=1, XGBoost=2, CatBoost=3}
        E = Esemble(i, data_dict['X_train_norm'], data_dict['X_val_norm'], data_dict['y_train'], data_dict['y_val'], 1000, 'original')

        study = optuna.create_study(direction='maximize')
        study.optimize(E.objective, n_trials=100)

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        E.save_best_model(study.best_trial.params)
