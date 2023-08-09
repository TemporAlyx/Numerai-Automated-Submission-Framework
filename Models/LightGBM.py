import numpy as np
import os, gc

from Models.framework_utils import *

import lightgbm


class CustomModel:
    name = 'LightGBM'
    submit_on = ['your_model_name_here']
    ensembled = False
    modeldata_directory = os.path.join(os.getcwd(), 'Models', 'Modeldata')
    
    isLoaded = False
    modeldata = {}
        
    @classmethod
    def load(cls):
        cls.modeldata['lgbm_model'] = lightgbm.Booster(model_file=os.path.join(cls.modeldata_directory, 'lgbm_model.txt'))
        cls.modeldata['features'] = np.load(os.path.join(cls.modeldata_directory, 'lgbm_model_features.npy'))
        cls.isLoaded = True
    
    @classmethod
    def predict(cls, X, I, features, submissions=None, clear=True):
        if not cls.isLoaded:
            cls.load()
                
        # get indices of features used in model
        feature_idxs = np.arange(len(features))[np.isin(features, cls.modeldata['features'])]
        X = X[:,feature_idxs]

        # predict
        P = []
        for E in range(len(I)):
            P.append(cls.modeldata['lgbm_model'].predict(X[I[E]]))
        P = np.concatenate(P, axis=-1)

        if clear: cls.clear()
        return P

    @classmethod
    def clear(cls):
        cls.modeldata = {}
        cls.isLoaded = False
        gc.collect()