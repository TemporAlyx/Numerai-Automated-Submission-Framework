import numpy as np
import os, gc

from Models.framework_utils import *

import lightgbm


class CustomModel:
    name = os.path.splitext(os.path.basename(__file__))[0] # use filename as model name
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
                
        X = X[cls.modeldata['features']]

        # predict
        P = []
        for E in range(len(I)): # note: X will have to be rescaled from 0-4 if not trained as such!
            P.append(cls.modeldata['lgbm_model'].predict(X.iloc[I[E]].values.astype(np.float32)))
        P = np.concatenate(P, axis=-1)

        if clear: cls.clear()
        return P

    @classmethod
    def clear(cls):
        cls.modeldata = {}
        cls.isLoaded = False
        gc.collect()