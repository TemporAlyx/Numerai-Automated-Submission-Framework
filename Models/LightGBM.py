import os, gc
import numpy as np
from Models.diagnostic_utils import *

import lightgbm

name = os.path.splitext(os.path.basename(__file__))[0] # use filename as model name
submit_on = ['your_model_name_here'] # list of numerai model names to submit predictions corrsponding to each column of P
ensembled = False # set to True if model is an ensemble of other models, ie, requires submissions
modeldata_directory = os.path.join(os.getcwd(), 'Models', 'Modeldata') # directory to store model data in

def predict(X, I, feature_sets, submissions=None, cleanup=True):
    lgbm_model = lightgbm.Booster(model_file=os.path.join(modeldata_directory, 'lgbm_model.txt'))
    features = np.load(os.path.join(modeldata_directory, 'lgbm_model_features.npy'))

    P = []
    for E in range(len(I)): # note: X will have to be rescaled from int 0-4 if not trained as such!
        P.append(lgbm_model.predict(X[features].iloc[I[E]].values.astype(np.float32)))
    P = np.concatenate(P, axis=-1)

    gc.collect()
    return P