import pandas as pd
import numpy as np
import time, os, sys, gc, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from Models.framework_utils import *

# load numerapi
public_id, secret_key = get_numerapi_config()
napi, modelnameids = get_napi_and_models(public_id, secret_key)

# load data
ds_version = "v4.2"
dataset_loc = os.path.join(os.getcwd(), 'live_datasets', ds_version)
currentRound, isnewRound = get_update_live_data(napi, dataset_loc, ds_version)

if not isnewRound:
    sys.exit()

np.random.seed(42)
print("# Loading data... ",end='')

# live submission data L* | X = features, P = prediction, I = era indices
live, LI = processData(os.path.join(dataset_loc, 'live_int8.parquet'))

with open(os.path.join(dataset_loc, "features.json"), "r") as f:
    feature_metadata = json.load(f)
feature_sets = feature_metadata['feature_sets']

BLP = pd.read_parquet(os.path.join(dataset_loc, 'live_benchmark_models.parquet'),engine="fastparquet")
ids = BLP.index.values

gc.collect()
print("done")


import Models

submissions = {}
upload_keys = {}

submissions['example_model'] = BLP['V42_LGBM_CT_BLEND']

model_modules = Models.models

n_submissions, model_modules = get_currentRound_submissions(
    currentRound, modelnameids, model_modules
)
submissions.update(n_submissions)

EnsembledMods = []
Mods = [Models.__dict__[x] for x in model_modules]
print(model_modules)


def build_and_submit_model(Model):
    try:
        if Model.ensembled:
            LP = Model.predict(live, LI, feature_sets, submissions)
        else:
            LP = Model.predict(live, LI, feature_sets)
        LP = erarank01(LP, LI)

        n_submissions, n_response_keys = submitPredictions(
            LP, Model, modelnameids, ids, currentRound, napi
        )
        if len(n_submissions) > 0:
            submissions.update(n_submissions)
            upload_keys.update(n_response_keys)
    except Exception as e:
        print(e)
        print("Model {} failed".format(Model.name))
        print()

for Model in Mods:
    if Model.ensembled:
        EnsembledMods.append(Model) # wait until other models are done for ensembles
    else:
        build_and_submit_model(Model)
for Model in EnsembledMods:
    build_and_submit_model(Model)