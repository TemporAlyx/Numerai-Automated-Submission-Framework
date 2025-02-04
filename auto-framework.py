import pandas as pd
import numpy as np
import time, os, sys, gc, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from Models.framework_utils import *

# load numerapi
public_id, secret_key = get_numerapi_config()
napi, modelnameids = get_napi_and_models(public_id, secret_key)

# load data
ds_version = "v5.0"
dataset_loc = os.path.join(os.getcwd(), 'live_datasets', ds_version)
currentRound, isnewRound = get_update_live_data(napi, dataset_loc, ds_version)

time_slept = 0
while not isnewRound:
    time.sleep(1800)
    time_slept += 1800
    ncurrentRound, isnewRound = get_update_live_data(napi, dataset_loc, ds_version)
    if ncurrentRound != currentRound:
        currentRound = ncurrentRound
        break
    if time_slept >= (3600 * 23):
        sys.exit()

np.random.seed(42)
print("# Loading data... ",end='')

# live submission data L* | X = features, P = prediction, I = era indices
live, LI = processData(os.path.join(dataset_loc, 'live.parquet'), return_target_names=False)

with open(os.path.join(dataset_loc, "features.json"), "r") as f:
    feature_metadata = json.load(f)
feature_sets = feature_metadata['feature_sets']

BLP = pd.read_parquet(os.path.join(dataset_loc, 'live_benchmark_models.parquet'),engine="fastparquet")
ids = live.index.values

gc.collect()
print("done")


import Models

submissions = {}
upload_keys = {}

submissions['example_model'] = BLP['v5_lgbm_ct_blend'].values[:,None]

model_modules = [Models.__dict__[x] for x in Models.models]

n_submissions, model_modules = get_currentRound_submissions(currentRound, model_modules,
                                                            avoid_resubmissions=True)
submissions.update(n_submissions)

print([x.name for x in model_modules])

def build_and_submit_model(Model):
    try:
        print("Building and Processing model {}...".format(Model.name))
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

for Model in model_modules:
    if not Model.ensembled: build_and_submit_model(Model)
for Model in model_modules:
    if Model.ensembled: build_and_submit_model(Model)