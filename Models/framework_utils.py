import pandas as pd
import numpy as np
import json, time, os, sys, gc

from scipy import stats
from IPython.display import clear_output

import numerapi

def percFin(iterator,listlen,rounded=3):
    clear_output(wait=True)
    print("Processing.. ",round(((iterator+1)/listlen) * 100,rounded),'%',flush=True)

def rank01(arr):
    arr = stats.rankdata(arr, method="average")
    arr = arr - 0.5
    return arr / len(arr)

def colrank01(arrs):
    return np.apply_along_axis(rank01,0,arrs)

def erarank01(arrs, I):
    arrsc = arrs.copy()
    if len(arrs.shape) == 2:
        for E in range(len(I)):
            arrsc[I[E]] = np.apply_along_axis(rank01,0,arrs[I[E]])
    else:
        for E in range(len(I)):
            arrsc[I[E]] = rank01(arrs[I[E]])
    return arrsc


def get_numerapi_config():
    # load id and key from json if exists, else create new file
    if not os.path.exists('config.json'):
        # create template json file
        with open('config.json', 'w') as f:
            json.dump({'id':'', 'key':''}, f)
        print('Please enter your Numerai ID and Key in created config.json and restart')
        time.sleep(5)
        sys.exit()
    else:
        with open('config.json', 'r') as f:
            config = json.load(f)
        if config['id'] == '' or config['key'] == '':
            print('Please enter your Numerai ID and Key in config.json and restart')
            time.sleep(5)
            sys.exit()
        else:
            public_id = config['id']
            secret_key = config['key']
            print('numerapi ID and Key loaded from config.json')
            return public_id, secret_key
        
def get_napi_and_models(public_id, secret_key):
    napi = None
    modelnameids = None
    napi_success = False; loops = 0
    while not napi_success:
        try:
            napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
            modelnameids = napi.get_models()
            napi_success = True
        except:
            loops += 1
            if loops > 10:
                print("NumerAPI connection failed, exiting...")
                sys.exit(); # maybe add some email notification here?
            print("NumerAPI connection failed, retrying...")
            time.sleep(5)
    return napi, modelnameids


train_files = ['train_int8.parquet', 'validation_int8.parquet',
         'validation_example_preds.parquet',
         'features.json', 'meta_model.parquet'] 

live_files = ['live_int8.parquet', 'live_example_preds.parquet',
               'features.json']

def chk_rm_ds(ds_file, dataset_loc):
    fp = os.path.join(dataset_loc, ds_file)
    if os.path.exists(fp): os.remove(fp)

def get_update_data(napi, dataset_loc, ds_version, files):
    if not os.path.exists(dataset_loc):
        os.makedirs(dataset_loc)

        with open(os.path.join(dataset_loc,'lastRoundAcq.txt'), 'w') as f:
            f.write('0')
        print('Created dataset directory at', dataset_loc)
    
    currentRound = napi.get_current_round()
    with open(os.path.join(dataset_loc, 'lastRoundAcq.txt'), 'r') as f:
        lastRound = int(f.read())
    newRound = lastRound != currentRound
    if newRound:
        print('Dataset not up to date, retrieving dataset... ',end='')
        for ds_file in files: 
            if currentRound - lastRound > 5 or 'train' not in ds_file: # avoid redownloading train too often
                chk_rm_ds(ds_file, dataset_loc)
        print('done')
        print('downloading new files... ',end='')
        for ds_file in files: 
            if currentRound - lastRound > 5 or 'train' not in ds_file:
                napi_success = False; loops = 0
                while not napi_success:
                    try:
                        napi.download_dataset(ds_version+'/'+ds_file, 
                                            os.path.join(dataset_loc,ds_file))
                        napi_success = True
                    except:
                        loops += 1
                        if loops > 5:
                            print('Numerapi data download failed')
                            break
                        print('Numerapi data download error, retrying...')
                        time.sleep(5)
        print('done')
        clear_output()
        with open(dataset_loc + '/lastRoundAcq.txt', 'w') as f:
            f.write(str(currentRound))
    print("Datasets are up to date.\nCurrent Round:", currentRound)
    return currentRound, newRound

def get_update_training_data(napi, dataset_loc, ds_version):
    return get_update_data(napi, dataset_loc, ds_version, train_files)

def get_update_live_data(napi, dataset_loc, ds_version):
    return get_update_data(napi, dataset_loc, ds_version, live_files)


def processData(df_loc, return_targets=False):
    df = pd.read_parquet(df_loc, engine="fastparquet")
    E = df['era'].values; uE = pd.unique(E)
    I = [(np.arange(len(E), dtype=np.int64)[x==E]) for x in uE]
    # features = [f for f in list(df.iloc[0].index) if "feature" in f]
    targets = [f for f in list(df.iloc[0].index) if "target" in f]
    # df = df[features+targets]; df = df.to_numpy(dtype=np.float16, na_value=0.5)
    # X = df[:,:-len(targets)]; Y = df[:,-len(targets):]; del df; gc.collect()
    if return_targets: return df, I, targets
    return df, I


def submitPredictions(LP, Model, modelids, liveids, currentRound, napi, verbose=2):
    name = Model.name
    sub_names = Model.submit_on
    if type(sub_names) != list: 
        sub_names = [sub_names]; LP = LP.reshape(-1, 1)
    elif len(LP.shape) == 1:
        LP = LP.reshape(-1, 1)
    print('building predictions for', name, sub_names)

    submissions = {}
    response_keys = {}

    for i in range(len(sub_names)):
        upload = sub_names[i]
        results_df = pd.DataFrame(data={'prediction' : LP[:,i]})
        joined = pd.DataFrame(liveids, columns=['id']).join(results_df)
        if verbose > 1: print(joined.head(3))

        subName = "submission"+name+"_"+upload[:5]+"_"+str(currentRound)+".csv"
        if verbose > 0: print("# Writing predictions to "+subName+"... ",end="")
        joined.to_csv("Submissions/"+subName, index=False)
        upload_key = None
        if not len(upload) > 0:
            if verbose > 1: print("No upload for these predictions. (may be base model)")
        else:
            napi_success = False; loops = 0
            while not napi_success:
                try:
                    upload_key = napi.upload_predictions("Submissions/"+subName, 
                                                    model_id=modelids[upload])
                    napi_success = True
                except:
                    loops += 1
                    if loops >= 5:
                        print("Failed to upload predictions for "+upload)
                        # remove failed submission file
                        os.remove("Submissions/"+subName)
                    print("Upload for "+upload+" failed, retrying... ",end="")
                    time.sleep(4)
            if verbose > 0: print(upload_key)
        submissions[upload] = joined
        response_keys[upload] = upload_key
        if verbose > 1: print("done")
    return submissions, response_keys

def get_currentRound_submissions(currentRound, modelnameids, modelmodules):
    if not os.path.exists('Submissions'): 
        os.makedirs('Submissions')
        subs = []
    else:
        subs = os.listdir('Submissions')
        sub_files = [x for x in subs if str(currentRound) in x and 'submission' in x]
        subs = [x.split('_')[:-1] for x in sub_files]
        subs = [x[-2:] if len(x) > 2 else x for x in subs]
        for i in range(len(subs)):
            subs[i][0] = subs[i][0][10:]
            subs[i].append(sub_files[i])
        subs = np.array(subs)
    
    submissions = {}
    for i in range(len(subs)):
        d = pd.read_csv('Submissions\\'+subs[i][-1], header=0).values[:,1].astype(float)
        if len(subs[i][1]) > 0:
            full_name = [x for x in list(modelnameids.keys()) if subs[i][1] == x[:len(subs[i][1])]][0]
        else:
            full_name = subs[i][0]
        submissions[full_name] = d
        
        if subs[i][0] in modelmodules:
            modelmodules.remove(subs[i][0])

    return submissions, modelmodules