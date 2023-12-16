import pandas as pd
import numpy as np

from scipy import stats
import matplotlib.pyplot as plt

from Models.framework_utils import *


def simple_corr(preds, targs): # ~pearson correlation
    preds_dm = preds - np.mean(preds)
    targs_dm = targs - np.mean(targs)

    sumdms = np.sum(preds_dm * targs_dm)
    sumsqrs = np.sqrt(np.sum(preds_dm ** 2) * np.sum(targs_dm ** 2))

    return sumdms / sumsqrs

def rank_corr(preds_ur, targs_ur): # ~spearman correlation
    preds = rank01(preds_ur)
    targs = rank01(targs_ur)
    
    preds_dm = preds - np.mean(preds)
    targs_dm = targs - np.mean(targs)

    sumdms = np.sum(preds_dm * targs_dm)
    sumsqrs = np.sqrt(np.sum(preds_dm ** 2) * np.sum(targs_dm ** 2))

    return sumdms / sumsqrs


def numerai_corr(preds, target):
    ranked_preds = (stats.rankdata(preds, method="average") - 0.5) / len(preds)
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)

    centered_target = target - target.mean()

    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5

    return simple_corr(preds_p15, target_p15)


def colrank_corr(preds_ur, targs_ur): # ~spearman correlation, optimized for multiple columns
    preds = rank01(preds_ur)
    if len(targs_ur.shape) < 2: targs_ur = targs_ur.reshape(-1,1)
    targs = colrank01(targs_ur)
    return col_corr(preds, targs)

def col_corr(preds, targs): # ~pearson correlation, optimized for multiple columns
    ddiffs = (preds - preds.mean()).reshape(-1, 1)
    idiff = targs - targs.mean(axis=0)
    
    idiff2 = np.sum(np.square(idiff), axis=0)
    prod = np.sum(ddiffs * idiff, axis=0)
    ddiffs2 = np.sum(np.square(ddiffs), axis=0)
    
    return prod / np.sqrt(ddiffs2 * idiff2)

def neutralize(target, by, proportion=1.0):
    by = by - np.mean(by)
    target = target - np.mean(target)
    scores = target - (proportion * by.dot(np.linalg.pinv(by).dot(target)))
    return scores / scores.std()

# def tf_neutralize(target, data):
#     data = data - tf.reduce_mean(data, axis=-2, keepdims=True)
#     target = target - tf.reduce_mean(target, axis=-2, keepdims=True)
#     invexp = tf.cast(tf.linalg.pinv(tf.cast(data, tf.float32)), target.dtype)
#     diff = tf.matmul(data,tf.matmul(invexp,target))
#     scores = target - diff
#     return scores / tf.math.reduce_std(scores)

def era_neutralize(target, by, I, proportion=1.0):
    out = np.empty(len(target))
    for E in I:
        targeti = target[E]; byi = by[E]
        exposures = np.hstack((byi, np.repeat(np.mean(targeti)), len(byi).reshape(-1, 1)))
        scores = target - (proportion * exposures.dot(np.linalg.pinv(exposures).dot(targeti)))
        out[E] = scores / scores.std()
    return out

def sortino_ratio(x, target=.02):
    xt = x[~np.isnan(x)] - target
    return np.mean(xt)/(((np.sum(np.minimum(0.000001, xt)**2)/(len(xt)-1))**.5))

def drawdown(x):
    max_val = 1.0; running = 1.0; dd = 0
    for i in x:
        running *= 1+i
        if running > max_val:
            max_val = running
        else:
            if (running / max_val)-1 < dd:
                dd = (running / max_val)-1
    return dd

def ts_stats(ts,name=''):
    diagnostics = {}
    diagnostics[name+' mean'] = np.nanmean(ts); diagnostics[name+' sdev'] = np.nanstd(ts)
    diagnostics[name+' sharpe'] = diagnostics[name+' mean'] / diagnostics[name+' sdev']; 
    diagnostics[name+' sortino (-0.02)'] = sortino_ratio(ts, 0.02)
    diagnostics[name+' drawdown'] = drawdown(ts)
    return diagnostics
    
#  P - predictions, Y - true target, X - base dimensions, I - era indexing
def run_diagnostics(P,Y,X,I, featexp=False, fnc=True, print_output=True, graph_corrs=True, compare=None):
    diagnostics = {}; raw_corrs = {}
    es = np.empty(len(I))
    for i in range(len(es)): es[i] = numerai_corr(P[I[i]],Y[I[i]])
    raw_corrs['corrv2'] = es
    diagnostics.update(ts_stats(es,'corrv2'))
    nes = np.empty(len(I)); NP = np.empty(len(P))
    if fnc:
        for E in I: NP[E] = neutralize(P[E],X.iloc[E].values.astype(np.float32))
        for i in range(len(nes)): nes[i] = rank_corr(NP[I[i]],Y[I[i]])
        raw_corrs['fnc'] = nes
        diagnostics.update(ts_stats(nes,'fnc'))
    if featexp: 
        feature_exposures = np.array([col_corr(P[E],X.iloc[E].values.astype(np.float32)) for E in I])
        diagnostics['mean abs feature exposure'] = np.nanmean(np.abs(feature_exposures))
        diagnostics['max abs feature exposure'] = np.nanmax(np.abs(np.nanmean(feature_exposures, axis=0)))
        diagnostics['sdev feature exposure'] = np.nanmean(np.nanstd(feature_exposures, axis=0))
    if compare is not None:
        if type(compare) != dict:
            compare = {'compare':compare}
        for k,v in compare.items():
            ces = np.empty(len(I))
            for i in range(len(ces)): ces[i] = rank_corr(P[I[i]],v[I[i]])
            # raw_corrs[k+' corr (preds)'] = ces
            diagnostics[k+' corr (preds) mean'] = np.nanmean(ces)
            diagnostics[k+' corr (preds) sdev'] = np.nanstd(ces)
            ces = np.empty(len(I))
            for i in range(len(ces)): ces[i] = numerai_corr(v[I[i]],Y[I[i]])
            raw_corrs[k+' corr (targs)'] = ces
            diagnostics[k+' corr (targs) mean'] = np.nanmean(ces)
            diagnostics[k+' corr (targs) sdev'] = np.nanstd(ces)
    if print_output:
        for k,v in diagnostics.items(): print(k,':',round(v,6))
    if graph_corrs:
        plt.figure(figsize=(14,6))
        for k,v in raw_corrs.items(): 
            if 'targs' not in k: plt.plot(v, label=k)
        # add a zero line
        plt.plot([0,len(I)],[0,0], color='white', linestyle='--')
        plt.legend()
        plt.show()

        # add a second plot with cumulative 1xcorr returns
        plt.figure(figsize=(14,6))
        for k,v in raw_corrs.items(): plt.plot(np.cumsum(v), label=k)
        # add a zero line
        plt.plot([0,len(I)],[0,0], color='white', linestyle='--')
        plt.legend()
        plt.show()

    return diagnostics, raw_corrs