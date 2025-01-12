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

    return sumdms / (np.nan_to_num(sumsqrs) + 1e-8)

def rank_corr(preds_ur, targs_ur): # ~spearman correlation
    preds = rank01(preds_ur)
    targs = rank01(targs_ur)
    
    preds_dm = preds - np.mean(preds)
    targs_dm = targs - np.mean(targs)

    sumdms = np.sum(preds_dm * targs_dm)
    sumsqrs = np.sqrt(np.sum(preds_dm ** 2) * np.sum(targs_dm ** 2))

    return sumdms / (np.nan_to_num(sumsqrs) + 1e-8)


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
    
    return prod / (np.nan_to_num(np.sqrt(ddiffs2 * idiff2)) + 1e-8)

def neutralize(target, by, proportion=1.0):
    by = by - np.mean(by)
    target = target - np.mean(target)
    scores = target - (proportion * by.dot(np.linalg.pinv(by).dot(target)))
    return scores / (np.nan_to_num(scores.std()) + 1e-8)

def era_neutralize(target, by, I, proportion=1.0):
    out = np.empty(target.shape)
    for E in I:
        targeti = target[E]; byi = by[E]
        out[E] = neutralize(targeti, byi, proportion)
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
            ndd = (running / max_val) - 1
            if ndd < dd:
                dd = ndd
    return dd

def wkly_stake_drawdown(x):
    stake = [1.0, 1.0, 1.0, 1.0]
    max_val = 1.0; running = 1.0; dd = 0
    for i in x:
        stake.append(max(stake[-1] + (stake[-4] * max(min(i, 0.25), -0.25)), 0.0))
        running = stake[-1]
        if running > max_val:
            max_val = running
        else:
            ndd = (running / max_val) - 1
            if ndd < dd:
                dd = ndd
    return dd

def ts_stats(ts,name=''):
    diagnostics = {}
    diagnostics[name+' mean'] = np.nanmean(ts); diagnostics[name+' sdev'] = np.nanstd(ts)
    diagnostics[name+' sharpe'] = diagnostics[name+' mean'] / diagnostics[name+' sdev']; 
    diagnostics[name+' sortino (-0.02)'] = sortino_ratio(ts, 0.02)
    # diagnostics[name+' drawdown'] = drawdown(ts)
    diagnostics[name+' staked drawdown'] = wkly_stake_drawdown(ts)
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
            ces = np.empty(len(I))
            for i in range(len(ces)): ces[i] = simple_corr(neutralize(P[I[i]].reshape(-1,1),v[I[i]].reshape(-1,1)),Y[I[i]])
            raw_corrs[k+' neut corr (mmc)'] = ces
            diagnostics.update(ts_stats(ces,k+' neut corr (mmc)'))
            # add comparison multipliers, note: will overwrite and only use last one
            raw_corrs['0.5xCorr 2xMMC'] = (raw_corrs['corrv2'] * 0.5) + (raw_corrs[k+' neut corr (mmc)'] * 2.0)
            diagnostics.update(ts_stats(raw_corrs['0.5xCorr 2xMMC'],'0.5xCorr 2xMMC'))
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

        # add a second plot with cumulative returns
        plt.figure(figsize=(14,6))
        for k,v in raw_corrs.items(): 
            stake = [1.0, 1.0, 1.0, 1.0]
            for i in range(len(v)):
                stake.append(max(stake[-1] + (stake[-4] * max(min(v[i], 0.25), -0.25)), 0.0))
            plt.plot(stake[3:], label=k)
        # make plot log scale
        plt.yscale('log')
        plt.legend()
        plt.show()

        # add a third plot only 1 year cumulative returns
        plt.figure(figsize=(14,6))
        for k,v in raw_corrs.items(): 
            v52 = v[-52:]
            stake = [1.0, 1.0, 1.0, 1.0]
            for i in range(len(v52)):
                stake.append(max(stake[-1] + (stake[-4] * max(min(v52[i], 0.25), -0.25)), 0.0))
            plt.plot(stake[3:], label=k)
        # make plot log scale
        plt.yscale('log')
        plt.legend()
        plt.show()

    return diagnostics, raw_corrs