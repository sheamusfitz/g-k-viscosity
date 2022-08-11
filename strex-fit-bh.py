from os import times
from sys import flags
import pandas as pd
import scipy as sp
from scipy import stats
# from scipy import odr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import time

# from statsmodels.stats.weightstats import DescrStatsW
# import cellbell
# import numdifftools
import lmfit as lf
import emcee
# from scipy.signal import savgol_filter
# import corner
# import os
# from multiprocessing import Pool

# import pickle

import argparse

parser = argparse.ArgumentParser(
    description='enter something between 0 and 15. this is the index of the lipid this script will process. 0-8 is in `integrals`, and 9-14 or 15 is in `integrals329`')

parser.add_argument('--index', '-i', nargs=1, type=int)
args = parser.parse_args()
thelipid = args.index[0]


font = {'family': 'serif',
        'weight': 'normal',
        'size': 18}

matplotlib.rc('font', **font)


def gamma_fit(t, a, b, t0):
    a = 10**a
    t0 = 10**t0
    if b <= 1:
        print('you need b > 1')
        return
    return(a * sp.special.gammainc(b, (t/t0)**(1/b)))


gamma_model = lf.Model(gamma_fit, independent_vars='t')


def gamma_limit(a, b, t0):
    return(10**a)


def log_uniform(min, max, n):
    lmin = np.log(min)
    lmax = np.log(max)
    rands = np.random.uniform(lmin, lmax, n)
    return(np.exp(rands))


temps = np.array([283, 293, 303, 313, 323], dtype=str)

integrals = pd.read_pickle('integrals.pkl')
print(integrals.columns)
most_times = integrals.times


lipids = ['dppc', 'dspc', 'dopc', 'dlpc']

integrals329 = pd.read_pickle('integrals329.pkl')

lipids329 = ['dapc', 'dlpc', 'dmpc', 'dopc', 'popc']


times329 = integrals329.times


def weighted_unc(arr):
    '''
    I took this from NIST. The structure of `arr` is:
    DataEntry_1 = arr[0, 0] ± arr[0, 1]
    '''
    w_mean = np.average(arr[:, 0], weights=arr[:, 1]**-2)

    return(
        np.sqrt(
            np.sum(
                arr[:, 1]**-2 * (arr[:, 0] - w_mean)**2
            )
            /
            (
                ((len(arr)-1) * np.sum(arr[:, 1]**-2))
                /
                (len(arr))
            )
        )
    )


integ_mean = pd.DataFrame()
integ_std = pd.DataFrame()

integ329_mean = pd.DataFrame()
integ329_std = pd.DataFrame()

print(lipids)

for lipid in lipids:
    integ_mean[lipid] = np.mean(integrals[
        integrals.columns.where(integrals.columns.str.contains(lipid)).dropna()
    ], axis=1)
    integ_std[lipid] = np.std(integrals[
        integrals.columns.where(integrals.columns.str.contains(lipid)).dropna()
    ], axis=1)

for lipid in temps:
    integ_mean[lipid] = np.mean(integrals[
        integrals.columns.where(integrals.columns.str.contains(lipid)).dropna()
    ], axis=1)
    integ_std[lipid] = np.std(integrals[
        integrals.columns.where(integrals.columns.str.contains(lipid)).dropna()
    ], axis=1)

for lipid in lipids329:
    integ329_mean[lipid] = np.mean(integrals329[
        integrals329.columns.where(
            integrals329.columns.str.contains(lipid)).dropna()
    ], axis=1)
    integ329_std[lipid] = np.std(integrals329[
        integrals329.columns.where(
            integrals329.columns.str.contains(lipid)).dropna()
    ], axis=1)

# mint = 10
# maxt = 8000*20*2
# num = 50000
# log_inds = np.unique(np.logspace(
# np.log10(mint), np.log10(maxt), num=num, dtype=int))
# log_inds = np.linspace(mint, maxt, num, dtype=int)
# display(log_inds)
# print(len(log_inds), log_inds[:3], log_inds[-3:])

##################################################

try:
    results
except:
    results = {}

##################################################

gamma_params = gamma_model.make_params()

# gamma_params.add('a', min = -100, max = 0, vary = True)
# gamma_params.add('b', min = 1, max = 70)
# gamma_params.add('t0', min = -2, max = 0, vary = True)

gamma_params.add('a',  min=-12,   max=-6, value=-10)    # 1e-11, 1e-8
gamma_params.add('b',  min=2,     max=20, value=6)      # 3, 18
gamma_params.add('t0', min=-15,   max=5,  value=-2)     # 1e-10, 1e-1

# initials = np.array([
#     log_uniform(gamma_params['a'].min, gamma_params['a'].max, nwalkers),
#     np.random.uniform(gamma_params['b'].min, gamma_params['b'].max, nwalkers),
#     log_uniform(gamma_params['t0'].max, gamma_params['t0'].min, nwalkers)
# ]).T


##################################################
# lip = '283'  # ['283', '293', '303', '313', '323', 'dlpc', 'dopc', 'dppc', 'dspc']
##################################################
for r in range(1, 6):
    starttime = time.time()
    print(thelipid)
    if 0 <= thelipid <= 8:
        lip = ['283', '293', '303', '313', '323',
               'dlpc', 'dopc', 'dppc', 'dspc'][thelipid]
        this_integral = integrals[f'{lip}-r{r}']
        uncs = integ_std[lip]
        means = integ_mean[lip]
        thesetimes = most_times
    elif 9 <= thelipid <= 13:
        lip = lipids329[thelipid-9]
        this_integral = integrals329[f'{lip}-r{r}']
        uncs = integ329_std[lip]
        means = integ329_mean[lip]
        thesetimes = times329
    mint = np.abs(thesetimes - 3).argmin() + 1
    maxt = np.abs(thesetimes - 5000).argmin() + 1
    print(mint, maxt)
    # log_inds = np.unique(np.logspace(
    #     np.log10(mint), np.log10(len(thesetimes)), num=50000, dtype=int))
    log_inds = np.unique(np.logspace(
        np.log10(mint), np.log10(maxt), num=50000, dtype=int))
    print(lip, r)
    results[f'{thelipid}-r{r}'] = \
        gamma_model.fit(this_integral[log_inds], gamma_params, t=thesetimes[log_inds],
                        method='basinhopping', weights=uncs[log_inds]**-2,
                        # fit_kws={'nwalkers': nwalkers, 'steps': nsteps,  # 10000
                        #          'pos': initials, 'workers': 16, 'progress': False},
                        calc_covar=True
                        )

    dur = time.time()-starttime
    durm = int(dur//60)
    durs = int(dur % 60)
    print(f'fitting time: {durm}m{durs}s')

    ##################################################

    raw_data = this_integral  # [mint:maxt]
    the_fit = results[f'{thelipid}-r{r}'].best_fit

    plt.figure(figsize=(9, 6))
    plt.plot(thesetimes, raw_data, 'k')
    plt.plot(thesetimes[log_inds], the_fit, 'r', linewidth=16, alpha=0.2)
    plt.plot(thesetimes, gamma_fit(thesetimes, *list(results[f'{thelipid}-r{r}'].best_values.values())),
             'r:', linewidth=4)
    plt.axhline(gamma_limit(*list(results[f'{thelipid}-r{r}'].best_values.values())), color='r', linestyle=':',
                linewidth=2
                )
    plt.fill_between(thesetimes, means+uncs,
                     means-uncs, alpha=0.2)
    # log
    plt.xscale('log')
    plt.ylim(bottom=1e-13, top=max([np.max(means),
                                    gamma_limit(
        *list(results[f'{thelipid}-r{r}'].best_values.values())),
        0]
    )*1.2)
    plt.yscale('log')

    # lin
    #     plt.xlim(-1000,70000)
    #     plt.ylim(0,0.7e-9)

    plt.title(
        f'Integrated\nStretched Exponential\n{lip.upper()} run {r}', pad=0)
    plt.xlabel('time (ps)')
    plt.ylabel('$\eta (t)$\n$(Pa\cdot m\cdot s)$')
    plt.savefig(f'{lip}-{thelipid}-r{r}-fit.png')
    plt.close()

    try:
        output_params
    except:
        output_params = {}
    try:
        output_params[lip]
    except:
        output_params[lip] = {}
    try:
        output_params[lip]['viscs'].shape
    except:
        output_params[lip]['viscs'] = np.zeros(5)

    try:
        output_params[lip]['mrt'].shape
    except:
        output_params[lip]['mrt'] = np.zeros(5)

    try:
        output_params[lip]['t0']
    except:
        output_params[lip]['t0'] = np.zeros(5)

    try:
        output_params[lip]['b']
    except:
        output_params[lip]['b'] = np.zeros(5)

    output_params[lip]['viscs'][r - 1] = (
        10**results[f'{thelipid}-r{r}'].params['a'].value + this_integral[0]
    )
    output_params[lip]['b'][r - 1] = (
        results[f'{thelipid}-r{r}'].params['b'].value
    )
    output_params[lip]['t0'][r - 1] = (
        10**results[f'{thelipid}-r{r}'].params['t0'].value
    )

try:
    output_combined
    print(0)
except:
    output_combined = {}
    print(1)

try:
    output_combined[lip]
    print(0)
except:
    output_combined[lip] = {}
    print(1)

print('------------------------------')

# output_combined[lip]['mean visc'] = np.average(
#     output_params[lip]['viscs'][:, 0],
#     weights=output_params[lip]['viscs'][:, 1]**-2
# )
# output_combined[lip]['visc unc'] = weighted_unc(output_params[lip]['viscs'])
#

output_combined[lip]['mean visc'] = np.mean(output_params[lip]['viscs'])
output_combined[lip]['visc unc'] = stats.sem(output_params[lip]['viscs'])

#
output_combined[lip]['mean t0'] = np.mean(output_params[lip]['t0'])
output_combined[lip]['t0 unc'] = stats.sem(output_params[lip]['t0'])
##
output_combined[lip]['mean b'] = np.mean(output_params[lip]['b'])
output_combined[lip]['b unc'] = stats.sem(output_params[lip]['b'])

with open(f'{lip}-{thelipid}-results.txt', 'a') as f:
    f.write(f"\n\n\n{lip.upper()} Results:")
    f.write(f"\nη\t= ({output_combined[lip]['mean visc']*1e11:0.4f} ± \
{output_combined[lip]['visc unc']*1e11:0.4f}) x10^-11 Pa.m.s")
    f.write(
        f"\nτ_0\t= ({output_combined[lip]['mean t0']:0.5f} ± {output_combined[lip]['t0 unc']:0.5f}) ps")
    f.write(
        f"\nb\t= ({output_combined[lip]['mean b']:0.3f} ± {output_combined[lip]['b unc']:0.3f})")
    f.write("\n\n\n")
    for thing in output_params[lip]:
        f.write(f"\n{thing}\n{output_params[lip][thing]}")
