from os import times
from sys import flags
import pandas as pd
import scipy as sp
from scipy import stats
# from scipy import odr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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
    #     a = 10**a
    #     t0 = 10**t0
    if b <= 1:
        print('you need b > 1')
        return
    return(a * sp.special.gammainc(b, (t/t0)**(1/b)))


gamma_model = lf.Model(gamma_fit, independent_vars='t')


def gamma_limit(a, b, t0):
    #     a = 10**a
    #     t0 = 10**t0
    return(a)


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

mint = 20
maxt = 8000
num = 1000
log_inds = np.unique(np.logspace(
    np.log10(mint), np.log10(maxt), num=num, dtype=int))
# log_inds = np.arange(mint, maxt)
# display(log_inds)
print(len(log_inds))


##################################################

try:
    results
except:
    results = {}

##################################################

nsteps = 500000  # 200000 # 40000 # psm used 4500*50 = 225000
# to get rid of the autocorrelation error, use AT LEAST 22969*50
nwalkers = 32  # 24  was pretty good #64, 16 for testing

##################################################
##################################################
##################################################
##################################################
##################################################

gamma_params = gamma_model.make_params()

# gamma_params.add('a', min = -100, max = 0, vary = True)
# gamma_params.add('b', min = 1, max = 70)
# gamma_params.add('t0', min = -2, max = 0, vary = True)

gamma_params.add('a',  min=1e-11,   max=1e-8)    # 1e-11 - 1e-1
gamma_params.add('b',  min=3,       max=18)      # in general, 1.1-20
gamma_params.add('t0', min=1e-10,   max=1e-1)     # 1e-15 - 1e4

initials = np.array([
    log_uniform(gamma_params['a'].min, gamma_params['a'].max, nwalkers),
    np.random.uniform(gamma_params['b'].min, gamma_params['b'].max, nwalkers),
    log_uniform(gamma_params['t0'].max, gamma_params['t0'].min, nwalkers)
]).T


##################################################
# lip = '283'  # ['283', '293', '303', '313', '323', 'dlpc', 'dopc', 'dppc', 'dspc']
##################################################
for r in range(1, 6):
    if 0 <= thelipid <= 8:
        lip = ['283', '293', '303', '313', '323',
               'dlpc', 'dopc', 'dppc', 'dspc'][thelipid]
        this_integral = integrals[f'{lip}-r{r}']
        uncs = integ_std[lip][log_inds]
        thesetimes = most_times
    elif 9 <= thelipid <= 13:
        lip = lipids329[thelipid]
        this_integral = integrals329[f'{lip}-r{r}']
        uncs = integ329_std[lip][log_inds]
        thesetimes = times329

    results[f'{thelipid}-r{r}'] = \
        gamma_model.fit(this_integral[log_inds], gamma_params, t=thesetimes[log_inds],
                        method='emcee', weights=uncs**-1,
                        fit_kws={'nwalkers': nwalkers, 'steps': nsteps,  # 10000
                                 'pos': initials, 'workers': 16},
                        calc_covar=True
                        )

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
    plt.fill_between(thesetimes, integ_mean[lip]+integ_std[lip],
                     integ_mean[lip]-integ_std[lip], alpha=0.2)
    # log
    plt.xscale('log')
    plt.ylim(bottom=1e-13, top=max([np.max(integ_mean[lip]),
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
    plt.savefig(f'{lip}-r{r}-fit.png')

    a_arr = np.log(results[f'{thelipid}-r{r}'].chain[-1, :, 0])
    a_arr0 = np.log(results[f'{thelipid}-r{r}'].chain[-1, :, 0])
    med = np.median(a_arr)
    q1 = np.quantile(a_arr, 0.25)
    q3 = np.quantile(a_arr, 0.75)
    iqr = q3-q1
    upper = q3+iqr*1.5
    lower = q1-iqr*1.5

    plt.figure()
    plt.plot(a_arr)
    plt.axhline(med)
    plt.axhline(q1)
    plt.axhline(q3)
    plt.axhline(upper, c='r')
    plt.axhline(lower, c='r')

    awhere_up = np.where(a_arr < upper)
    awhere_lo = np.where(a_arr > lower)
    awhere = np.intersect1d(awhere_up, awhere_lo)
    print(awhere)
    plt.plot(awhere, a_arr[awhere], 'k')
    plt.title(f'{lip}-r{r}: a')
    plt.savefig(f'{lip}-r{r}-awhere.png')

    ##################################################

    b_arr = np.log(results[f'{thelipid}-r{r}'].chain[-1, :, 1])
    b_arr0 = np.log(results[f'{thelipid}-r{r}'].chain[-1, :, 1])
    med = np.median(b_arr)
    q1 = np.quantile(b_arr, 0.25)
    q3 = np.quantile(b_arr, 0.75)
    iqr = q3-q1
    upper = q3+iqr*1.5
    lower = q1-iqr*1.5

    plt.figure()
    plt.plot(b_arr)
    plt.axhline(med)
    plt.axhline(q1)
    plt.axhline(q3)
    plt.axhline(upper, c='r')
    plt.axhline(lower, c='r')

    bwhere_up = np.where(b_arr < upper)
    bwhere_lo = np.where(b_arr > lower)
    bwhere = np.intersect1d(bwhere_up, bwhere_lo)
    print(bwhere)
    plt.plot(bwhere, b_arr[bwhere], 'k')
    plt.title(f'{lip}-r{r}: b')
    plt.savefig(f'{lip}-r{r}-bwhere.png')

    ##################################################

    t_arr = np.log(results[f'{thelipid}-r{r}'].chain[-1, :, 2])
    t_arr0 = np.log(results[f'{thelipid}-r{r}'].chain[-1, :, 2])
    med = np.median(t_arr)
    q1 = np.quantile(t_arr, 0.25)
    q3 = np.quantile(t_arr, 0.75)
    iqr = q3-q1
    upper = q3+iqr*1.5
    lower = q1-iqr*1.5

    plt.figure()
    plt.plot(t_arr)
    plt.axhline(med)
    plt.axhline(q1)
    plt.axhline(q3)
    plt.axhline(upper, c='r')
    plt.axhline(lower, c='r')

    twhere_up = np.where(t_arr < upper)
    twhere_lo = np.where(t_arr > lower)
    twhere = np.intersect1d(twhere_up, twhere_lo)
    print(twhere)
    plt.plot(twhere, t_arr[twhere], 'k')
    plt.title(f'{lip}-r{r}: t0')
    plt.savefig(f'{lip}-r{r}-twhere.png')

    ##################################################

    wheres = np.intersect1d(awhere, bwhere)
    wheres = np.intersect1d(wheres, twhere)

    lastn = 1000
    lastviscs = np.zeros((lastn, len(wheres)))
    last_mrts = np.zeros((lastn, len(wheres)))
    last_b = np.zeros((lastn, len(wheres)))

    for i, walker in enumerate(wheres):
        lastviscs[:, i] = (
            results[f'{thelipid}-r{r}'].chain[-lastn:, walker, 0])
        last_mrts[:, i] = (results[f'{thelipid}-r{r}'].chain[-lastn:, walker, 2]) *\
            sp.special.gamma(
                results[f'{thelipid}-r{r}'].chain[-lastn:, walker, 1]+1)
        last_b[:, i] = results[f'{thelipid}-r{r}'].chain[-lastn:, walker, 1]
    plt.figure()
    plt.plot(np.arange(lastn), lastviscs)
    plt.title('viscosity')
    plt.yscale('log')
    plt.ylim(1e-11, 1e-5)
    plt.savefig(f'{lip}-r{r}-visc-walkers.png')
    plt.figure()
    plt.plot(np.arange(lastn), last_mrts)
    plt.title('mean relaxation time')
    plt.ylim(0, 60)
    plt.savefig(f'{lip}-r{r}-t-walkers.png')

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
        output_params[lip]['viscs'] = np.zeros((5, 2))

    try:
        output_params[lip]['mrt'].shape
    except:
        output_params[lip]['mrt'] = np.zeros((5, 2))

    try:
        output_params[lip]['b']
    except:
        output_params[lip]['b'] = np.zeros((5, 2))

    output_params[lip]['viscs'][r -
                                1] = (np.mean(lastviscs)+integrals[f'{lip}-r{r}'][0], np.std(lastviscs))
    output_params[lip]['mrt'][r -
                              1] = (np.mean(last_mrts), np.std(last_mrts))
    output_params[lip]['b'][r-1] = (np.mean(last_b), np.std(last_b))

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

output_combined[lip]['mean visc'] = np.average(output_params[lip]['viscs'][:, 0],
                                               weights=output_params[lip]['viscs'][:, 1]**-2)
output_combined[lip]['visc unc'] = weighted_unc(
    output_params[lip]['viscs'])
#
#
output_combined[lip]['mean mrt'] = np.average(output_params[lip]['mrt'][:, 0],
                                              weights=output_params[lip]['mrt'][:, 1]**-2)
output_combined[lip]['mrt unc'] = weighted_unc(output_params[lip]['mrt'])
#
#
output_combined[lip]['mean b'] = np.average(output_params[lip]['b'][:, 0],
                                            weights=output_params[lip]['b'][:, 1]**-2)
output_combined[lip]['b unc'] = weighted_unc(output_params[lip]['b'])

print(f"{lip.upper()} Results:")
print(f"η\t= ({output_combined[lip]['mean visc']*1e11:0.4f} ± \
{output_combined[lip]['visc unc']*1e11:0.4f}) x10^-11 Pa.m.s")
print(
    f"τ_0\t= ({output_combined[lip]['mean mrt']:0.5f} ± {output_combined[lip]['mrt unc']:0.5f}) ps")
print(
    f"b\t= ({output_combined[lip]['mean b']:0.3f} ± {output_combined[lip]['b unc']:0.3f})")
# np.savez_compressed(results)
