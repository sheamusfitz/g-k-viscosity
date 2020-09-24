import pandas as pd
import dask.dataframe as dd

import scipy as sp
import numpy as np
from scipy import stats
from scipy import linalg

print('started')

# Aprint('NOTE: this does not yet account for the water viscosity or the
# thickness of the membrane *relative* to the box.')

xvgname = "./pressure-tensor.xvg"
xvgplace = "".join(xvgname.partition("/")[:-1])
heightname = "./thickness.xvg"
struct_filename = "./step8_nvt.gro"
with open(struct_filename) as f:
    sizeline = f.readlines()[-1]
# Aprint(sizeline)
sizes = sizeline.split()
for i, size in enumerate(sizes):
    sizes[i] = float(size)

skipping = 0  # 97000 # 10,000,000 for 2fs sampling

print('pandas importing')

bigpressure = dd.read_csv(xvgname, skiprows=26+skipping, header=None, delim_whitespace=True)


bigpressure.columns = [
    "time (ps)",
    "temp",
    # "xx",
    "xy",
    # "xz",
    "yx",
    # "yy",
    # "yz",
    # "zx",
    # "zy",
    # "zz",
]

# @ s0 legend "Temperature"
# @ s1 legend "Pres-XX"
# @ s2 legend "Pres-XY"
# @ s3 legend "Pres-XZ"
# @ s4 legend "Pres-YX"
# @ s5 legend "Pres-YY"
# @ s6 legend "Pres-YZ"
# @ s7 legend "Pres-ZX"
# @ s8 legend "Pres-ZY"
# @ s9 legend "Pres-ZZ"

height_v = pd.read_csv(heightname, skiprows=17, header=None, delim_whitespace=True)
height_v.columns = ["t", "x", "y", "z"]

stepsize = (
    bigpressure["time (ps)"].loc[1].values
    - bigpressure["time (ps)"].loc[0].values
).compute() * 10 ** (-12)

stress = bigpressure[['xy','yx']].mean(axis=1)

# NOTE: this shifts the "stress" array down so that the mean is zero
stress -= stress.mean()

thickness = sp.absolute(sp.mean(height_v.z)) * 10 ** (-9)
thicksem = sp.stats.sem(height_v.z) * 10 ** (-9)


def autocor(arr, tau):
    if tau > len(arr):
        print("tau ", tau, "    lenarr ", len(arr))
        return "tau set too large"
    aaa = arr.loc[tau:].values * arr.loc[:len(stress) - tau - 1].values
    # if tau % 100 == 0:
    #     print(f"\r{(tau/len(arr))*100:6.2f}", "%   ", end="")
    return(aaa.mean(),(aaa.var()/(len(arr)-tau))**0.5)

# stress_autocor = sp.zeros((len(stress)-1, 2))
# for tau in range(len(stress)-1):
#     stress_autocor[tau] = autocor(stress, tau)

# print('got autocor')

def autocor_full(arr):
#     arr = arr_in.to_dask_array(lengths=True)
    autocor = sp.zeros(Len-2*Max+1)
#     autocor_unc = sp.zeros(Len-2*Max+1)
    for tau in range(Len-2*Max+1):
        if tau%1000 == 0:
            print(tau/(Len-2*Max+1))
        autocor[tau] = arr.autocorr(tau, split_every=100).compute()
    return(autocor)

temp = bigpressure["temp"].mean().compute()
tempsem = bigpressure["temp"].sem().compute()

boxvol = sizes[0] * sizes[1] * sizes[2] * 10 ** -27  # volume in m^3

Len = len(stress)
Max = 10000

stress_autocor, sem_list = autocor_full(stress)



# TODO i can do better here on the integration
integrated = sp.array([sp.sum(stress_autocor[:i+1]) for i in len(range(stress_autocor))])

# integrated = sp.array([sp.sum(stress_autocor[: i + 1, 0]) for i in range(len(stress_autocor))])

print('integrated')

unclist = sp.array(
    [sp.sqrt(sp.sum(sem_list[:i+1] ** 2))
        for i in range(len(integrated))]
)
unclist = sp.insert(unclist, 0, unclist[0])
unclist = sp.array(unclist, dtype=float)
# #Aprint(sp.amin(unclist**-2))

print('uncertainties')

##################################################

# movingavg = sp.zeros_like(stress_autocor[:, 0])
# windowsize = 25
# for i in range(len(stress_autocor) - windowsize):
#     movingavg[i] = sp.average(stress_autocor[i : i + windowsize, 0])

##################################################

# integrated = [sp.sum(stress_autocor[:i]) for i in range(len(stress_autocor))]

viscosityfactor = (
    10 ** 10  # bar^2 t Pa^2
    * stepsize
    * boxvol
    * (1.38064852 * 10 ** -23) ** (-1)  # 1/k_B
)

visco_arr = viscosityfactor / temp * thickness * integrated
visco_uncertainties = sp.absolute(visco_arr) * sp.sqrt(
    (tempsem / temp) ** 2 + (thicksem / thickness) ** 2 + (unclist / integrated) ** 2
)


def normed(x):
    return x / sp.linalg.norm(x)


# visco_rolling_stats = sp.array([DescrStatsW(visco_arr[:i], weights=normed
# (visco_uncertainties[:i]**(-2)), ddof=0) for i in range(1,len(integrated))])

# visco_rolling = sp.array([stat.mean for stat in visco_rolling_stats])

visco_rolling = sp.array(
    [
        sp.average(visco_arr[:i], weights=visco_uncertainties[:i] ** (-2))
        for i in range(1, len(visco_arr))
    ]
)
visco_rolling = sp.insert(visco_rolling, 0, visco_arr[0])
visco_rolling = sp.array(visco_rolling, dtype=float)

print('visco rolling avg')

# Aprint('means')

# THIS ONE DIDN'T WORK
# visco_sem = sp.array(
#   [
#     sp.var(visco_uncertainties[:i])
#     * sp.sum(visco_uncertainties[:i]**(-4))
#     / (sp.sum(visco_uncertainties[:i]**(-2))**2)
#     for i in range(1, len(visco_arr))
#     ]
#   )**(1/2)

visco_sem = sp.array(
    [
        (
            (
                sp.average(visco_arr[:i] ** 2, weights=visco_uncertainties[:i] ** (-2))
                - sp.average(visco_arr[:i], weights=visco_uncertainties[:i] ** (-2))
                ** 2
            )
        )
        for i in range(2, len(visco_arr))
    ]
) ** (1 / 2)
visco_sem = sp.insert(visco_sem, 0, visco_sem[0])
visco_sem = sp.insert(visco_sem, 0, visco_sem[0])
visco_sem = sp.array(visco_sem, dtype=float)

print('viscosem')

names = sp.array(
    [
        "names",
        "viscosity at each timestep (from 0)",
        "viscosity uncertainties",
        "rolling weighted average viscosity from 0",
        "SEM for rolling average viscosity",
        "original autocorrelation function (and uncertainties)"
    ]
)

print('saving')

sp.savez_compressed(
    "./py_output_shift.npz", names, visco_arr, visco_uncertainties,
    visco_rolling, visco_sem, stress_autocor
)