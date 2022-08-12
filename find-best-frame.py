import scipy as sp
import pandas as pd
import numpy as np
import os

runpath = os.getcwd()
lipidpath = runpath
digits = [str(i) for i in np.arange(10)]

if lipidpath[-1] == '/':
    lipidpath = lipidpath[:-1]
if lipidpath[-1] in digits:
    lipidpath = lipidpath[:-1]
    if lipidpath[-1] == 'r':
        lipidpath = lipidpath[:-1]
    else:
        raise


# print('input lipid folder name:')
# lipidfolder = input()
def finderror(x, z):
    error = (x-targetx)**2+(z-targetz)**2
    return error


print(0)
open('./grompp.txt', 'w').close()

colnames = ['t', 'x', 'y', 'z']

box = pd.read_csv('box.xvg', header=None, skiprows=29, usecols=[
                  1, 2, 3, 4], names=colnames, sep='\t')

targetx = sp.mean(box.x)
targetz = sp.mean(box.z)

minerror = 1e10
for frame in range(len(box)):
    thiserror = finderror(box.x[frame], box.z[frame])
    if thiserror < minerror:
        bestframe = frame
        minerror = thiserror

besttime = box.t[bestframe]

with open(lipidpath+'/grompp.txt', 'a') as f:
    grompp = 'gmx_mpi grompp -f ' + lipidpath + '/step8_nvt.mdp -t ' + runpath + '/step7_boxsize.trr -time ' + \
        str(besttime) + ' -o ' + runpath + '/step8_nvt.tpr -c ' + runpath + \
        '/step7_boxsize.gro -n ' + runpath + '/index.ndx -p ' + runpath + '/topol.top'
    f.write(grompp+'\n')
