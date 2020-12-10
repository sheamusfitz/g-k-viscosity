import scipy as sp
from scipy import stats
import pandas as pd
import numpy as np
# import MDAnalysis as mda
# import re
# import json
# import readline
# from termcolor import cprint
# import os, sys, getopt
# import warnings

# print('input lipid folder name:')
# lipidfolder = input()
def finderror(x, z):
  error = (x-targetx)**2+(z-targetz)**2
  return error
print(0)
open('./grompp.txt', 'w').close()

for lipid in [
  'psm'
  ]:

  print(lipid)
  lipidfolder = lipid
  print(lipidfolder)

  for i in range(1,6):
    colnames = ['t', 'x', 'y', 'z']
    boxi = pd.read_table(lipidfolder+f'/r{i}/box.xvg', header=None, skiprows=29, usecols=[1,2,3,4], names=colnames)

    targetx = sp.mean(boxi.x)
    targetz = sp.mean(boxi.z)

    bestarr = sp.zeros((5),dtype=int)
    minerror = 100
    for frame in len(boxi):
      thiserror = finderror(boxi.x[frame], boxi.z[frame])
      if thiserror < minerror:
        bestarr[b] = frame
        minerror = thiserror

  bestarr *= boxi.t[1]
  nvtfolder = lipid

  with open('./grompp.txt', 'a') as f:
    for i in range(5):
      ri = '/r'+str(i+1)
      grompp = 'gmx grompp -f ' + nvtfolder + '/step8_nvt.mdp -t ' + lipidfolder+ri + '/step7_boxsize.trr -time ' + str(bestarr[i]) + ' -o ' + nvtfolder+ri + '/step8_nvt.tpr -c ' + lipidfolder+ri + '/step7_boxsize.gro -n ' + nvtfolder+ri + '/index.ndx -p ' + nvtfolder+ri + '/topol.top'
      f.write(grompp+'\n')
    # print(grompp,'\n')
