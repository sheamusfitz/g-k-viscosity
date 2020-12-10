import scipy as sp
from scipy import stats
import pandas as pd
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
  'dmpc-8nm',
  'dmpc-12nm',
  'dmpc-15nm',
]:
  print(lipid)
  lipidfolder = lipid
  print(lipidfolder)
  colnames = ['t','x','y','z']

  box1 = pd.read_table(lipidfolder+'/r1/box.xvg', header=None, skiprows=29, usecols=[1,2,3,4], names=colnames)
  print('r1')
  box2 = pd.read_table(lipidfolder+'/r2/box.xvg', header=None, skiprows=29, usecols=[1,2,3,4], names=colnames)
  print('r2')
  box3 = pd.read_table(lipidfolder+'/r3/box.xvg', header=None, skiprows=29, usecols=[1,2,3,4], names=colnames)
  print('r3')
  box4 = pd.read_table(lipidfolder+'/r4/box.xvg', header=None, skiprows=29, usecols=[1,2,3,4], names=colnames)
  print('r4')
  box5 = pd.read_table(lipidfolder+'/r5/box.xvg', header=None, skiprows=29, usecols=[1,2,3,4], names=colnames)
  print('r5')

  allbox = pd.concat([box1,box2,box3,box4,box5])

  targetx = sp.mean(allbox.x)
  targetz = sp.mean(allbox.z)

  # with open(lipidfolder+'/targetvals.txt','w') as f:
  #    f.write(str(targetx)+'\t'+str(targetz))



  bestarr = sp.zeros((5),dtype=int)
  for b, box in enumerate([box1,box2,box3,box4,box5]):
    minerror = 100
    for frame in range(100,len(box)):
      thiserror = finderror(box.x[frame],box.z[frame])
      if thiserror < minerror:
        bestarr[b] = frame
        minerror = thiserror
        # print(bestarr[b],minerror)

  bestarr *= box1.t[1]
  # print(bestarr)

  # with open(lipidfolder+'/bestndx.txt', 'w') as f:
  #    for i in range(5):
  #       f.write(str(bestarr[i])+'\n')

  # gmx grompp -f r1/gromacs/step7_boxsize.mdp -t r1/gromacs/step7_boxsize.trr -time 160.0 -o r1/gromacs/ZZZstep8_nvt.tpr -c r1/gromacs/step6.6_equilibration.gro -n  r1/gromacs/index.ndx -p r1/gromacs/topol.top
  nvtfolder = lipid

  with open('./grompp.txt', 'a') as f:
    for i in range(5):
      ri = '/r'+str(i+1)
      grompp = 'gmx grompp -f ' + nvtfolder + '/step8_nvt.mdp -t ' + lipidfolder+ri + '/step7_boxsize.trr -time ' + str(bestarr[i]) + ' -o ' + nvtfolder+ri + '/step8_nvt.tpr -c ' + lipidfolder+ri + '/step7_boxsize.gro -n ' + nvtfolder+ri + '/index.ndx -p ' + nvtfolder+ri + '/topol.top'
      f.write(grompp+'\n')
    # print(grompp,'\n')
