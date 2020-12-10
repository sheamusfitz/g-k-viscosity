import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
# import matplotlib.pyplot as plt
# from statsmodels.stats.weightstats import DescrStatsW
from scipy import linalg

#Aprint('NOTE: this does not yet account for the water viscosity or the thickness of the membrane *relative* to the box.')

# xvgname = '../nvt_analysis/pressure.xvg'
# heightname = '../nvt_analysis/thickness.xvg'
# struct_filename = '../step7_short.gro'

xvgname = './pressure-tensor.xvg'
heightname = './thickness.xvg'
struct_filename = './step8_nvt.gro'
with open(struct_filename) as f:
  sizeline = f.readlines()[-1]
#Aprint(sizeline)
sizes = sizeline.split()
for i,size in enumerate(sizes):
  sizes[i] = float(size)

print("importing...")
bigpressure = pd.read_csv(xvgname,
                          skiprows=26,
                          header=None, delim_whitespace=True, dtype=np.float,
                          low_memory=False)

bigpressure.columns = ['time (ps)', 'temp', 'xy', 'yx']

print('importing height')

height_v = pd.read_csv(heightname, skiprows=17, header=None, delim_whitespace=True, dtype=np.float)
height_v.columns = ['t','x','y','z']

stepsize = (bigpressure['time (s)'][1]-bigpressure['time (s)'][0]) * 10**(-12)
print("stepsize =", stepsize, "s")

def main(nn = 1, npz_name = 'py_output.npz'):
  #make stress tensor
  print('just some simple manipulations')
  # actually use 10000
  stress = np.mean([bigpressure.xy[::nn], bigpressure.yx[::nn]], axis=0)
  stress = stress - np.mean(stress)

  print('mean stress^2 (Pa^2) =', np.mean(stress**2)*10**10)

  thickness = np.absolute(np.mean(height_v.z)) * 10**(-9)
  # thicksem = sp.stats.sem(height_v.z) * 10**(-9)
  print('thickness =', thickness, "m")

  #  this all seems wrong to me... i'll fix it here we go:
  temp = bigpressure['temp'][::nn].mean()
  tempsem = sp.stats.sem(bigpressure['temp'][::nn])
  print("temperature =",temp,"K")

  boxvol = sizes[0]*sizes[1]*sizes[2] * 10**-27 # volume in m^3
  print("boxvol =", boxvol, "m^3")
  print('autocor')

  def autocor(sig):
    return(
      sp.signal.correlate(sig, sig, mode='full')[len(sig)-1:] / 
      np.arange(len(sig)+1,1,-1)
    )

  stress_autocor = autocor(stress)

  viscosityfactor = f"(\
    10**10 bar^2 to Pa^2\
    * stepsize {stepsize*nn}\
    * boxvol {boxvol}\
    * (1.38064852 * 10**-23)**(-1) 1/k_B)"

  print('saving')

  names = np.array([
    'names',
    'xy stress autocorrelation',
    'viscosityfactor'
    ])

  np.savez_compressed(
    './'+npz_name,
    names,
    stress_autocor,
    viscosityfactor
    )
  # visco_rolling_stats = sp.array([DescrStatsW(visco_arr[:i], weights=normed(visco_uncertainties[:i]**(-2)), ddof=0) for i in range(1,len(integrated))])

if stepsize==2e-15:
  main(nn=1000, npz_name='pico_autocor.npz')
  main(nn=1, npz_name='femto_autocor.npz')
elif stepsize==2e-12:
  main(nn=1, npz_name='pico_autocor.npz')
else:
  main(nn=1, npz_name='py_output.npz')
