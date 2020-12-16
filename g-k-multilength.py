import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
# import matplotlib.pyplot as plt #NOTE remove this before running on shit

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

bigpressure.columns = ['time (s)', 'temp', 'xy', 'yx']

# print(np.mean(bigpressure.xy))

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

print('importing height')

height_v = pd.read_csv(heightname, skiprows=17, header=None, delim_whitespace=True, dtype=np.float)
height_v.columns = ['t','x','y','z']

stepsize = bigpressure['time (s)'][1] * 10**(-12)
print("stepsize =", stepsize, "s")

def subsampler(nn = 1, npz_name = 'py_output.npz'):
  '''
  This takes the output trajectory and makes a subsampled 'lower resolution'
  copy. *yes. i absolutely could have used this for the normal shit.*
  '''

  #make stress tensor
  print('just some simple manipulations')
  # actually use 10000
  stress = np.mean([bigpressure.xy[::nn], bigpressure.yx[::nn]], axis=0)
  stress = stress - np.mean(stress)
  print('len stress', len(stress))

  print('mean stress^2 (Pa^2) =', np.mean(stress**2)*10**10)

  thickness = np.absolute(np.mean(height_v.z)) * 10**(-9)
  # thicksem = sp.stats.sem(height_v.z) * 10**(-9)
  print('thickness =', thickness, "m")

  #  this all seems wrong to me... i'll fix it here we go:
  temp = bigpressure['temp'][::nn].mean()
  # tempsem = sp.stats.sem(bigpressure['temp'][::nn])
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

  print('integrating')

  integrated = np.cumsum(stress_autocor)

  viscosityfactor = (
    10**10                  #bar^2 to Pa^2
    * stepsize*nn
    * boxvol
    * (1.38064852 * 10**-23)**(-1) #1/k_B
  )

  visco_arr = viscosityfactor / temp * thickness * integrated

  print('saving')

  names = np.array([
    'names',
    'times',
    'xy stress autocorrelation',
    'viscosity at each timestep (from 0)',
    ])

  mm = np.max((1, int(len(stress_autocor)//1e6)))

  print('mm', mm)
  print('len(stress_autocor)', len(stress_autocor))

  np.savez_compressed(
    './'+npz_name,
    names,
    bigpressure['time (s)'][::nn*mm],
    stress_autocor[::mm],
    visco_arr[::mm],
    )


def trunc_integ(maxi = 100000, npz_name = 'py_output.npz'):
  maxi = int(maxi)
  #make stress tensor
  print('just some simple manipulations')
  # actually use 10000
  stress = np.mean([bigpressure.xy[:maxi], bigpressure.yx[:maxi]], axis=0)
  stress = stress - np.mean(stress)
  print('len stress', len(stress))

  print('mean stress^2 (Pa^2) =', np.mean(stress**2)*10**10)

  thickness = np.absolute(np.mean(height_v.z)) * 10**(-9)
  # thicksem = sp.stats.sem(height_v.z) * 10**(-9)
  print('thickness =', thickness, "m")

  #  this all seems wrong to me... i'll fix it here we go:
  temp = bigpressure['temp'][:maxi].mean()
  # tempsem = sp.stats.sem(bigpressure['temp'][::nn])
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

  print('integrating')

  integrated = np.cumsum(stress_autocor)

  viscosityfactor = (
    10**10                  #bar^2 to Pa^2
    * stepsize
    * boxvol
    * (1.38064852 * 10**-23)**(-1) #1/k_B
  )

  visco_arr = viscosityfactor / temp * thickness * integrated

  print('saving')

  names = np.array([
    'names',
    'times',
    'xy stress autocorrelation',
    'viscosity at each timestep (from 0)',
    ])

  mm = np.max((1, int(len(stress_autocor)//1e6)))

  print('mm', mm)
  print('len(stress_autocor)', len(stress_autocor))

  np.savez_compressed(
    './'+npz_name,
    names,
    bigpressure['time (s)'][:maxi:mm],
    stress_autocor[::mm],
    visco_arr[::mm],
    )

for ns in np.linspace(1, 196, 6):
  trunc_integ(maxi=ns*1000/0.002, npz_name=f'{ns}ns_sim.npz')


# if stepsize==2e-15:
#   subsampler(nn=1000, npz_name='pico_autocor.npz')
#   subsampler(nn=1, npz_name='femto_autocor.npz')
# elif stepsize==2e-12:
#   subsampler(nn=1, npz_name='pico_autocor.npz')
# else:
#   subsampler(nn=1, npz_name='py_output.npz')