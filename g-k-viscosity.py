import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal


#Aprint('NOTE: this does not yet account for the water viscosity or the thickness of the membrane *relative* to the box.')

# xvgname = '../nvt_analysis/pressure.xvg'
# heightname = '../nvt_analysis/thickness.xvg'
# struct_filename = '../step7_short.gro'
# #Aprint(xvgplace)

xvgname = './pressure-tensor.xvg'
xvgplace = ''.join(xvgname.partition('/')[:-1])
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
                          skiprows=33,
                          header=None, delim_whitespace=True)

bigpressure.columns = ['time (s)', 'temp', 'xy', 'yx']

print('importing height')

height_v = pd.read_csv(heightname, skiprows=17, header=None, delim_whitespace=True, dtype=np.float)
height_v.columns = ['t','x','y','z']

stepsize = bigpressure['time (s)'][1] * 10**(-12)
print("stepsize =", stepsize, "s")


def main():
  """
  This is the basic. What I want. Anything further should be developed using this code.
  """
  stress = np.mean([bigpressure.xy, bigpressure.yx], axis=0)
  print('\nlen stress', len(stress))
  print('mean stress (Pa) =', np.mean(stress)*10**5)
  print('mean stress^2 (Pa^2) =', np.mean(stress**2)*10**10)

  thickness = np.absolute(np.mean(height_v.z)) * 10**(-9)
  print('\nthickness =', thickness, "m")

  temp = bigpressure['temp'].mean()
  print("\ntemperature =",temp,"K")

  boxvol = sizes[0]*sizes[1]*sizes[2] * 10**-27
  print("\nboxvol =", boxvol, "m^3")
  print("\ncalculating autocor")


  def autocor(sig):
    return(
      sp.signal.correlate(sig, sig, mode='full')[len(sig)-1:] / 
      np.arange(len(sig)+1,1,-1)
    )

  stress_autocor = autocor(stress)

  print("\nintegrating")

  integrated = np.cumsum(stress_autocor)

  viscosityfactor = (
    10**10                  #bar^2 to Pa^2
    * stepsize
    * boxvol
    * (1.38064852 * 10**-23)**(-1) #1/k_B
  )
  
  visco_arr = viscosityfactor / temp * thickness * integrated
  
  print("\nsaving")

  names = np.array([
    'names',
    'times',
    'xy stress autocorrelation',
    'viscosity at each timestep (from 0)',
    ])

  print('\nlen(stress_autocor)', len(stress_autocor))

  nn = 1000
  print(f"saving every {nn} timesteps, which is every {nn*stepsize*10**12} picoseconds")

  np.savez_compressed(
    './py_output.npz',
    names,
    bigpressure['time (s)'][::nn],
    stress_autocor[::nn],
    visco_arr[::nn],
    )

main()