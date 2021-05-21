import numpy as np
import pandas as pd
# import scipy as sp
from scipy import signal

import argparse

# I want to add an option to also save the fourier transform of the autocorrelation function.
# Initialize the parser
parser = argparse.ArgumentParser()
# create argument
parser.add_argument("-f", "--fft", help = "Output fourier transform of autocorrelation function", action="store_true")




# read argument from command line
args = parser.parse_args()





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
  sizes[i] = float(size) * 10**(-9)

print("importing...")
bigpressure = pd.read_csv(xvgname,
                          skiprows=33,
                          header=None, delim_whitespace=True)

bigpressure.columns = ['time (s)', 'temp', 'xy', 'yx']

print('importing height')

height_v = pd.read_csv(heightname, skiprows=17, header=None, delim_whitespace=True, dtype=np.float)
height_v.columns = ['t','x','y','z']

stepsize = (bigpressure['time (s)'][1]-bigpressure['time (s)'][0]) * 10**(-12)
print("stepsize =", stepsize, "s")

def water_visc(temp):
  a = 6.0097e-5
  b = 363.741484
  c = 79.9472888
  return(a * np.exp(b/(temp - c)))

def water_correction(eta_tot_arr, H, h, temp):
  """
  This takes the viscosity integral η(t) of the entire system and corrects it
    for the water viscosity as a function of temperature.
  **inputs**
  eta_tot_arr:  the viscosity integral
  H:            Box-Z dimension
  h:            membrane thickness (z)
  temp:         system temperature
  """
  return(H * eta_tot_arr - (H-h) * water_visc(temp))


def main(datapoints = 1000000):
  """
  This is the basic algorithm. It takes 'thickness.xvg', 'step8_nvt.gro', 'pressure-tensor.xvg' and outputs an npz file with times, the autocorrelation function, and the viscosity as a function of time. Anything further should be developed using this code.

  Optional keyword arguments:

  datapoints: sets (roughly) the number of data points in the output file. Set to 0 to output the entire series.

  ---sheamusfitz
  """

  stress = np.mean([bigpressure.xy, bigpressure.yx], axis=0)
  stress = stress - np.mean(stress)
  print('\nlen stress', len(stress))
  print('mean stress (Pa) =', np.mean(stress)*10**5)
  print('mean stress^2 (Pa^2) =', np.mean(stress**2)*10**10)

  thickness = np.absolute(np.mean(height_v.z)) * 10**(-9)
  print('\nthickness =', thickness, "m")

  temp = bigpressure['temp'].mean()
  print("\ntemperature =",temp,"K")

  boxvol = sizes[0]*sizes[1]*sizes[2]
  print("\nboxvol =", boxvol, "m^3")
  print("\ncalculating autocor")


  def autocor(sig):
    return(
      signal.correlate(sig, sig, mode='full')[len(sig)-1:] / 
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
  
  # in the following line, the "sizes[2]" part was removed, because that gets
  # factored in with the 'water_correction()' function.
  visco_total = viscosityfactor / temp * integrated # * sizes[2]

  visco_arr = water_correction(visco_total, sizes[2], thickness, temp) 
  
  print("\nsaving")

  names = np.array([
    'names',
    'times',
    'xy stress autocorrelation',
    'viscosity at each timestep (from 0)',
    ])

  print('\nlen(stress_autocor)', len(stress_autocor))

  if datapoints==0:
    nn = 1
  else:
    nn = max(len(bigpressure)//datapoints,1)

  print(f"saving every {nn} timesteps, which is every {nn*stepsize*10**12} picoseconds")

  np.savez_compressed(
    './py_output.npz',
    names,
    bigpressure['time (s)'][::nn],
    stress_autocor[::nn],
    visco_arr[::nn],
    )

  if args.fft:
    np.savez_compressed(
      './autocor_fft.npz',
      np.fft.fft(stress_autocor)
    )

main()