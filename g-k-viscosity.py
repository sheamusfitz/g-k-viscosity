import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from scipy import linalg

print('NOTE: this does not yet account for the water viscosity or the thickness of the membrane *relative* to the box.')

# xvgname = '../nvt_analysis/pressure.xvg'
# heightname = '../nvt_analysis/thickness.xvg'
# struct_filename = '../step7_short.gro'
# print(xvgplace)

xvgname = './pressure-tensor.xvg'
xvgplace = ''.join(xvgname.partition('/')[:-1])
heightname = './thickness.xvg'
struct_filename = './step8_nvt.gro'

with open(struct_filename) as f:
  sizeline = f.readlines()[-1]
print(sizeline)
sizes = sizeline.split()
for i,size in enumerate(sizes):
  sizes[i] = float(size)


bigpressure = pd.read_csv(xvgname,
                          skiprows=33,
                          header=None, delim_whitespace=True)

bigpressure.columns = ['time (ps)', 'temp',
                      'xx', 'xy', 'xz',
                      'yx', 'yy', 'yz',
                      'zx', 'zy', 'zz']

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

height_v = pd.read_csv(heightname,skiprows=17,header=None,delim_whitespace=True)
height_v.columns = ['t','x','y','z']

stepsize = bigpressure['time (ps)'][1] * 10**(-12)

print(stepsize)

#make stress tensor

# actually use 10000
skipping = 80000 #97000
stress = sp.zeros(len(bigpressure)-skipping)
for i in range(len(bigpressure)-skipping):
  stress[i] = sp.mean([bigpressure.xy[skipping+i], bigpressure.yx[skipping+i]])
stressmean = [sp.mean([stress[sp.maximum(0,i-25):i+1]])\
  for i in range(len(stress)-1)]
stressuncertainty = [sp.std([stress[:i+1]])/sp.sqrt(i+1)\
  for i in range(len(stress)-1)]
# print(len(stress))

thickness = sp.absolute(sp.mean(height_v.z[skipping//100:])) * 10**(-9)
thicksem = sp.stats.sem(height_v.z[skipping//100:]) * 10**(-9)
print('thickness ', thickness*10**9)

def autocor(arr, tau):
  if tau>len(arr):
    return('tau set too large')
  aaa = [arr[i]*arr[i+tau] for i in range(len(arr)-tau)]
  if tau%250 == 0:
    print(f'{(tau/len(arr))**(1/2)*100:6.2f}', '%')
  return(sp.mean(aaa),sp.stats.sem(aaa))

#  this all seems wrong to me... i'll fix it here we go:
temp = bigpressure['temp'][skipping:].mean()
tempsem = sp.stats.sem(bigpressure['temp'][skipping:])
print(tempsem)
# print(temp)
boxvol = sizes[0]*sizes[1]*sizes[2] * 10**-27 # volume in m^3

# viscosityfactor = (
#   10**10                    # bar^2 to Pa^2
#   * stepsize
#   * (boxvol)
#   * (1.38064852 * 10**-23 * temp)**(-1) # kb T
#   * thickness
# )

stress_autocor = sp.array([[autocor(stress,tau)[0], autocor(stress,tau)[1]]\
  for tau in range(len(stress)-1)])

integrated = sp.array([sp.sum(stress_autocor[:i+1,0]) for i in range(len(stress_autocor))])
# print(integrated[:10])
# print(len(integrated))

unclist = sp.array([sp.sqrt(sp.sum(stress_autocor[:i,1]**2)) for i in range(1,len(integrated))])
unclist = sp.insert(unclist, 0, unclist[0])
unclist = sp.array(unclist,dtype=float)
# print(sp.amin(unclist**-2))
print('uncertainties')

##################################################

movingavg = sp.zeros_like(stress_autocor[:,0])
windowsize = 25
for i in range(len(stress_autocor)-windowsize):
    movingavg[i] = sp.average(stress_autocor[i:i+windowsize,0])

##################################################

# integrated = [sp.sum(stress_autocor[:i]) for i in range(len(stress_autocor))]

viscosityfactor = (
  10**10                  #bar^2 t Pa^2
  * stepsize
  * boxvol
  * (1.38064852 * 10**-23)**(-1) #1/k_B
)

visco_arr = viscosityfactor / temp * thickness * integrated
visco_uncertainties = visco_arr * sp.sqrt(
  (tempsem/temp)**2 + (thicksem/thickness)**2 + (unclist/integrated)**2
)

def normed(x):
  return(x/sp.linalg.norm(x))

visco_rolling_stats = sp.array([DescrStatsW(visco_arr[:i], weights=normed(visco_uncertainties[:i]**(-2)), ddof=0) for i in range(1,len(integrated))])

visco_rolling = sp.array([stat.mean for stat in visco_rolling_stats])
visco_rolling = sp.insert(visco_rolling, 0, visco_arr[0])
visco_rolling = sp.array(visco_rolling, dtype=float)
print('means')
visco_sem = sp.array([stat.std_mean for stat in visco_rolling_stats[1:]])
visco_sem = sp.insert(visco_sem, 0, visco_sem[0])
visco_sem = sp.insert(visco_sem, 0, visco_sem[0])
visco_sem = sp.array(visco_sem, dtype = float)

#TODO i need to now *save* all of these arrays... duh.
names = sp.array(['names', 'viscosity at each timestep (from 0)', 'viscosity uncertainties', 'rolling weighted average viscosity from 0', 'SEM for rolling average viscosity'])

sp.savez_compressed('./py_output.npz', names, visco_arr, visco_uncertainties, visco_rolling, visco_sem)

def plotter():

  xarr = sp.arange(0,len(stress_autocor))*stepsize

  plt.figure()
  plt.plot(stress,'.k',alpha=0.1)
  # plt.plot(stressuncertainty,'r',alpha=0.5)
  plt.plot(stressmean)
  plt.title('Off-Diagonal Pressure')
  plt.ylabel('$P_{xy}$ (bar)')
  plt.xlabel('timestep')
  plt.savefig(xvgplace+'Pxy.png')
  # plt.show()

  ##################################################

  plt.figure()
  plt.plot(xarr, stress_autocor[:,0],color='#999999',label='autocorrelation')
  plt.plot(xarr, movingavg, label='rolling average over 25 frames (25ps)')
  plt.ylim(-200,200)
  plt.title('$P_{xy}$ autocorrelation')
  plt.ylabel('$(bar^2)$')
  plt.xlabel('$\Delta\,\\tau$ (sec)')
  plt.xlim(-0.2E-10,0.5E-9)
  plt.legend()
  # plt.show()
  plt.savefig(xvgplace+'stress-autocor-zoom.png')

  # plt.plot(xarr, stress_autocor[:,0],color='#999999',label='autocorrelation')
  # plt.plot(xarr, movingavg, label='rolling average over 25 frames (25ps)')
  # plt.plot(xarr, sp.ones_like(xarr))
  plt.ylim(-2000,2000)
  # plt.title('$P_{xy}$ autocorrelation')
  # plt.ylabel('$(bar^2)$')
  # plt.xlabel('$\Delta\,\\tau$ (sec)')
  plt.legend()
  plt.xlim()
  # plt.xlim(-1E-11,21E-11)
  # plt.show()
  plt.savefig(xvgplace+'stress-autocor.png')

  ##################################################

  plt.figure()
  # imean = sp.mean(integrated)
  # istd = sp.std(integrated)
  #TODO start editing here
  # plt.plot(xarr, integrated * viscosityfactor, '-', c='C0', alpha = 0.2,label='viscosity')
  # plt.plot(xarr, meanlist * viscosityfactor, '-', c='C1', alpha = 0.2,
  #   label='average viscosity from $\Delta\\tau$=0')
  # plt.fill_between(xarr, (integrated+unclist) * viscosityfactor,
  #   (integrated-unclist) * viscosityfactor, alpha = 0.2, color='C0')
  # plt.fill_between(xarr, (meanlist+stdlist) * viscosityfactor,
  #   (meanlist-stdlist) * viscosityfactor, alpha = 0.2, color='C1')

  plt.plot(xarr, visco_arr, '-', c='C0')
  plt.fill_between(xarr, visco_arr+visco_uncertainties, visco_arr-visco_uncertainties, color='C0', alpha=0.2)

  plt.plot(xarr, visco_rolling, '-', c='C1')
  plt.fill_between(xarr, visco_rolling+visco_sem, visco_rolling-visco_sem, color='C1', alpha=0.2)

  plt.plot([0,2E-8],sp.array([1,1])*19.68E-11,'--',c='black',
    label='previous measurement')
  plt.fill_between([0,2E-8],
                  sp.array([1,1])*(19.68-0.69)*1E-11,
                  sp.array([1,1])*(19.68+0.69)*1E-11,
                  alpha = 0.1, color='black')

  plt.title('Green Kubo')
  plt.ylabel('Viscosity $(Pa\cdot m\cdot s)$')
  plt.xlabel('$\Delta\\tau$ (sec)')
  plt.xscale('log')
  plt.xlim(xarr[1]/1.01,xarr[-1]*1.1)
  plt.ylim(0,0.5e-9)

  plt.legend(loc=0)
  plt.savefig(xvgplace+'visco-integral.png')
  plt.show()



# plotter()
