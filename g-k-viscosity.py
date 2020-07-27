import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from scipy import linalg

xvgname = 'nvt_analysis/pressure.xvg'
xvgplace = ''.join(xvgname.partition('/')[:-1])
print(xvgplace)
heightname = 'nvt_analysis/thickness.xvg'
struct_filename = 'step7_short.gro'

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

skipping = 90000 # 10000
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


# stepsize = 2*10**-12 #in seconds          # this probably needs to change
# # bigpressure['time (fs)'][1] - bigpressure['time (fs)'][0]
# boxvol = 1102.7 #in nm^3                  # this needs to change
# viscosityfactor = (                       # this needs to change             
#     (stepsize * 10**10)                   # this needs to change
#   * (boxvol * (10**-9)**3)                # this needs to change
#   * (1.38064852 * 10**-23 * 303)**(-1)    # this needs to change
#   * (4.2*10**(-9))                        # wtf is this
# )

#  this all seems wrong to me... i'll fix it here we go:
temp = bigpressure['temp'].mean()
tempsem = sp.stats.sem(bigpressure['temp'])
print(tempsem)
# print(temp)
boxvol = sizes[0]*sizes[1]*sizes[2] * 10**-27 # volume in m^3

viscosityfactor = (
  10**10                    # bar^2 to Pa^2
  * stepsize
  * (boxvol)    
  * (1.38064852 * 10**-23 * temp)**(-1) # kb T
  * thickness
)

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

weights = unclist**-2 / sp.linalg.norm(unclist**-2)
# print(weights)

meanlist_stats = sp.array([
  DescrStatsW(integrated[:i], weights=weights[:i], ddof = 0)\
     for i in range(1,len(integrated))
])

# meanlist = [sp.average(integrated[:i],weights=unclist[:i]**-2) for i in range(1,len(integrated))]
meanlist = sp.array([stat.mean for stat in meanlist_stats])
meanlist = sp.insert(meanlist, 0, integrated[0])
meanlist[1] = integrated[1]
meanlist = sp.array(meanlist,dtype=float)
# print('first means', meanlist[:10])
# print(meanlist)

print('means')
# First attempt
# stdlist = [sp.std(integrated[:i]) for i in range(1,len(integrated))]

# Second attempt
# stdlist = [sp.sum(unclist[:i]**-2)**(-1/2) for i in range(1,len(integrated))]
# stdlist.append(stdlist[-1])
# stdlist = sp.array(stdlist)
# stdlist[0] = stdlist[1]

# Third Attempt
stdlist = sp.array([stat.std_mean for stat in meanlist_stats[1:]])
# stdlist[0] = integrated[0]
stdlist = sp.insert(stdlist, 0, stdlist[0])
stdlist = sp.insert(stdlist, 0, stdlist[0])
stdlist = sp.array(stdlist,dtype=float)
# print(stdlist[:10])
# print('len',len(stdlist))
print('standarddev')

##################################################

movingavg = sp.zeros_like(stress_autocor[:,0])
windowsize = 25
for i in range(len(stress_autocor)-windowsize):
    movingavg[i] = sp.average(stress_autocor[i:i+windowsize,0])

##################################################

# integrated = [sp.sum(stress_autocor[:i]) for i in range(len(stress_autocor))]

visco2av = sp.average(meanlist,weights=stdlist**-2)
# unc2av = sp.sqrt(sp.sum(stdlist**2))
unc2av = sp.sum(stdlist**-2)**(-1/2)

viscoraw = sp.average(integrated,weights=unclist**-2)
# uncraw = sp.sqrt(sp.sum(unclist**2))
uncraw = sp.sum(unclist**-2)**(-1/2)

# print(str(visco2av *viscosityfactor)+'   '+str(unc2av *viscosityfactor))
# print(str(viscoraw *viscosityfactor)+'   '+str(uncraw *viscosityfactor))

names = sp.array([''])

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
  integrated = sp.array(integrated)
  plt.plot(xarr, integrated * viscosityfactor, '-', c='C0', alpha = 0.2,label='viscosity')
  print(len(xarr), len(meanlist))
  plt.plot(xarr, meanlist * viscosityfactor, '-', c='C1', alpha = 0.2,
    label='average viscosity from $\Delta\\tau$=0')
  plt.fill_between(xarr, (integrated+unclist) * viscosityfactor,
    (integrated-unclist) * viscosityfactor, alpha = 0.2, color='C0')
  plt.fill_between(xarr, (meanlist+stdlist) * viscosityfactor,
    (meanlist-stdlist) * viscosityfactor, alpha = 0.2, color='C1')
  plt.plot([0,2E-8],sp.array([1,1])*19.68E-11,'--',c='black',
    label='previous measurement')
  plt.fill_between([0,2E-8],
                  sp.array([1,1])*(19.68-0.69)*1E-11,
                  sp.array([1,1])*(19.68+0.69)*1E-11,
                  alpha = 0.1, color='black')


  # mmm = 40000
  # plt.ylim(-mmm,mmm)

  plt.title('Green Kubo')
  plt.ylabel('Viscosity $(Pa\cdot m\cdot s)$')
  plt.xlabel('$\Delta\\tau$ (sec)')
  plt.xscale('log')
  plt.xlim(xarr[1]/1.01,xarr[-1]*1.1)
  plt.ylim(0,0.5e-9)
  # plt.show()

  #this didn't work too well
  # where = sp.argmin(stdlist[1:])
  # plt.axvline(x=xarr[where])

  # plt.axhline(y=visco2av*viscosityfactor,linestyle='--',color='C2',label = 'viscosity from rolling average')
  # plt.axhline(y=(visco2av-unc2av)*viscosityfactor,color='C2',alpha=0.5)
  # plt.axhline(y=(visco2av+unc2av)*viscosityfactor,color='C2',alpha=0.5)

  # plt.axhline(y=viscoraw*viscosityfactor,linestyle='--',color='C3',label = 'viscosity from raw')
  # plt.axhline(y=(viscoraw-uncraw)*viscosityfactor,color='C3',alpha=0.5)
  # plt.axhline(y=(viscoraw+uncraw)*viscosityfactor,color='C3',alpha=0.5)
  # print(viscomaybe*viscosityfactor)

  plt.legend(loc=0)
  plt.savefig(xvgplace+'visco-integral.png')
  # plt.show()



# plotter()