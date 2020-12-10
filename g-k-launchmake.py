pwd = '/scratch/07324/sheafitz/wdepths/'
lipids = ['dmpc-8nm', 'dppc-12nm', 'dppc-15nm']


with open('./manylaunch.bat', 'w') as f:

  #TODO okay shea figure out what you're doing here this is a mess.

  f.write('\
#!/bin/bash\n\
#SBATCH -J vis-all-parr         # Name of job; user chosen\n\
#SBATCH -p skx-normal           # queue: always use skx-normal\n\
#SBATCH -N 2                    # number of nodes\n\
#SBATCH -n 80\n\
#SBATCH -t 02:00:00\n\
#SBATCH --mail-user=sheafitz@udel.edu\n\
#SBATCH --mail-type=all\n\n\
module load intel/18.0.2\n\
module load impi/18.0.2\n\
module load python3\n\n')

  for l, lipid in enumerate(lipids):
    for i in range(1,6):
      f.write(
        'cd ' + pwd + lipid + '/r'+str(i)+'\n'
        'ibrun -n 2 -o ' + str((5*l+i-1)*2) + ' python3 ../../g-k-viscosity.py &\n'
        )
  f.write('wait')
