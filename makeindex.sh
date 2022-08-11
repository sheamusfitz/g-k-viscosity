#!/bin/bash
for rn in r*/; do
        echo -e 'name 2 MEMB\nname 3 SOL_ION\n\nq' | gmx_mpi \
        make_ndx -f $rn/step6.6_equilibration.gro -o \
        $rn/index.ndx
        gmx grompp -f step7_boxsize.mdp \
        -o $rn/step7_boxsize.tpr \
        -c $rn/step6.6_equilibration.gro \
        -p $rn/topol.top -n $rn/index.ndx -maxwarn 1
done
