import pickle, copy

from subprocess import Popen

import pprint, shutil
import numpy, os, sys
from pyscf import gto, scf, mcscf
from pyscf.mrpt import nevpt2


mol = gto.Mole()
mol.build(
    verbose = 0,
    output = None,
    atom = [
            ['N',(  0.000000,  0.000000, -0.54875)],
            ['N',(  0.000000,  0.000000,  0.54875)], ],
    basis = {'N': 'ccpvtz', },
    symmetry = True,
    symmetry_subgroup = 'C1'
)

myhf = scf.RHF(mol)
myhf.conv_tol = 1e-10
myhf.kernel()

myhf.analyze()

print myhf.e_tot

mycas = mcscf.CASCI(myhf, 6, 6)
mycas.kernel()
numpy.save('ci.npy', mycas.ci)

e = nevpt2.NEVPT(mycas).kernel()
print 'NEVPT2 energy {}'.format(e)


