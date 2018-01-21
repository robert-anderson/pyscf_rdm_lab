import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import fci
import rdm_factory
import tensor_utils

bond_length = 1.1

mol = gto.Mole()
mol.build(
verbose = 0,
output = None,
atom = [
    ['N',(  0.000000,  0.000000, -bond_length*0.5)],
    ['N',(  0.000000,  0.000000,  bond_length*0.5)], ],
basis = {'N': 'ccpvtz', },
symmetry = False,
symmetry_subgroup = 'D2h'
)

myscf = scf.RHF(mol)
myscf.scf()

print 'RHF energy: {}'.format(myscf.e_tot)

nelec = (2,2)
norb = 4

mycasci = mcscf.CASCI(myscf, norb, nelec)
mycasci.kernel()

dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', mycasci.ci, mycasci.ci, norb, nelec)
'''
mydm2 = rdm_factory.make_pose_spinfree_rdm(mycasci.ci, norb, nelec, 2)
print tensor_utils.allclose_transpose(dm2, mydm2)
mydm3 = rdm_factory.make_pose_spinfree_rdm(mycasci.ci, norb, nelec, 3)
print tensor_utils.allclose_transpose(dm3, mydm3)

mydm2 = rdm_factory.make_no_spinfree_rdm(mycasci.ci, norb, nelec, 2)
print tensor_utils.allclose_transpose(dm2, mydm2)
'''
fci.rdm.reorder_dm1234(dm1, dm2, dm3, dm4)
mydm3 = rdm_factory.make_no_spinfree_rdm(mycasci.ci, norb, nelec, 3)

sr_dm3 = make_no_spin_resolved_rdm(mycasci.ci, norbs, nelecs, rank)

print tensor_utils.allclose_transpose(dm3, mydm3)
sf_dm3 = spin_trace_no(my):

print 'CASCI energy: {}'.format(mycasci.e_tot)
