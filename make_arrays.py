import pickle, copy

from subprocess import Popen

import pprint, shutil
import numpy, os, sys
import pyscf
from pyscf import gto, scf, mcscf
from pyscf.mrpt import nevpt2
from pyscf.future import fciqmcscf
import neci_inps

ncas = 4
nelecas = (2,2)

if __name__=='__main__':

    mol = gto.Mole()
    mol.build(
        verbose = 0,
        output = None,
        atom = [
                #['N',(  0.000000,  0.000000, -0.54875)],
                #['N',(  0.000000,  0.000000,  0.54875)], ],
                ['Be',(  0.000000,  0.000000,  0.00000)], ],
        #basis = {'N': 'ccpvdz', },
        basis = {'Be': 'ccpvdz', },
        symmetry = True,
        symmetry_subgroup = 'C1'
    )

    myhf = scf.RHF(mol)
    myhf.conv_tol = 1e-10
    myhf.kernel()

    myhf.analyze()

    with open('neci.inp', 'w') as f:
        f.write(neci_inps.casscf_non_rel({'nelecas':sum(nelecas)}))

    print myhf.e_tot

    mycas = mcscf.CASCI(myhf, ncas, nelecas)
    mycas.kernel()


    mo_core, mo_cas, mo_virt = nevpt2._extract_orbs(mycas, mycas.mo_coeff)
    h1e = mycas.h1e_for_cas()[0]
    h2e = pyscf.ao2mo.restore(1, mycas.ao2mo(mo_cas), mycas.ncas).transpose(0,2,1,3)

    numpy.save('h1e.npy', h1e)
    numpy.save('h2e.npy', h2e)

    print 'casci energy: {}'.format(mycas.e_tot)
    cas_fock = mycas.get_fock()[mycas.ncore:mycas.ncore+mycas.ncas, mycas.ncore:mycas.ncore+mycas.ncas]
    #cas_fock = numpy.ones((mycas.ncas, mycas.ncas))
    numpy.save('cas_fock.npy', cas_fock)
    numpy.save('ci.npy', mycas.ci)

    fock_header = \
'''  
&FCI NORB=   6,NELEC= 6,MS2=0,
 ORBSYM=1,1,1,1,1,1
 ISYM=1,
&END
'''
    fock_header = \
'''  
&FCI 
&END
'''
    with open('FOCKMAT', 'w') as f:
        f.write(fock_header)
        for i in range(ncas):
            for j in range(ncas):
                if abs(cas_fock[i,j])>1e-12:
                    f.write('{}   {}   {}\n'.format(cas_fock[i,j], i+1, j+1))

    e = nevpt2.NEVPT(mycas).kernel()
    print 'NEVPT2 energy {}'.format(e)

    mycas = mcscf.CASCI(myhf, ncas, nelecas)
    fciqmc_obj = fciqmcscf.FCIQMCCI(mol)
    fciqmc_obj.only_ints = True
    mycas.fcisolver = fciqmc_obj
    mycas.kernel()


