import numpy as np
import tensor_utils

h2e = np.load('h2e.npy')
dm4 = np.load('dm4.npy')
f3ac = np.load('f3ac.npy')


#f3ac_py_contract = np.einsum('pqrb,kiqjprac->ijkabc', h2e, dm4)
f3ac_py_contract = np.einsum('pqra,kibjqcpr->ijkabc', h2e, dm4)
print tensor_utils.allclose_transpose(f3ac.transpose(1,3,0,4,2,5), f3ac_py_contract, t_quit_on_first=False)
print f3ac.shape
print f3ac_py_contract.shape
