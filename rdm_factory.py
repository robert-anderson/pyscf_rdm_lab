from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
import numpy as np
import itertools
import tensor_utils

'''
cre_a(ci0, norb, neleca_nelecb, ap_id)
'''
def zip_tup(tup1, tup2):
    return tuple(val for pair in zip(tup1, tup2) for val in pair)

def compute_inner_product(civec, norbs, nelecs, ops, cres, alphas):
    neleca, nelecb = nelecs
    ciket = civec.copy()
    assert(len(ops)==len(cres))
    assert(len(ops)==len(alphas))
    for i in reversed(range(len(ops))):
        if alphas[i]:
            if cres[i]:
                ciket = cre_a(ciket, norbs, (neleca, nelecb), ops[i])
                neleca += 1
            else:
                if neleca==0:
                    return 0
                ciket = des_a(ciket, norbs, (neleca, nelecb), ops[i])
                neleca -= 1
        else:
            if cres[i]:
                ciket = cre_b(ciket, norbs, (neleca, nelecb), ops[i])
                nelecb += 1
            else:
                if nelecb==0:
                    return 0
                ciket = des_b(ciket, norbs, (neleca, nelecb), ops[i])
                nelecb -= 1
    return np.dot(civec.flatten(), ciket.flatten())


def calc_no_spin_resolved_element(civec, norbs, nelecs, hole_orbs, elec_orbs, hole_alphas, elec_alphas):
    assert(len(hole_orbs)==len(elec_orbs))
    return compute_inner_product(civec, norbs, nelecs, hole_orbs+tuple(reversed(elec_orbs)),
            (True,)*len(hole_orbs)+(False,)*len(hole_orbs), hole_alphas+tuple(reversed(elec_alphas)))

def calc_pose_spin_resolved_element(civec, norbs, nelecs, hole_orbs, elec_orbs, hole_alphas, elec_alphas):
    assert(len(hole_orbs)==len(elec_orbs))
    return compute_inner_product(civec, norbs, nelecs, zip_tup(hole_orbs, elec_orbs),
            (True,False)*len(hole_orbs), zip_tup(hole_alphas, elec_alphas))

def calc_no_spinfree_element(civec, norbs, nelecs, hole_orbs, elec_orbs):
    assert(len(hole_orbs)==len(elec_orbs))
    tot = 0
    rank = len(hole_orbs)
    for spin_sig in itertools.product((True, False), repeat=rank):
        tot+=calc_no_spin_resolved_element(civec, norbs, nelecs, hole_orbs,
                elec_orbs, spin_sig, spin_sig)
    return tot

def calc_pose_spinfree_element(civec, norbs, nelecs, hole_orbs, elec_orbs):
    assert(len(hole_orbs)==len(elec_orbs))
    tot = 0
    rank = len(hole_orbs)
    for spin_sig in itertools.product((True, False), repeat=rank):
        tot+=calc_pose_spin_resolved_element(civec, norbs, nelecs, hole_orbs,
                elec_orbs, spin_sig, spin_sig)
    return tot

def make_pose_spinfree_rdm(civec, norbs, nelecs, rank):
    rdm = np.zeros((norbs,)*(2*rank))
    for inds in itertools.product(range(norbs), repeat=2*rank):
        rdm[inds] = calc_pose_spinfree_element(civec, norbs, nelecs,
                tuple(inds[2*i] for i in range(rank)), 
                tuple(inds[2*i+1] for i in range(rank)))
    return rdm

def make_no_spinfree_rdm(civec, norbs, nelecs, rank):
    rdm = np.zeros((norbs,)*(2*rank))
    for inds in itertools.product(range(norbs), repeat=2*rank):
        rdm[inds] = calc_no_spinfree_element(civec, norbs, nelecs,
                tuple(inds[2*i] for i in range(rank)), 
                tuple(inds[2*i+1] for i in range(rank)))
    return rdm

def split_spin_inds(inds):
    return tuple(i/2 for i in inds), tuple(i%2==0 for i in inds)

def make_no_spin_resolved_rdm(civec, norbs, nelecs, rank):
    rdm = np.zeros((2*norbs,)*(2*rank))
    for inds in itertools.product(range(2*norbs), repeat=2*rank):
        spat_inds, alphas = split_spin_inds(inds)
        rdm[inds] = calc_no_spin_resolved_element(civec, norbs, nelecs,
                tuple(spat_inds[2*i] for i in range(rank)),
                tuple(spat_inds[2*i+1] for i in range(rank)),
                tuple(alphas[2*i] for i in range(rank)),
                tuple(alphas[2*i+1] for i in range(rank)))
    return rdm

def spin_trace_no(rdm):
    rank = len(rdm.shape)/2
    norbs = rdm.shape[0]/2
    sf_rdm = np.zeros((norbs,)*(2*rank))
    for inds in itertools.product(range(2*norbs), repeat=2*rank):
        sf_rdm[split_spin_inds(inds)[0]]+=rdm[inds]
    return sf_inds


civec=np.load('ci.npy')
rdm=make_pose_spinfree_rdm(civec, 6, (3,3), 2)
dm2 = np.load('dm2.npy')
print np.allclose(rdm, dm2)
print tensor_utils.allclose_transpose(rdm, dm2)

