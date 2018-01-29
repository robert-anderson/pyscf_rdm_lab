from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from pyscf.fci import rdm as pyscf_rdm
import numpy as np
import itertools
import tensor_utils

def explicit_parity(tup):
  # lazy implementation to test faster ones
  work = list(tup)
  t = 0
  while True:
    for i in range(len(tup)):
      if work[i]!=i:
        try:
          if work[i]>work[i+1]:
            work[i], work[i+1] = work[i+1], work[i]
            t+=1
        except IndexError:
          work[i], work[0] = work[0], work[i]
          t+=len(tup)-1
    if work==range(len(tup)):
      break
  return 1-2*(t%2)

def make_all_perms(rank):
	all_perms = []
	for order in itertools.permutations(range(rank), rank):
		all_perms.append([order, explicit_parity(order)])
	return all_perms

def apply_spin_resolved_rdm_symmetries(rdm):
	# iterate over upper triangle
  nbasis = rdm.shape[0]
  rank = len(rdm.shape)/2
  all_perms = make_all_perms(rank)
  for hole_inds in itertools.combinations(range(nbasis), rank):
    for elec_inds in itertools.combinations(range(nbasis), rank):
      for hole_perm in all_perms:
        for elec_perm in all_perms:
          rdm[tuple(hole_inds[i] for i in hole_perm[0])+tuple(elec_inds[i] for i in elec_perm[0])] = \
          rdm[hole_inds+elec_inds]*hole_perm[1]*elec_perm[1]

def spin_trace(sr_rdm):
  nbasis = sr_rdm.shape[0]
  rank = len(sr_rdm.shape)/2
  sf_rdm = np.zeros((nbasis/2,)*rank*2)
  for inds in itertools.product(range(nbasis/2), repeat=rank*2):
    print "tracing", inds
    for spin_sig in itertools.product((0,1), repeat=rank):
      sf_rdm[inds]+=sr_rdm[tuple(2*inds[i]+spin_sig[i] for i in range(rank))+tuple(2*inds[rank+i]+spin_sig[i] for i in range(rank))]
  return sf_rdm




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
	# only fill upper triangle
  for hole_inds in itertools.combinations(range(2*norbs), rank):
    spat_hole_inds, hole_alphas = split_spin_inds(hole_inds)
    for elec_inds in itertools.combinations(range(2*norbs), rank):
      spat_elec_inds, elec_alphas = split_spin_inds(elec_inds)
      if sum(elec_alphas)!=sum(hole_alphas):
        continue
      rdm[hole_inds+elec_inds] = calc_no_spin_resolved_element(civec, norbs, nelecs,
          spat_hole_inds, spat_elec_inds, hole_alphas, elec_alphas)
  apply_spin_resolved_rdm_symmetries(rdm)
  return rdm

def exact_caspt2_intermediate(complete_no_sf_rdm4, diag_cas_fock):
  norbs = complete_no_sf_rdm4.shape[0]
  intermediate = np.zeros((norbs,)*6)
  for p in range(norbs):
    for first_3_inds in itertools.product(range(norbs), repeat=3):
      for last_3_inds in itertools.product(range(norbs), repeat=3):
        intermediate[first_3_inds+last_3_inds]+=diag_cas_fock[p]*complete_no_sf_rdm4[first_3_inds+(p,)+last_3_inds+(p,)]
  return intermediate

top_rdm = 2

if __name__=='__main__':
  civec=np.load('ci.npy')
  if top_rdm>0:
    no_sr_rdm1=make_no_spin_resolved_rdm(civec, 6, (3,3), 1)
    np.save('no_sr_rdm1.npy', no_sr_rdm1)
    no_sf_rdm1=spin_trace(no_sr_rdm1)
    np.save('no_sf_rdm1.npy', no_sf_rdm1)
  if top_rdm>1:
    no_sr_rdm2=make_no_spin_resolved_rdm(civec, 6, (3,3), 2)
    np.save('no_sr_rdm2.npy', no_sr_rdm2)
    no_sf_rdm2=spin_trace(no_sr_rdm2)
    np.save('no_sf_rdm2.npy', no_sf_rdm2)
  if top_rdm>2:
    no_sr_rdm3=make_no_spin_resolved_rdm(civec, 6, (3,3), 3)
    np.save('no_sr_rdm3.npy', no_sr_rdm3)
    no_sf_rdm3=spin_trace(no_sr_rdm3)
    np.save('no_sf_rdm3.npy', no_sf_rdm3)
  if top_rdm>3:
    no_sr_rdm4=make_no_spin_resolved_rdm(civec, 6, (3,3), 4)
    np.save('no_sr_rdm4.npy', no_sr_rdm4)
    no_sf_rdm4=spin_trace(no_sr_rdm4)
    np.save('no_sf_rdm4.npy', no_sf_rdm4)

  pyscf_pose_sf_rdm1, pyscf_pose_sf_rdm2, pyscf_pose_sf_rdm3, pyscf_pose_sf_rdm4 = pyscf_rdm.make_dm1234('FCI4pdm_kern_sf', civec, civec, 6, (3,3))
  pyscf_no_sf_rdm1, pyscf_no_sf_rdm2, pyscf_no_sf_rdm3, pyscf_no_sf_rdm4 = pyscf_rdm.reorder_dm1234(pyscf_pose_sf_rdm1, pyscf_pose_sf_rdm2, pyscf_pose_sf_rdm3, pyscf_pose_sf_rdm4)
  np.save('pyscf_no_sf_rdm1.npy', pyscf_no_sf_rdm1)
  np.save('pyscf_no_sf_rdm2.npy', pyscf_no_sf_rdm2)
  np.save('pyscf_pose_sf_rdm2.npy', pyscf_pose_sf_rdm2)


  assert(0)

  print tensor_utils.allclose_transpose(pyscf_no_sf_rdm1, no_sf_rdm1)
  print tensor_utils.allclose_transpose(pyscf_no_sf_rdm2, no_sf_rdm2)
  print tensor_utils.allclose_transpose(pyscf_no_sf_rdm3, no_sf_rdm3)
  print tensor_utils.allclose_transpose(pyscf_no_sf_rdm4, no_sf_rdm4)


#exact_D = exact_caspt2_intermediate(complete_no_sf_rdm4, diag_cas_fock)


#print rdm2[0,0,0,0]
#print rdm2[0,1,0,1]



#rdm=make_pose_spinfree_rdm(civec, 6, (3,3), 2)
#dm2 = np.load('dm2.npy')
#print np.allclose(rdm, dm2)
#print tensor_utils.allclose_transpose(rdm, dm2)







