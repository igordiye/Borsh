import pyscf
from pyscf import gto
mol = gto.M(atom="N 0 0 0; N 0 0 1.2", basis='cc-pvdz')

# Ground state energy
ci = mol.apply("CISD").run()
ks = mol.apply("RKS").run()
# Excited state energy
ci.nstates = 5
ci.run()
from pyscf import tdscf
td = tdscf.TDRKS(ks)
td.nstates = 5
td.run()


# Nuclear gradients for excited states
# ground state nuclear gradients
force = ci.nuc_grad_method().kernel()
force = ks.nuc_grad_method().kernel()
# excited states nuclear gradients
ci.nstates = 5
force = ci.nuc_grad_method().kernel(state=1)
td = ks.apply("TDRKS")
td.nstates = 5
#force = td.nuc_grad_method().kernel(state=1) # this does not work rn


#Energy and gradients for multiple geometries
mol = gto.M(atom="N; N 1 1.2", basis="ccpvdz")
# ci_scan = mol.apply("CASCI", 8, 10).as_scanner()
# e, force = ci_scan("N; N 1 1.3")
# e, force = ci_scan("N; N 1 1.4")
# grad_scan = mc.nuc_grad_method().as_scanner()
# e, force = grad_scan("N; N 1 1.3")
# e, force = grad_scan("N; N 1 1.4")
# The 4th excited state
# grad_scan = mc.nuc_grad_method().as_scanner(state=4)
# e, force = grad_scan("N; N 1 1.3")
# e, force = grad_scan("N; N 1 1.4")


#Transition density matrix
mol = gto.M(atom="N; N 1 1.2", basis="ccpvdz")
ci = mol.apply("CISD")
ci.nstates = 4
ci.run()
t_dm1 = myci.trans_rdm1(ci.ci[3], ci.ci[0])
#Transition dipole
td = mol.apply("RKS").apply("TDA")
td.nstates = 5
td.run()
t_dip = td.transition_dipole()

#One-electron and two-electron integrals
mol = gto.M(atom="N; N 1 1.2", basis="ccpvdz")
hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
eri = mol.intor("int2e")
with mol.with_common_origin([0,0,0.6]):
    dip = mol.intor("int1e_r")
#Integrals with periodic boundary condition
from pyscf.pbc import gto, df
cell = gto.M(...)
hcore_kpt = (cell.pbc_intor("int1e_kin", kpt=kpt)
+ df.FFTDF(cell).get_pp(kpt))
eri_kpt = df.FFTDF(cell).get_eri([kpt1,kpt2,kpt3,kpt4])
