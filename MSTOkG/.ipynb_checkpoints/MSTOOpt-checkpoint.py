
import numpy as np
import pylab as pl
import dill as pickle
import os
import pyscf
from pyscf import gto, dft, scf, ci
import basis_set_exchange
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

from prettytable import PrettyTable, HRuleStyle
import pprint
import time

def getcoeffs(mol):
    '''
    Takes in a `pyscf.gto.mole.Mole` object and returns the element and shell with its basis set.

    :param mol: A `pyscf.gto.mole.Mole` molecule
    :type mol: `pyscf.gto.mole.Mole`

    :return: A tuple containing (1) elements and their shell identifiers (2) numpy array containing the STO-kG basis set
    :rtype: tuple

    :Example:
    
    >>> # choose your element
    >>> element = 'H'
    >>> # get the default molecule in your basis
    >>> mol = moldict('sto-6g')[element]
    >>> getcoeffs(mol)
    ([['H', 0]],
    array([[[3.55232212e+01, 9.16359628e-03],
         [6.51314372e+00, 4.93614929e-02],
         [1.82214290e+00, 1.68538305e-01],
         [6.25955266e-01, 3.70562800e-01],
         [2.43076747e-01, 4.16491530e-01],
         [1.00112428e-01, 1.30334084e-01]]]))
         
    '''
    basis = mol.basis
    b = []
    iden = []
    for element in basis.keys():
        for row in range(len(basis[element])):
            shell = basis[element][row][0]
            coeffs = basis[element][row][1:]
            # print(element, shell, coeffs)
            iden.append([element, shell])
            b.append(coeffs)
            # print (iden, " : ", b)
            # print()
    b = np.array(b)
    return iden, b
##################################################

def putcoeffs(iden, b):
    '''Creates a new basis set for a `pyscf.gto.mole.Mole` object from the element/shell identifiers and the array of basis coefficients. These need not be created by hand (you can if you want to and know what you are doing). This function was created to modify the basis of the molecule object and then inject the modified basis back into the molecule. The best way to use it would be to extract the identifier and basis set array using :func:`MSTOkG.MSTOOpt.getcoeffs`, modify the basis appropriately and then use this function to generate the new basis. This new basis should be used to replace the basis in the `pyscf.gto.mole.Mole` object (and subsequently build it).

    :param iden: A list of element name and shell identifiers
    :type iden: list
    :param b: Basis set coefficients.
    :type b: `numpy.array`

    :returns: Basis set for `pyscf.gto.mole.Mole` object
    :rtype: numpy.ndarray

    :Example:
    
    >>> element = 'LiH' # choose your element
    >>> mol = moldict('sto-2g')[element] # retrieve the molecule with standard parameters in the sto-2g basis
    >>> iden, basis = getcoeffs(mol) # extract the identifiers and the basis set coefficients
    >>> iden
    [['H', 0], ['Li', 0], ['Li', 0], ['Li', 1]]
    >>> basis
    array([[[1.30975638, 0.4301285 ],
        [0.23313597, 0.67891353]],
       [[6.16384503, 0.4301285 ],
        [1.09716131, 0.67891353]],
       [[0.24591632, 0.04947177],
        [0.06237087, 0.96378241]],
       [[0.24591632, 0.51154071],
        [0.06237087, 0.6128199 ]]])
    >>> basis[0][1][0] # changing this element to 0.222222
    np.float64(0.2331359749)
    >>> new_basis = basis[:] # creating a copy
    >>> new_basis[0][1][0] = 0.222222 # modifying the copy
    >>> new_basis # let's check
    array([[[1.30975638, 0.4301285 ],
        [0.222222  , 0.67891353]],
       [[6.16384503, 0.4301285 ],
        [1.09716131, 0.67891353]],
       [[0.24591632, 0.04947177],
        [0.06237087, 0.96378241]],
       [[0.24591632, 0.51154071],
        [0.06237087, 0.6128199 ]]])
    >>> new_basis_set = putcoeffs(iden, basis) # create the basis from the new coefficients
    >>> from copy import deepcopy # we need this to create a copy of the molecule
    >>> mol1 = deepcopy(mol) # `mol1` is the new copy
    >>> mol1.basis = new_basis # modify the basis of the new molecule
    >>> mol1.basis # now build this molecule and proceed with your computations
    array([[[1.30975638, 0.4301285 ],
        [0.222222  , 0.67891353]],
       [[6.16384503, 0.4301285 ],
        [1.09716131, 0.67891353]],
       [[0.24591632, 0.04947177],
        [0.06237087, 0.96378241]],
       [[0.24591632, 0.51154071],
        [0.06237087, 0.6128199 ]]])
        
    '''
    newbasis = defaultdict(list)
    for i in range(len(iden)):
        # print(iden[i], b[i, :])
        newbasis[iden[i][0]].append( [iden[i][1], *b[i,:]] )
    return dict(newbasis)
##################################################

def getEnergies(mol, nroots=4, fci=False, cisd=True):
    '''Retrieve the Hartree-Fock, FCI and CISD energies for a molecule.

    :param mol: The molecule whose energies are to be determined
    :type mol: :obj:`pyscf.gto.mole.Mole`
    :param nroots: The number of roots to be retrieved by the FCI solver
    :type nroots: integer, (optional), defaults to 4
    :returns: Computed energies and molecule objects
    :rtype: dict
    
    :Example:
    
    >>> element = 'LiH' # choose your element
    >>> mol = moldict('sto-2g')[element] # get the default molecule in your basis
    >>> getEnergies(mol)
    {'HF': np.float64(-7.609810463500133),
    'FCI': array([-7.62907093, -7.51271216, -7.49826914, -7.46853078]),
    'CISD': np.float64(-7.629059601320526),
    'mol_HF': <pyscf.scf.hf.RHF at 0x75ace8e40c20>,
    'mol_FCI': <pyscf.fci.FCI.<locals>.CISolver at 0x75ace8e41010>,
    'mol_cisd': <pyscf.ci.cisd.RCISD at 0x75ace8e40d70>}
    
    '''
    
    mol.build()
    #########################
    # compute HF energy
    #########################
    mf = scf.RHF(mol)
    mf.max_cycle=5000
    mf.verbose=0
    myhf = mf.run()
    ans = {"HF": myhf.e_tot}
    ans['mol_HF'] = myhf
    #########################

    if fci:
        #########################
        # compute FCI energy
        #########################
        cisolver = pyscf.fci.FCI(myhf)
        cisolver.nroots = nroots
        fcievals, _ = cisolver.kernel()
        ans['FCI'] = fcievals
        ans['mol_FCI'] = cisolver
        #########################

    if cisd:
        #########################
        # compute CISD energy
        #########################
        # get cisd not ccsd
        mc = ci.CISD(myhf)
        mc.verbose = 0
        mc.nroots=nroots
        mc.max_cycle = 5000
        myci = mc.run()
        ans['CISD'] = myci.e_tot[0]
        ans['mol_CISD'] = myci
        #########################

    return ans
##################################################

def BasisReshape(mol, k):
    '''Reshapes the STO-nG basis of the molecule to cardinal number k.

    :param mol: the `pyscf` molecule object
    :type mol: :obj:`pyscf.gto.mole.Mole`
    :param k: Cardinal number for which the basis is to be reshaped.
    :type k: :obj:`int`
    :returns: the `pyscf` molecule object with reshaped basis
    :rtype:  :obj:`pyscf.gto.mole.Mole`
    
    :Example:
    
    >>> element = 'H' # choose your element
    >>> mol = moldict('sto-6g')[element] # get the default molecule in your basis
    >>> mol.basis
    {'H': [[0,
       [35.52322122, 0.00916359628],
       [6.513143725, 0.04936149294],
       [1.822142904, 0.1685383049],
       [0.625955266, 0.3705627997],
       [0.243076747, 0.4164915298],
       [0.100112428, 0.1303340841]]]}
    >>> mol1 = BasisReshape(mol, 7)
    >>> mol1.basis # The new coefficients are appended as [1.0, 1.0] for the deficit cardinal numbers
    {'H': [[0,
       array([3.55232212e+01, 9.16359628e-03]),
       array([6.51314372, 0.04936149]),
       array([1.8221429, 0.1685383]),
       array([0.62595527, 0.3705628 ]),
       array([0.24307675, 0.41649153]),
       array([0.10011243, 0.13033408]),
       array([1., 1.])]]}
    >>> mol2 = BasisReshape(mol1, 5)
    >>> mol2.basis # The excess coefficients are removed.
    {'H': [[0,
       array([3.55232212e+01, 9.16359628e-03]),
       array([6.51314372, 0.04936149]),
       array([1.8221429, 0.1685383]),
       array([0.62595527, 0.3705628 ]),
       array([0.24307675, 0.41649153])]]}

    This molecule object can now be used to optimize the basis set or compute the energies.

    '''
    iden,b = getcoeffs(mol)
    b = list(b)

    for j in range(len(b)):
        b[j] = list(b[j])
        if len(b[j])>k:
            b[j] = b[j][:k]
        else:
            while len(b[j])<k:
                b[j] = b[j]+ [[1,1] ]
                # b[j] = b[j]+ b[j][-1]
    b = np.array(b)
    mol.basis = putcoeffs(iden, b)
    mol.build();
    return mol


def dumpBSI(BSI_instance, fname="BS_EO_pickled"):
    '''An instance of the computed basis sets can be dumped as a pickle file. It will be useful if one wishes to reuse these values for further improvement.

    :param BSI_instance: The python object that will be pickled for easy retrieval later. This is used internally as the :obj:`getEnergies` object for the optimized basis set. The "BSI" stands for basis set improved.
    :type BSI_instance: Prefereably :obj:`getEnergies`. (*Note:* It will not raise an error otherwise.)
    
    :param fname: The filename where the pickled object will be dumped.
    :type fname: :obj:`str`

    :returns: Nothing
    :rtype: None
    '''
    f = open(fname, 'wb')
    pickle.dump(BSI_instance, f)
    f.flush()
    f.close()

def loadBSI(fname="BS_EO_pickled"):
    '''Returns an instance of the dumped basis set computations. 
    Use this to improve the coefficients further.
    '''
    f = open(fname, 'rb')
    return pickle.load(f)
    f.close()

def randomize(a, fac):
    '''Add randomized fluctuations to the array by a defined factor.'''
    from numpy.random import random
    a = np.array(a)
    return a*(1+fac*(2*random(a.shape)-1))

    
def dmol(mol, fac_basis, fac_r):
    '''Creates a molecule with a modified basis (by random changes to the basis set and bond length parameters).
    '''
    from copy import deepcopy

    done = False
    while not done:
        try:
            iden,basis = getcoeffs(mol)
            shape = basis.shape
            # basis = basis.reshape(np.prod(shape))
            
            # co_template = np.random.randint(0,1+1, size=np.prod(shape)).reshape(shape)
            # basis = basis*(1-co_template) +randomize(co_template*basis, fac_basis)
            db = np.sin(np.random.random(size=shape)*2*np.pi)*fac_basis
            basis = basis+db+0.5*db**2
            # basis = basis.reshape(shape)
            
            mol.basis = putcoeffs(iden, basis)
            mol.build()
            mol._atom = [(a,randomize(c, fac_r)) for a,c in mol._atom]
            # mol.build()
            en = getEnergies(mol)
            done = True
        except:
            pass
    
    return deepcopy(mol)
    



def CrossOvers(pop):
    '''Creates a population of molecules by crossovers from a given population.'''
    # The crossovers
    colst = []
    Nparents = len(pop)
    if Nparents%2==1:
        Nparents = Nparents-1
    parents = np.arange(Nparents)
    np.random.shuffle(parents)
    iden,basis = getcoeffs(pop[0][1])
    shape = basis.shape
    # for j in range(2):
    #     for i in range(0, len(parents), 2):
    for k in range(Nparents*3//4):
        np.random.shuffle(parents)
        try:
            # crossovers can fail
            i,j = parents[:2]
            co_template = np.random.randint(0,1+1, size=np.prod(shape)).reshape(shape)
            iden,basis1 = getcoeffs(pop[i][1])
            iden,basis2 = getcoeffs(pop[j][1])
            newbasis = co_template*basis1 +(1-co_template)*basis2
            newmol = deepcopy(pop[i][1])
            newmol.basis = putcoeffs(iden, newbasis)
            newmol.build()
            en = getEnergies(newmol)
            colst.append([en['CISD'], deepcopy(en['mol_CISD'].mol)])
        except:
            pass
    return colst

def Mutate(pop, facs):
    '''Creates a population of molecules by mutations from a given population. Do not use this. Use `Mutate_v2` instead.'''
    mulst = pop[:]
    Nparents = len(pop)
    iden,basis = getcoeffs(pop[0][1])
    shape = basis.shape
    
    for i in range(Nparents):
        newmol = dmol(mulst[0][1], *facs)
        # newmol.build()
        en = getEnergies(newmol)
        if en['CISD']<mulst[0][0]:
            mulst.append([en['CISD'], deepcopy(en['mol_CISD'].mol)])
            mulst = sorted(mulst, key=lambda x:x[0])
            mulst = mulst[:-1]

    return mulst

def Mutate_v2(pop, facs):
    '''Creates a population of molecules by mutations from a given population.'''
    mulst = [pop[0]] #pop[:]
    Nparents = len(pop)
    iden,basis = getcoeffs(pop[0][1])
    shape = basis.shape
    
    for i in range(Nparents-1):
        for j in range(i+1, Nparents):
            iden,basis0 = getcoeffs(pop[i][1])
            iden,basis1 = getcoeffs(pop[j][1])
            # db = (basis1-basis0)/3
            db = (basis1-basis0)*np.random.random(basis1.shape)/3
            basis = putcoeffs(iden,basis0+db)
            newmol = deepcopy(pop[i][1])
            newmol.basis = deepcopy(basis)
            newmol.build()
            try:
                en = getEnergies(newmol)
                if en['CISD']<mulst[0][0]:
                    mulst.append([en['CISD'], deepcopy(en['mol_CISD'].mol)])
                    mulst = sorted(mulst, key=lambda x:x[0])[:-1]
            except:
                pass
    return mulst

def GenerateInitialPopulation(mol, Nmax, facs):
    '''Creates a population of molecules by mutations from a given molecule.'''

    
    pop = [[deepcopy(getEnergies(mol)['CISD']), deepcopy(mol)]]
    # for i in range(Nmax-1):
    while len(pop)<Nmax-1:
        try:
            # sometimes the choice of facs can cause the evaluation to fail
            facs = [np.random.random(), 0]
            mol1 = dmol(mol, *facs)
            pop.append([deepcopy(getEnergies(mol1)['CISD']), deepcopy(mol1)])
        except:
            pass
    # each row has the structure [<energy>, <mol object>]. 
    pop = sorted(pop, key=lambda x:x[0])
    return pop

def OptimizeElement(mol, Nmax, fname, trials=10, tol=1e-10, trials_mu=100, trials_co=100):
    '''
    Optimization of the basis set coefficients using a genetic algorithm-inspired approach.
    Modified to focus on continuous refinement of the population.

    :param mol: the `pyscf` molecule object (Initial best candidate)
    :type mol: :obj:`pyscf.gto.mole.Mole`
    :param Nmax: Population size of candidates to consider
    :type Nmax: :obj:`int`
    :param fname: File where the optimized candidate is stored/updated.
    :type fname: :obj:`str`
    :param trials: The number of major optimization cycles (generations)
    :type trials: :obj:`int`
    :param tol: Tolerance of energy (difference between best/worst) for convergence in a cycle
    :type tol: :obj:`float`
    :param trials_mu: Number of mutation attempts (minor generations) per major cycle
    :type trials_mu: :obj:`int`
    :param trials_co: Number of crossover attempts per mutation trial
    :type trials_co: :obj:`int`

    :returns: a tuple of (<list of the final population>, <list of generations>, <list of best energy in each generation>)
    :rtype: :obj:`tuple`
    '''
    
    # --- 1. Generate Initial Population ---
    # Start with the input molecule, then generate Nmax-1 diverse initial candidates.
    facs_init = [0.5, 0] # Use a relatively large random factor for initial diversity
    initial_pop = GenerateInitialPopulation(mol, Nmax, facs_init)
    pop_current = sorted(initial_pop, key=lambda x:x[0])[:Nmax]

    # lists to keep track of the progress
    gen_counter = 0
    gen = [0]
    en = [pop_current[0][0]]

    print(f"--- Starting Optimization (Nmax={Nmax}) ---", flush=True)

    for i in range(trials):
        
        # Track the best energy at the start of the current generation cycle
        E_start_cycle = pop_current[0][0]
        iters_mu = 0
        
        # --- 2. Mutation & Refinement Cycle ---
        # The mutation cycle attempts to refine the current best candidate
        while iters_mu < trials_mu:
            
            # --- 2a. Dynamic Mutation Factor ---
            # Gradually reduce the mutation factor to fine-tune the coefficients
            facs_mu = [np.random.random() / 10**(i/2), 0]
            
            # Create mutated individuals (Mutate_v2 uses two parents)
            mutated_candidates = Mutate_v2(pop_current, facs_mu)
            
            # Combine current population with new mutations and select the best Nmax
            pop_current = sorted(pop_current + mutated_candidates, key=lambda x:x[0])[:Nmax]
            
            gen_counter += 1
            gen.append(gen_counter)
            en.append(pop_current[0][0])
            
            # --- 2b. Crossover Cycle ---
            # Apply crossovers to increase population diversity
            for j in range(trials_co):
                crossover_candidates = CrossOvers(pop_current)
                
                # Combine current population with new crossovers and select the best Nmax
                pop_current = sorted(pop_current + crossover_candidates, key=lambda x:x[0])[:Nmax]
                
                gen_counter += 1
                gen.append(gen_counter)
                en.append(pop_current[0][0])
            
            # --- 2c. Check Convergence and Update ---
            E_best = pop_current[0][0]
            E_worst = pop_current[-1][0]
            err = np.abs(E_best - E_worst) # Energy difference across the population
            
            print(f"[Cycle: {i:3d}; M:{iters_mu:4d}] Best: {E_best:12.8g} +/- {err:12.8g} Ha, facs={facs_mu[0]:5.3e}", flush=True, end='\r')
            
            # Save the best candidate immediately
            if E_best < E_start_cycle:
                 dumpBSI(deepcopy(getEnergies(pop_current[0][1], fci=True)), fname=fname)
                 E_start_cycle = E_best # Update the benchmark for this cycle
                 
            if err < tol:
                print(f"\nCycle {i} converged (Population spread < {tol:5.1e}). Breaking mutation loop.")
                break

            iters_mu += 1

    # Final dump of the absolute best candidate found
    dumpBSI(deepcopy(getEnergies(pop_current[0][1], fci=True)), fname=fname)
    print("\n--- Optimization Complete ---", flush=True)
    return pop_current, gen, en

    
# def OptimizeElement(mol, Nmax, fname, trials=10, tol=1e-10, trials_mu=100, trials_co=100):
#     '''Optimization of the basis set coefficients using a genetic algorithm-inspired approach. The differences will be noted upon updation of the document.

#     :param mol: the `pyscf` molecule object
#     :type mol: :obj:`pyscf.gto.mole.Mole`
#     :param Nmax: Population of candidates to consider in the optimization
#     :type Nmax: :obj:`int`
#     :param fname: File where the optimized candidate is stored/updated.
#     :type fname: :obj:`str`
#     :param trials: The number of generations to run
#     :type trials: :obj:`str`
#     :param tol: Tolerance of energy for convergence
#     :type tol: :obj:`float`
#     :param tol_mu: Number of mutations to consider under each trial
#     :type trials_mu: :obj:`int`
#     :param trials_co: Number of crossovers to consider after each mutation
#     :type trials_co: :obj:`int`
    
#     :returns: a tuple of (<list of the final population>, <list of generations>, <list of best energy in each generation>)
#     :rtype: :obj:`tuple`

#     :Example:

#     >>> element = 'H'
#     >>> k = 2
#     >>> dirname ='./testdata' # the best candidates will be stored here
#     >>> mol = FetchBestMol(element, k, dirname=dirname)['mol_CISD'].mol # fetch the best available candidate
#     >>> fname = os.path.join(dirname, f"{prefix}_{element}_msto-{k:02d}g.pickle") # the best candidate is pickled here
#     >>> pop2, gen, en = OptimizeElement(mol, Nmax, fname, trials=20, tol=1e-16, trials_mu=15, trials_co=20, itermax=15) # optimize
    
#     '''
#     # Here we will try exploring the parameter space heuristically with a modified mutation algorithm by mutating along the parameter gradient.
#     facs = [np.random.random(), 0]
#     pop = [[deepcopy(getEnergies(mol)['CISD']), deepcopy(mol)]]+GenerateInitialPopulation(mol, 5*Nmax, facs)
#     # sort them by energies
#     pop2 = sorted(pop, key=lambda x:x[0])[:Nmax]
    
#     # lists to keep track of the progress
#     gen = [0]
#     en = [pop2[0][0]]

#     iters=0
#     for i in range(trials):
        
#         err = 1e9
#         print(f"[i: {i:4d}; M:{iters:5d}; C:{0:5d}] Best: {pop2[0][0]:12.8g} +/- {np.abs(pop2[0][0]-pop2[-1][0]):12.8g} Ha, facs={facs}", flush=True, end='\r')
#         while err>tol and iters<trials_mu:
#             iters = iters+1
#             j=0
#             # generate the initial population
#             pop2 = GenerateInitialPopulation(deepcopy(pop2[0][1]), 2*Nmax, facs)
#             # sort them by energies
#             pop2 = sorted(pop2, key=lambda x:x[0])[:Nmax]
    
#             facs = [np.random.random()/10**(i%10), 0]
#             pop2 = sorted(pop2+Mutate_v2(pop2, facs), key=lambda x:x[0])[:Nmax]
#             print(f"[i: {i:4d}; M:{iters:5d}; C:{j:5d}] Best: {pop2[0][0]:12.8g} +/- {np.abs(pop2[0][0]-pop2[-1][0]):12.8g} Ha, facs={facs}", flush=True, end='\r')
#             iters = iters+1
#             gen.append(iters)
#             en.append(pop2[0][0])
#             for j in range(trials_co):
#                 pop2 = sorted(pop2+CrossOvers(pop2), key=lambda x:x[0])[:Nmax]
#                 print(f"[i: {i:4d}; M:{iters:5d}; C:{j:5d}] Best: {pop2[0][0]:12.8g} +/- {np.abs(pop2[0][0]-pop2[-1][0]):12.8g} Ha, facs={facs}", flush=True, end='\r')
#             iters = iters+1
#             gen.append(iters)
#             en.append(pop2[0][0])
#             dumpBSI(deepcopy(getEnergies(pop2[0][1], fci=True)), fname=fname)
            
#             # gen.append(iters)
#             # en.append(pop2[0][0])
            
#             err = np.abs(pop2[0][0]-pop2[-1][0])
#     dumpBSI(deepcopy(getEnergies(pop2[0][1], fci=True)), fname=fname)
#     return pop2, gen, en

def AggressiveRefinement(mol, fname, trials=10, tol=1e-16, fci=True, itermax=100):
    iden,basis = getcoeffs(mol)
    en = getEnergies(mol)['CISD']
    fac = np.random.random()
    
    for i in range(trials):
        fac = np.random.random()#*fac
        # fac = fac*(fac<0.5)+(1/10**l)*(fac>=0.5)
        err = 1e9
    
        
        
        for j in range(10):
            iters=0
            while err>tol and iters<itermax:
                
                try:
                    iters=iters+1
                    print(f"[i: {i:4d}.{iters:05d}] Best: {en:20.16g} +/- {err:5.3g} Ha, facs={fac:5.3g}"+' '*20, flush=True, end='\r')
                    iters = iters+1
                    db = np.sin(np.random.random(size=basis.shape)*2*np.pi)
                    # # db = np.sin(np.random.normal(loc=0, scale=1, size=basis.shape)*2*np.pi)
                    basis1 = basis+fac*db/10**j
                    
                    mol1 = deepcopy(mol)
                    # mol1 = dmol(mol, fac, 0)
                    mol1.basis = putcoeffs(iden, basis1)
                    
                    mol1.build()
                    en1 = getEnergies(mol1)['CISD']
                    
                    if en1<en:
                        err = np.abs(en1-en)
                        basis = deepcopy(basis1)
                        mol.basis = putcoeffs(iden, basis)
                        mol.build()
                        en = deepcopy(en1)
                        dumpBSI(deepcopy(getEnergies(mol, fci=True)), fname=fname)
                        iters = min([0, iters//2])
                        print(f"[i: {i:4d}.{iters:05d}] Best: {en:20.16g} +/- {err:5.3g} Ha, facs={fac:5.3g}"+' '*20, flush=True, end='\r')
                except:
                    pass
                
            print(f"[i: {i:4d}.{iters:05d}] Best: {en:20.16g} +/- {err:5.3g} Ha, facs={fac:5.3g}"+' '*20, flush=True, end='\r')
    # print()

def LogicalRefinement(element, Nmax, trials=20, tol=1e-16, trials_mu=15, trials_co=20, prefix = 'MOBS', dirname='./', fci=True, itermax=25, agg_trial=True, ga_trial=True, kmax=12):
    ''' Refine the MSTO basis sets generated. For 3 <= k <= 11 if the FCI energy for a given k is larger than that for k-1, an optimization is performed.

        :param mol: the `pyscf` molecule object
        :type mol: :obj:`pyscf.gto.mole.Mole`
        :param Nmax: Population of candidates to consider in the optimization
        :type Nmax: :obj:`int`
        :param trials: The number of generations to run
        :type trials: :obj:`str`
        :param tol: Tolerance of energy for convergence
        :type tol: :obj:`float`
        :param tol_mu: Number of mutations to consider under each trial
        :type trials_mu: :obj:`int`
        :param trials_co: Number of crossovers to consider after each mutation
        :type trials_co: :obj:`int`
        :param prefix: Prefix of the pickled database
        :type prefix: :obj:`str` (optional. Defaults to `MOBS`)
        :param dirname: Directory to look for the dumped database files.
        :type dirname: :obj:`str` (optional. Defaults to `./`)
        
        :returns: Nothing
        :rtype: None
    
        :Example:

        >>> element = 'H'
        >>> Nmax = 10
        >>> LogicalRefinement(element, Nmax, trials=20, tol=1e-16, trials_mu=15, trials_co=20, prefix = 'MOBS', dirname=dirname, klst=range(3, 12))
    '''

    if not (ga_trial and agg_trial):
        agg_trial=True # have the agg_trial if both are set to False
    k = 2
    while k<kmax:
        k = k+1
        # print(f"k={k}")
        pfname = os.path.join(dirname, f"{prefix}_{element}_msto-{k-1:02d}g.pickle")
        en0 = loadBSI(pfname)['CISD']
        nfname = os.path.join(dirname, f"{prefix}_{element}_msto-{k+1:02d}g.pickle")
        en2 = loadBSI(nfname)['CISD']
        
        # try:
        fname = os.path.join(dirname, f"{prefix}_{element}_msto-{k:02d}g.pickle")
        dump = loadBSI(fname)
        en1 = dump['CISD']
        
        # while en1 > 0.5*(en0+en2) or en1>en0:
        while en1>en0:
            
            mol = dump['mol_CISD'].mol
            if ga_trial:
                pop2, gen, en = OptimizeElement(mol, Nmax, fname, trials=trials, tol=tol, trials_mu=trials_mu, trials_co=trials_co)
                mol = deepcopy(pop2[0][1])
            if agg_trial:
                AggressiveRefinement(mol, fname, trials=trials, tol=tol, fci=fci, itermax=itermax)
            dump = loadBSI(fname)
            en1 = dump['CISD']
            # print()
            # k=k-1
     

def moldict(basis):
    '''Returns a `pyscf.gto.mole.Mole` molecule dictionary for the chosen basis. The atom positions are from standard data. The elements/molecules are limited and can be improved. This is to provide a ready database of molecules.

    :param basis: a string that specifies the standard basis set. It is expected to be a minimal basis set.
    :type basis: string

    :returns: A dictionary of molecules with standard parameters.
    :rtype: dict

    :Example:
    
    >>> # choose your element
    >>> element = 'H'
    >>> # get the default molecule in your basis
    >>> mol = moldict('sto-6g')[element]
    >>> type(mol)
    pyscf.gto.mole.Mole
    >>> moldict('sto-6g')
    {'H': <pyscf.gto.mole.Mole at 0x78fd225c96d0>,
    'He': <pyscf.gto.mole.Mole at 0x78fd225c9810>,
    'Li': <pyscf.gto.mole.Mole at 0x78fcae2851d0>,
    'Be': <pyscf.gto.mole.Mole at 0x78fcae285310>,
    'B': <pyscf.gto.mole.Mole at 0x78fcae285090>,
    'C': <pyscf.gto.mole.Mole at 0x78fcae284f50>,
    'N': <pyscf.gto.mole.Mole at 0x78fcae284e10>,
    'O': <pyscf.gto.mole.Mole at 0x78fcae284cd0>,
    'F': <pyscf.gto.mole.Mole at 0x78fcae284b90>,
    'H2': <pyscf.gto.mole.Mole at 0x78fcae284a50>,
    'HF': <pyscf.gto.mole.Mole at 0x78fcae284910>,
    'LiH': <pyscf.gto.mole.Mole at 0x78fcae2847d0>,
    'H2O': <pyscf.gto.mole.Mole at 0x78fcae284690>}
        
    '''
    mol={}
    
    mol['H'] = pyscf.M(
                atom = '''H 0.0 0.0 0.0;''', # in Angstrom
                basis = {'H': gto.load(basis, 'H')},
                symmetry = False,
                spin=1
            )
    mol['He'] = pyscf.M(
                atom = '''He 0.0 0.0 0.0;''', # in Angstrom
                basis = {'He': gto.load(basis, 'He')},
                symmetry = False,
                spin=0
            )
    
    mol['Li'] = pyscf.M(
                atom = '''Li 0.0 0.0 0.0;''', # in Angstrom
                basis = {'Li': gto.load(basis, 'Li')},
                symmetry = False,
                spin=1
            )
    
    mol['Be'] = pyscf.M(
                atom = '''Be 0.0 0.0 0.0;''', # in Angstrom
                basis = {'Be': gto.load(basis, 'Be')},
                symmetry = False,
                spin=0
            )
    
    mol['B'] = pyscf.M(
                atom = '''B 0.0 0.0 0.0;''', # in Angstrom
                basis = {'B': gto.load(basis, 'B')},
                symmetry = False,
                spin=1
            )

    mol['C'] = pyscf.M(
                atom = '''C 0.0 0.0 0.0;''', # in Angstrom
                basis = {'C': gto.load(basis, 'C')},
                symmetry = False,
                spin=2
            )

    mol['N'] = pyscf.M(
                atom = '''N 0.0 0.0 0.0;''', # in Angstrom
                basis = {'N': gto.load(basis, 'N')},
                symmetry = False,
                spin=3
            )

    mol['O'] = pyscf.M(
                atom = '''O 0.0 0.0 0.0;''', # in Angstrom
                basis = {'O': gto.load(basis, 'O')},
                symmetry = False,
                spin=2
            )
    
    mol['F'] = pyscf.M(
                atom = '''F 0.0 0.0 0.0;''', # in Angstrom
                basis = {'F': gto.load(basis, 'F')},
                symmetry = False,
                spin=1
            )
    
   
    
    
    mol['H2'] = pyscf.M(
                atom = '''H 0.0 0.0 0.0; H 0.0 0.0 0.735; ''', # in Angstrom
                basis = {'H': gto.load(basis, 'H')},
                symmetry = False,
                spin=0
            )
    
    
    
    mol['HF'] = pyscf.M(
                atom = '''  F 0.0 0.0 0.0; 
                            H 0.0 0.0 0.9168;''', # in Angstrom
                basis = {'H': gto.load(basis, 'H'),
                         'F': gto.load(basis, 'F')},
                symmetry = False,
                spin=0
            )

    mol['LiH'] = pyscf.M(
                atom = '''  Li 0.0 0.0 0.0; 
                            H 0.0 0.0 1.5949;''', # in Angstrom
                basis = {'H': gto.load(basis, 'H'),
                         'Li': gto.load(basis, 'Li')},
                symmetry = False,
                spin=0
            )
    
    mol['H2O'] = pyscf.M(
                atom = '''  O 0.0 0.0 0.1173; 
                            H 0.0 0.7572 -0.4692; 
                            H 0.0 -0.7572 -0.4692;''', # in Angstrom
                basis = {'H': gto.load(basis, 'H'),
                         'O': gto.load(basis, 'O')},
                symmetry = False,
                spin=0
            )

    return mol


def PrintEnergies(element, dirname = "./AtomOptimizedv3/", table=True, plot=False, prefix="MOBS", save=True, ylim=None, kmax=11, inset_loc = [0.6,0.6, 0.35, 0.35], inset_xlim=(1.9,11.1), inset_ylim=(-10,0)):
    try:
        STO = {}
        for k in range(2,7):
            # mol = pyscf.M(
            #                             atom = f'''{element} 0.0 0.0 0.0;''', # in Angstrom
            #                             basis = {element: gto.load(f"sto-{k if k<=6 else 6}g", element)},
            #                             symmetry = False,
            #                             spin=1
            #                         )
            fname = os.path.join(dirname, f"BS_{element}_sto-{k}g.pickle")
            STO[k] = loadBSI(fname) #getEnergies(mol, nroots=4)
            
        
        MSTO = dict([(k, {'HF':'-', 'FCI':['-'], 'CISD':'-'}) for k in range(2,12)])
        MSTO_coeffs = {}
        files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and f[-7:]=='.pickle']
        for k in range(2,kmax+1):
            fname = os.path.join(dirname, f"{prefix}_{element}_msto-{k:02d}g.pickle")
            try:
                MSTO[k]=loadBSI(fname)
                MSTO_coeffs[k] = MSTO[k]['mol_CISD'].mol.basis
            except:
                pass
    except:
        pass
        
    T = PrettyTable()
    T.field_names = ['k', 'HF (STO)', 'HF (MSTO)', 'FCI (STO)', 'FCI (MSTO)', 'CISD (STO)', 'CISD (MSTO)']
    for k in range(2,7):
        try:
            T.add_row([k, "%10.7f"%STO[k]['HF'], "%10.7f"%MSTO[k]['HF'], "%10.7f"%STO[k]['FCI'][0], "%10.7f"%MSTO[k]['FCI'][0], "%10.7f"%STO[k]['CISD'], "%10.7f"%MSTO[k]['CISD']] )
        except:
            pass
    
    for k in range(7, kmax+1):
        try:
            T.add_row([k, '', "%10.7f"%MSTO[k]['HF'], '', "%10.7f"%MSTO[k]['FCI'][0], '', "%10.7f"%MSTO[k]['CISD']] )
        except:
            pass
    T.align='l'
    T.hrules = HRuleStyle.ALL

    T1 = PrettyTable()
    T1.field_names = ['Basis', 'HF', 'FCI', 'CISD']
    for basis in ['6-31g', 'ccpvdz']:
        try:
            en = loadBSI(os.path.join(dirname, f"BS_{element}_{basis}.pickle") )
            T1.add_row([basis, "%10.7f"%en['HF'], "%10.7f"%en['FCI'][0], "%10.7f"%en['CISD']])
        except:
            pass

    T1.align='l'
    T1.hrules = HRuleStyle.ALL
    
    print(f"Element: {element} (Last update: {time.strftime('%Y-%M-%D  %H:%M')})") 
    if table:
        print(T1)
        print(T)
        pprint.pprint(MSTO_coeffs)
    if plot:
        # pl.ion()
        x = [k for k in range(2, kmax+1)]
        sto_fci = np.array([[k, STO[k]['FCI'][0]] for k in STO.keys() if k in STO.keys()] )
        sto_hf = np.array([[k, STO[k]['HF']] for k in STO.keys() if k in STO.keys()])
        sto_cisd = np.array([[k, STO[k]['CISD']] for k in STO.keys() if k in STO.keys()])
        
        msto_fci = np.array([[k, MSTO[k]['FCI'][0]] for k in MSTO.keys() if MSTO[k]['FCI'][0]!='-'])
        msto_hf = np.array([[k, MSTO[k]['HF']] for k in MSTO.keys() if MSTO[k]['FCI'][0]!='-'])
        msto_cisd = np.array([[k, MSTO[k]['CISD']] for k in MSTO.keys() if MSTO[k]['FCI'][0]!='-'])
        # pl.ion()
        fig = pl.figure()
        ax1 = fig.add_subplot(1,1,1)
        
        
        

        
        ax1.plot(sto_hf[:,0], sto_hf[:,1], 'o-', label='STO [HF]', alpha=0.5)
        ax1.plot(msto_hf[:,0], msto_hf[:,1], 'o-', label='MSTO [HF]', alpha=0.5)
        
        ax1.plot(msto_cisd[:,0], msto_cisd[:,1], 'o-', label='MSTO [CISD]', alpha=0.5)
        ax1.plot(sto_cisd[:,0], sto_cisd[:,1], 'o-', label='STO [CISD]', alpha=0.5)

        try:
            ax1.plot(sto_fci[:,0], sto_fci[:,1], 'o-', label='STO [FCI]', alpha=0.5)
            ax1.plot(msto_fci[:,0], msto_fci[:,1], 'o-', label='MSTO [FCI]', alpha=0.5)
        except:
            pass
        
        en631g = loadBSI(os.path.join(dirname, f"BS_{element}_6-31g.pickle"))
        ax1.axhline(en631g['HF'], color="#0000ff", linestyle='dashed', label='6-31G (HF)', alpha=0.5)
        try:
            ax1.axhline(en631g['FCI'][0], color="#0000ff", linestyle='solid', label='6-31G (FCI)', alpha=0.5)
        except:
            pass
            
        enccpvdz = loadBSI(os.path.join(dirname, f"BS_{element}_ccpvdz.pickle"))
        ax1.axhline(enccpvdz['HF'], color="#00ff00", linestyle='dashed', label='cc-pVDZ (HF)', alpha=0.5)
        try:
            ax1.axhline(enccpvdz['FCI'][0], color="#00ff00", linestyle='solid', label='cc-pVDZ (FCI)', alpha=0.5)
        except:
            pass
            
        ax1.set_title(element)
        ax1.set_xlabel(r"$k$")
        ax1.set_ylabel(r"$E$ (Ha)")
        ax1.legend(loc='right', framealpha=0.5)
        if ylim is not None:
            ax1.set_ylim(ylim)
        pl.grid(True)
        ############################################
        ## adding the inset plot
        axins = ax1.inset_axes(inset_loc, xlim=inset_xlim, ylim=inset_ylim)
        axins.plot(sto_hf[:,0], sto_hf[:,1], 'o-', label='STO [HF]', alpha=0.5)
        axins.plot(msto_hf[:,0], msto_hf[:,1], 'o-', label='MSTO [HF]', alpha=0.5)
        
        axins.plot(msto_cisd[:,0], msto_cisd[:,1], 'o-', label='MSTO [CISD]', alpha=0.5)
        axins.plot(sto_cisd[:,0], sto_cisd[:,1], 'o-', label='STO [CISD]', alpha=0.5)

        try:
            axins.plot(sto_fci[:,0], sto_fci[:,1], 'o-', label='STO [FCI]', alpha=0.5)
            axins.plot(msto_fci[:,0], msto_fci[:,1], 'o-', label='MSTO [FCI]', alpha=0.5)
        except:
            pass
        
        # en631g = loadBSI(os.path.join(dirname, f"BS_{element}_6-31g.pickle"))
        axins.axhline(en631g['HF'], color="#0000ff", linestyle='dashed', label='6-31G (HF)', alpha=0.5)
        try:
            axins.axhline(en631g['FCI'][0], color="#0000ff", linestyle='solid', label='6-31G (FCI)', alpha=0.5)
        except:
            pass
            
        # enccpvdz = loadBSI(os.path.join(dirname, f"BS_{element}_ccpvdz.pickle"))
        axins.axhline(enccpvdz['HF'], color="#00ff00", linestyle='dashed', label='cc-pVDZ (HF)', alpha=0.5)
        try:
            axins.axhline(enccpvdz['FCI'][0], color="#00ff00", linestyle='solid', label='cc-pVDZ (FCI)', alpha=0.5)
        except:
            pass
        axins.set_xticks(range(int(inset_xlim[0]), int(inset_xlim[-1]+1)), [str(i) for i in range(int(inset_xlim[0]), int(inset_xlim[-1]+1))])
        # axins.set_yticklabels('')
        axins.grid(True)
        if save:
            fig.savefig(f"{element}.png", dpi=300)
        return fig, T1, T, MSTO, MSTO_coeffs

def FetchBestMol(element, k, dirname='./'):
    '''Fetches the best available basis for optimization of `element` for cardinal number `k`.
    
    The order is as follows for a cardinal number `k`:
    
    * MSTO-kG 
    * STO-kG
    * MSTO-(k-1)G
    * STO-(k-1)G
    * ........ 

    It is best to ensure that at least the STO-2G basis is available.

    To do:
    
    * if k<2 fetch the sto-2g basis within this function.
    '''
    try:
        basis = f"msto-{k:02d}g"
        mol = loadBSI(os.path.join(dirname, f"MOBS_{element}_{basis}.pickle"))['mol_CISD'].mol
    except:
        try:
            basis = f"sto-{k}g"
            mol = loadBSI(os.path.join(dirname, f"BS_{element}_{basis}.pickle"))['mol_CISD'].mol
        except:
            basis = f"sto-{6}g"
            mol = loadBSI(os.path.join(dirname, f"BS_{element}_{basis}.pickle"))['mol_CISD'].mol
    return BasisReshape(mol,k)

def DumpStandardBasis(element, basis, dirname='./', forcedump=False):
    '''Create the molecule object for a molecule in a given basis and dump in a file. 
    A new file will be created only if it does not exist. 
    The format of the filename will be `BS_<element>_<basis>.pickle` and stored in the specified directory.

    :param element: A molecule defined in :func:`MSTOOpt.moldict`.
    :type element: :obj:`str`
    :param basis: A basis name defined in `basis_set_exchange package` for which the energy is to be computed.
    :type basis:  :obj:`str`
    :param dirname: The directory where the molecule object is to be dumped.
    :type dirname: :obj:`str`

    :Example:

    >>> DumpStandardBasis('H', 'sto-4g', dirname='./')
    '''
    from os.path import exists
    
    mol = moldict(basis)[element]
    fname = os.path.join(dirname,f"BS_{element}_{basis}.pickle")
    if forcedump or (not exists(fname)):
        dumpBSI(deepcopy(getEnergies(mol, fci=True)), fname=fname)

    
def CreateMoleculesStdBasisSets(molecules='H Li Be B C N O F H2 LiH HF', basissets=['6-31g', 'ccpvdz'], forcedump=False, dirname='./'):
    '''Batch creation of molecule objects from standard basis sets

    :param molecules: A string of space separated molecules as specified in :func:`MSTOOpt.moldict`
    :type molecules: :obj:`str`
    :param basissets: A list of basis set names for which the molecule objects are computed.
    :type basissets: :obj:`list`

    :Example:
    
    >>> CreateMoleculesStdBasisSets(molecules='H Li Be B C N O F H2 LiH HF', basissets=['6-31g', 'ccpvdz'])
    '''
    for element in molecules.split(' '):
        for k in range(2, 7):
            basis = f"sto-{k}g"
            try: 
                print(f"    :   {element} {basis}", end='\r', flush=True)
                DumpStandardBasis(element, basis, dirname=dirname, forcedump=forcedump)
                print(f"Done:   {element} {basis}")
            except:
                print(f"Failed: {element} {basis}")
        for basis in basissets:
            try: 
                print(f"    :   {element} {basis}", end='\r', flush=True)
                DumpStandardBasis(element, basis, dirname=dirname, forcedump=forcedump)
                print(f"Done:   {element} {basis}")
            except:
                print(f"Failed: {element} {basis}")

def EvsR_dimers(elements, mol, bondlengths):
    enlst={'HF': [],
    'FCI': [],
    'CISD': []}
    
    for bl in tqdm(bondlengths):
        mol.atom = f'''{elements[0]} 0 0 0; {elements[1]} 0 0 {bl};'''
        mol.build()
        en = getEnergies(mol, nroots=4, fci=True, cisd=True)
        en['FCI'] = en['FCI'][0]
        for approx in ['HF', 'FCI', 'CISD']:
            enlst[approx].append(en[approx])
        # print(en)

    return bondlengths, enlst