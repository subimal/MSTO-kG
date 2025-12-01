from MSTOkG.MSTOOpt import *
from joblib import Parallel, delayed, parallel_config
import sys

assert len(sys.argv)==2, "Usage: python ComputeHtoF.py <element>"

dirname = "./testdata/"
prefix = 'MOBS'

# choose your element

def optimize_level1(element):
    print(f"Computing (optimization by GA) for {element}")
    # get the default molecule in your basis
    mol = moldict('sto-2g')[element]
    # print the energy (HF and FCI)
    Nmax = 10
    # facs = [np.random.random(), 0]
    tol = 1e-10
    itermax=10
    # print(getEnergies(mol))
    
    
    def optimize_level2(k):
        print(f"pass {passes}; k={k}")
        mol = FetchBestMol(element, k, dirname=dirname) 
        fname = os.path.join(dirname, f"{prefix}_{element}_msto-{k:02d}g.pickle")
        pop2, gen, en = OptimizeElement(mol, Nmax, fname, trials=20, tol=tol, trials_mu=20, trials_co=20)
        print()
        print("Trying an aggressive refinement")
        AggressiveRefinement(pop2[0][1], fname, trials=50, tol=tol, fci=True, itermax=100)
        print()

        
    for passes in range(3):
        with parallel_config(backend='threading', n_jobs=10):
            Parallel()(delayed(optimize_level2)(k) for k in range(2, 13))
            
            
    print(f"Computing (logical refinement) for {element}")
    LogicalRefinement(element, Nmax, trials=12, tol=tol, trials_mu=50, trials_co=10, 
                      prefix = 'MOBS', dirname=dirname, ga_trial=True, agg_trial=False, itermax=200)

# Parallel(n_jobs=4)(delayed(optimize_level1)(element) for element in ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F'])
# for element in ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F'][::-1]:
element = sys.argv[1]
optimize_level1(element)

# for element in ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F']:
#     print(f"Computing (logical refinement) for {element}")
#     LogicalRefinement(element, Nmax, trials=12, tol=tol, trials_mu=50, trials_co=10, 
#                       prefix = 'MOBS', dirname=dirname, ga_trial=True, agg_trial=False, itermax=200)


