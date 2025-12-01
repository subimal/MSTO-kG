from MSTOkG.MSTOOpt import *
from joblib import Parallel, delayed
import sys


dirname = "./testdata/"
prefix = 'MOBS'

# choose your element
element = 'B'
# get the default molecule in your basis
mol = moldict('sto-2g')[element]
# print the energy (HF and FCI)
Nmax = 10
# facs = [np.random.random(), 0]
tol = 1e-10
itermax=10
# print(getEnergies(mol))



for passes in range(3):
    for k in range(2, 13):
        print(f"pass {passes}; k={k}")
        mol = FetchBestMol(element, k, dirname=dirname) 
        fname = os.path.join(dirname, f"{prefix}_{element}_msto-{k:02d}g.pickle")
        pop2, gen, en = OptimizeElement(mol, Nmax, fname, trials=20, tol=tol, trials_mu=20, trials_co=20)
        print()
        # print("Trying an aggressive refinement")
        # AggressiveRefinement(pop2[0][1], fname, trials=10, tol=tol, fci=True, itermax=10)
        # print()


LogicalRefinement(element, Nmax, trials=12, tol=tol, trials_mu=50, trials_co=10, 
                  prefix = 'MOBS', dirname=dirname, ga_trial=True, agg_trial=False, itermax=200)


