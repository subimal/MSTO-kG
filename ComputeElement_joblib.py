import os
import numpy as np
import sys
import logging
import csv
from datetime import datetime
from copy import deepcopy
from joblib import Parallel, delayed

# Import your quantum chemistry modules
from MSTOkG.MSTOOptpar import *



def getonecandidate_local(candidate, fac=1e-1):
    try:
        mol = candidate['mol_CISD'].mol
        fac_basis, fac_r = np.random.random(2)*fac
        mol1 = dmol(mol, fac_basis, fac_r)
        en = getEnergies(mol1, nroots=4, fci=False, cisd=True)
        return en
    except Exception:
        return None

def AggressiveRefinement_v2(mol, fname, shots=10, tol=1e-8, itermax=10, n=3, passes=5, mu=50, ncores=10):
    en_data = getEnergies(mol, nroots=4, fci=False, cisd=True)
    en0_val = en_data['CISD']
    facs = 1/10**np.arange(0, n+1)
    err = 1e9
    n_cores = ncores

    # Initialize CSV
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Pass', 'Fac', 'Iteration', 'Tasks_Finished', 'CISD_Energy', 'Error'])

    passno = 0
    while passno<passes:
        for fac_idx, fac in enumerate(facs):
            for shot in range(shots):
                
                Improved = True
                iters = 0
                while Improved and err > tol and iters < itermax:
                    faci = fac * np.random.random()
                    Improved = False
                    # total_tasks = shots * mu
                    total_tasks =  mu
                    
                    tasks = (delayed(getonecandidate_local)(en_data, fac=faci) for _ in range(total_tasks))
                    joblib_generator = Parallel(n_jobs=n_cores, return_as='generator')(tasks)
    
                    tasks_finished = 0
                    for result in joblib_generator:
                        tasks_finished += 1
                        
                        if result is not None:
                            if result['CISD'] < en_data['CISD']:
                                en_data = result
                                Improved = True
                                dumpBSI(deepcopy(getEnergies(en_data['mol_CISD'].mol, fci=True)), fname=fname)
                                err = np.abs(en0_val - en_data['CISD'])
                                en0_val = en_data['CISD']
                                
                                # Log improvement to the .log file
                                logging.info(f"IMPROVEMENT: Pass {passno}, Iter {iters} | New Energy: {en_data['CISD']:.10f}")
                                
                                # Log to CSV for plotting later
                                with open(csv_filename, mode='a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([datetime.now(), passno, faci, iters, tasks_finished, en_data['CISD'], err])
                                # iters=max([0,iters-1])
                                # Terminal Progress Update (Status Bar)
                                print(f"{element} | k = {k} | Pass {passno} shot: {shot} | Progress: {tasks_finished}/{total_tasks} | Best E: {en_data['CISD']:.10f} | fac: {faci} | err: {err:.4g}", end='\r', flush=True)
    
                        # Terminal Progress Update (Status Bar)
                        print(f"{element} | k = {k} | Pass {passno} shot: {shot} | Progress: {tasks_finished}/{total_tasks} | Best E: {en_data['CISD']:.10f} | fac: {faci} | err: {err:.4g}", end='\r', flush=True)
                    
                    print()
                    logging.info(f"Finished Iteration {iters} of Pass {passno}")
                    iters += 1
        print(f"{element} | k = {k} | Pass {passno} shot: {shot} | Progress: {tasks_finished}/{total_tasks} | Best E: {en_data['CISD']:.10f} | fac: {faci} | err: {err:.4g}", end='\r', flush=True)
        passno =passno+1
    print()

if __name__ == "__main__":

    element = sys.argv[1]

    # --- Setup Logging ---
    log_filename = f"refinement_{element}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    csv_filename = f"convergence_{element}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)]
    )
    dirname = "./data/"
    if len(sys.argv)>2:
        klst = [int(k) for k in sys.argv[2:]]
    else:
        klst = range(2, 12)
    for k in klst:
        fname = os.path.join(dirname, f"MOBS_{element}_msto-{k:02d}g.pickle")
        mol = FetchBestMol(element, k, dirname=dirname)
        
        logging.info(f"Starting refinement for {element} (k={k})")
        AggressiveRefinement_v2(mol, fname, shots=2, tol=1e-8, itermax=20, n=10, passes=10, mu=1000, ncores=15)
    logging.info("Refinement process complete.")
