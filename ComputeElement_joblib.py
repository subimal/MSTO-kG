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
