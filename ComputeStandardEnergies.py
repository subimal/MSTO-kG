#!/usr/bin/env python
# coding: utf-8

# In[1]:


from MSTOkG.MSTOOpt import *
# from joblib import Parallel, delayed
import sys
from pyscf import gto, scf, fci

import pandas as pd


# In[2]:


df = {}


# In[ ]:


#for element in 'B C N O F'.split(' '):
#for element in 'H Li Be B'.split(' '):
for element in 'N O F'.split(' '):
    #print(f"Element: {element:3s}", end='; ', flush=True)
    df[element] = pd.DataFrame(columns=["k", "Basis", "E_HF", "E_FCI", "E_cor(FCI)"])

    for k in range(2,7):
        basis = f'sto-{k}g'
        mol = moldict(basis)[element]
        en = getEnergies(mol, cisd=False, fci=True)
        ecor = en['FCI'][0] - en['HF']
        row = [f"{k}", 'STO', en['HF'], en['FCI'][0], ecor ]
        df[element].loc[len(df[element])] = row
        #print(f"{basis}", end='. ', flush=True)
        print(f"{element:5s} | {basis:10s} | HF: {en['HF']:20.15f} | FCI: {en['FCI'][0]:20.15f} | Ecor: {ecor:20.15f}", flush=True)

    for k in range(2,12):
        try:
            basis = f'msto-{k}g'
            dirname = './data/'
            # mol = moldict(basis)['H']
            en = loadBSI(dirname + f"MOBS_{element}_msto-{k:02d}g.pickle")
            ecor = en['FCI'][0] - en['HF']
            row = [f"{k}", 'MSTO', en['HF'], en['FCI'][0], ecor ]
            df[element].loc[len(df[element])] = row
            #print(f"{basis}", end='. ', flush=True)
            print(f"{element:5s} | {basis:10s} | HF: {en['HF']:20.15f} | FCI: {en['FCI'][0]:20.15f} | Ecor: {ecor:20.15f}", flush=True)
        except:
            pass
    
    bases = {'H': ['6-31g', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ'],
             'Li': ['6-31g', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ'],
             'Be': ['6-31g', 'cc-pVDZ', 'cc-pVTZ'],
             'B': ['6-31g', 'cc-pVDZ', 'cc-pVTZ'],
             'C': ['6-31g', 'cc-pVDZ', 'cc-pVTZ'],
             'N': ['6-31g'],
             'O': ['6-31g'],
             'F': ['6-31g'],
             }
    for basis in bases[element]:

        k = ''
        mol = moldict(basis)[element]
        mol.max_memory = 200000
        en = getEnergies(mol, cisd=False, fci=True)
        ecor = en['FCI'][0] - en['HF']
        row = [f"{k}", basis, en['HF'], en['FCI'][0], ecor ]
        df[element].loc[len(df[element])] = row
        #print(f"{basis}", end='. ', flush=True)
        print(f"{element:5s} | {basis:10s} | HF: {en['HF']:20.15f} | FCI: {en['FCI'][0]:20.15f} | Ecor: {ecor:20.15f}", flush=True)
    print('', flush=True)        

    with pd.ExcelWriter(f"EnergiesRevised_{element}.xlsx") as writer:
        for sheet_name, dfi in df.items():
            dfi.to_excel(writer, sheet_name=sheet_name, index=False)









