#!/usr/bin/env python
# coding: utf-8

# # ULMS infercnvpy for the revision of the paper
# - Looping through by batch for a reviewer comment

# In[ ]:


import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import infercnvpy as cnv
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpacnv


# In[ ]:


# version control
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scanpy:", sc.__version__)
print("seaborn:", sns.__version__)
print("infercnv:", cnv.__version__)


# In[ ]:


# Set up input and output directories
CURRENT_DIR = Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent
print(PROJECT_DIR)

DATA_DIR = PROJECT_DIR / 'preprocessed'
print(DATA_DIR)

OUTPUT_MASTER_DIR = jpacnv.create_output_dir(PROJECT_DIR, 'cnv_by_batch', change_dir=True)


# In[ ]:


N_JOBS = 16
WINDOW_SIZE = 500
IMMUNE_TYPES = ["B", "Myeloid", "T_and_NK", "Mast", "Plasma", "pDC"]


# # Running the infercnvpy pipeline

# In[17]:


# Get the ensg map to make the GOF
ref = PROJECT_DIR / 'ref/GRCh38-2024-A.txt'
sample_adata = sc.read_h5ad(DATA_DIR / 'Batch01_pp_singlet_adata.h5ad')
ensg_map = sample_adata.var['gene_ids'].to_dict()
del sample_adata


# In[ ]:


# Get the annotations
ann = sc.read_h5ad(PROJECT_DIR / 'objects/annotated_raw_counts.h5ad')
print(ann)


# In[ ]:


adata_list = jpacnv.import_data(DATA_DIR)
adata_list.sort(key=lambda x: x.uns['filename'])
print(len(adata_list))
print(adata_list)


# In[ ]:


resolution = 0.5
str_res = str(resolution).replace('.', '_')
leiden_key = 'leiden' + str_res

for adata in adata_list:
    print()
    batch = adata.uns['batch']
    print(batch)
    output_dir = jpacnv.create_output_dir(OUTPUT_MASTER_DIR, batch, change_dir=True)
    print('Lognormalization...')
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    print("Making the gene order file...")
    adata = jpacnv.gof_format(adata, ref=ref, ensg_map=ensg_map)
    adata.var.loc[:, ["ensg", "chromosome", "start", "end"]].head()

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key)
    sc.pl.umap(adata, color=leiden_key, save=f'{batch}_{str_res}.png')
    
    # Apply the annotations
    # Keep only cells that exist in ann
    print("Applying the annotation...")
    common_idx = adata.obs.index.isin(ann.obs.index)
    adata = adata[common_idx].copy()
    adata.obs['celltype'] = ann.obs.loc[adata.obs.index, 'celltype'].values
    types = adata.obs["celltype"].unique()
    print(*types)
    immune_types = [t for t in IMMUNE_TYPES if t in types]
    # this time keep the X chromosome and exclude only the Y chromosome
    # Increased the window size to more genes to make the contrast between tumor and immune cells more obvious and less noisy
    cnv.tl.infercnv(adata, 
                    reference_key="celltype", 
                    reference_cat=immune_types,
                    exclude_chromosomes=('chrY',), 
                    window_size=WINDOW_SIZE, 
                    n_jobs=N_JOBS
                   )
    adata = jpacnv.cnv_plot(adata, resolution=resolution)
    print(f"Finished {batch}")

