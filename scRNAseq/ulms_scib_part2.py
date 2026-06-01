#!/usr/bin/env python
# coding: utf-8

# # This second notebook does the scib benchmarking. Make sure to run in environment with python 3.10 or higher.

# In[ ]:


from pathlib import Path
import os
import sys
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import harmonypy as hm
import scvi

# version control
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("harmonypy:", hm.__version__)

N_JOBS=16


# In[ ]:


def create_output_dir(master_dir, sub_dir_name, change_dir=False):
    '''
    Create an output directory as a subdirectory 'sub_dir_name' string within a parent directory master_dir
    Will not overwrite the files within that directory
    2025-05-28 moved to functions file
    2025-08-29 added change_dir
    '''
    output_dir = master_dir / sub_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Created output directory {output_dir}')
    if change_dir:
        os.chdir(output_dir)
        sc.settings.figdir = output_dir
        print(f'Default output directory changed to {output_dir}')
    return output_dir


# In[ ]:


# Set up input and output directories
CURRENT_DIR = Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent
print(PROJECT_DIR)

SCIB_DIR = create_output_dir(PROJECT_DIR, 'scib', change_dir=True)


# In[ ]:


adata = sc.read_h5ad('adata_hvg2000_beforescib.h5ad')
print(adata)


# In[ ]:


sc.tl.pca(adata)
adata.obsm["Unintegrated"] = adata.obsm["X_pca"]


# In[ ]:


# Get PCs from the AnnData object
pcs = adata.obsm['X_pca']
print(pcs.shape)  # (n_cells, n_pcs)

# Run Harmony on the PCA embedding. Use both batch and sample
harmony_out = hm.run_harmony(data_mat=pcs, meta_data=adata.obs, vars_use=["batch", "sample"])

# Store corrected PCs back in the AnnData object
adata.obsm['Harmony'] = harmony_out.Z_corr


# In[ ]:


# correcting for sample and batch
# Assumed that batch effect is primarily from the batch variable
# We have already run scVI so no need to run it again
model_dir = PROJECT_DIR / 'scvi_high_quality'
model = scvi.model.SCVI.load(model_dir, prefix='scVI')
print(model)

adata.obsm["scVI"] = model.get_latent_representation()


# In[ ]:


bm = Benchmarker(
    adata,
    batch_key="batch",
    label_key="celltype",
    bio_conservation_metrics=BioConservation(),
    batch_correction_metrics=BatchCorrection(),
    embedding_obsm_keys=["Unintegrated", "Harmony", "scVI"],
    n_jobs=N_JOBS,
)
bm.benchmark()


# In[ ]:


bm.plot_results_table(show=False, save_dir=SCIB_DIR)


# In[ ]:


bm.plot_results_table(min_max_scale=False, show=False, save_dir=SCIB_DIR)


# In[ ]:


from rich import print
df = bm.get_results(min_max_scale=False)
print(df)


# In[ ]:


df.transpose()


# In[ ]:


csv_path = Path(SCIB_DIR / 'scib.csv')
df.to_csv(csv_path, index=False)


# In[ ]:


adata.write_h5ad('post_scib.h5ad')

