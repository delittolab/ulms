#!/usr/bin/env python
# coding: utf-8

# # Integrating all ULMS samples for the revision after removal of low-quality cells and RBCs - higher resolutions
# - This is after QC removed the low-quality cluster and after RBCs were removed.
# - Final model (before RBC and low-quality cluster removal)
# - This notebook adds higher leiden clustering resolutions

# In[ ]:


import os
import sys
import numpy as np
import scanpy as sc
import torch
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi


# In[ ]:


torch.cuda.is_available()


# In[ ]:


# version control
print("seaborn:", sns.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("scvi:", scvi.__version__)
scvi.settings.seed = 1234
sns.set_theme()
torch.set_float32_matmul_precision("high")
sc._settings.ScanpyConfig.n_jobs=-1


# In[ ]:


# Set up input and output directories
CURRENT_DIR = Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent
print(PROJECT_DIR)

DATA_DIR = PROJECT_DIR / 'scvi_high_quality'
print(DATA_DIR)

output_dir = jpascvi.create_output_dir(PROJECT_DIR, 'scvi_high_quality', change_dir=True)


# In[ ]:


adata = sc.read_h5ad(DATA_DIR / 'scVIadata.h5ad')
print(adata)
model = scvi.model.SCVI.load(DATA_DIR, prefix='scVI', adata=adata)
print(model)


# In[ ]:


jpa_markers = jpascvi.import_markers('/labs/delitto/james/ref/jpa_sc_markers.csv', output_type='dict')
mmk_markers = jpascvi.import_markers('/labs/delitto/james/ref/mmk_sc_markers.csv', output_type='dict')
djd_markers = jpascvi.import_markers('/labs/delitto/ulms_cellbender/ref/markers_4.csv', output_type='dict')


# In[ ]:


sc._settings.ScanpyConfig.figdir = output_dir
sc._settings.ScanpyConfig.autoshow = False
sc._settings.ScanpyConfig.autosave = True

resolutions = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
for resolution in resolutions:
    print("Clustering with resolution " + str(resolution))
    str_res = str(resolution).replace('.', '_')
    leiden_key = "leiden" + str_res
    sc.tl.leiden(adata, neighbors_key="N_scVI", key_added=leiden_key, resolution=resolution, flavor="igraph", n_iterations=2)
    jpascvi.plot_umap(adata, resolution, neighbors_key="N_scVI")
    jpascvi.scvi_degs(adata, model, resolution, djd_markers, rep_key="X_scVI", norm_layer="scvi_normalized")
    jpascvi.sc_degs(adata, resolution, use_rep='X_scVI')

# Save adata with umap and leiden clustering
model.save(dir_path=output_dir, prefix='scVI_clustered_highres', overwrite=True, save_anndata=True)


# In[ ]:


# calculating clustering statistics
jpascvi.cluster_stats(adata, resolutions)

