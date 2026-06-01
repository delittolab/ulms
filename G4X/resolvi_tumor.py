#!/usr/bin/env python
# coding: utf-8

# # Looking at the resolVI embedding of the ULMS tumor cells only
# - Subsetting resolVI for tumor cells - manually annotated based off the scVIVA embedding, now mapping back to resolVI embedding
# - Did not retrain a tumor-only model since resolVI needs spatial information, so just recalcuated the neighbors graph and UMAP from the original embedding

# In[12]:


import sys
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import re

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi


# In[3]:


# version control
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)

plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.ioff()
sc.settings.autoshow = False
sc.settings.n_jobs = -1  # Use all available cores


# In[6]:


CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

SCVIVA_ANN_DIR = PARENT_DIR / 'scviva_tumor'

output_dir = jpascvi.create_output_dir(PARENT_DIR, 'resolvi_tumor', change_dir=True)


# In[7]:


jpa_markers = jpascvi.import_markers((PARENT_DIR / 'ref/jpa_g4x_breast_panel.csv'), output_type='dict')
jpa_markers = {key: value for key, value in jpa_markers.items() if key != 'Plasma_cell'} # JCHAIN and IGHG1 not in this segmentation run
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


# # Import data

# In[9]:


# load in the annotated scVIVA object for the tumor subset
print(f"Loading annotated scVIVA object for the tumor subset from {SCVIVA_ANN_DIR}")
adata = sc.read_h5ad(SCVIVA_ANN_DIR / 'tumor_annotated.h5ad')
print(adata)


# In[13]:


# Clean up
# Drop obs columns matching patterns
pattern = re.compile(r'^(leiden\d|_scvi)')
adata.obs = adata.obs[[c for c in adata.obs.columns if not pattern.match(c)]]

# Drop uns keys matching patterns
uns_pattern = re.compile(r'^(leiden\d|dendrogram_leiden|_scvi)')
static_drops = {'log1p', 'neighbors', 'rank_genes_groups', 'umap'}
for key in list(adata.uns.keys()):
    if uns_pattern.match(key) or key in static_drops:
        del adata.uns[key]

del adata.obsp['connectivities']
del adata.obsp['distances']
del adata.obsm['X_umap']

print(adata)


# In[ ]:


# Make sure to use the X_resolVI embedding
sc.pp.neighbors(adata, use_rep="X_resolVI")
sc.tl.umap(adata, min_dist=0.3)


# In[ ]:


# saving the anndata now that umap has been computed
adata.write_h5ad(output_dir / 'resolvi_tumor.h5ad')


# # Clustering and Differential Expression

# In[ ]:


sc.pl.umap(adata, color=['Section', 'Patient'], frameon=False, save='section_and_patient.png')
sc.pl.umap(adata, color=['component', 'volume', 
                         'surface_area', 'n_genes_by_counts', 
                         'log1p_n_genes_by_counts', 'total_counts', 
                         'log1p_total_counts', 'n_counts', 'n_genes',], frameon=False, save='qc.png')


# In[ ]:


jpascvi.featureplot(adata, jpa_markers, save='resolvi.png', use_raw=False, vmax='p98') # False to allow scaled values to be plotted from adata.X
jpascvi.featureplot(adata, jpa_markers, save='corr_resolvi.png', layer="generated_expression", vmax='p98')
jpascvi.featureplot(adata, jpa_markers, save='norm_resolvi.png', layer="X_normalized_resolVI", vmax='p98')


# In[ ]:


markers = ['MYH11', 'MYLK', 'DES', 'TAGLN', 'ESR1', 'COL1A1', 'SDC1', 'SOX10']
sc.pl.umap(adata, color=markers, size=0.2, save='ulms_feature_plot.png', vmax='p98')

markers = ['VEGFA', 'SLC2A1']
sc.pl.umap(adata, color=markers, size=0.2, save='ischemia_feature_plot.png', vmax='p98')

markers = ['VWF', 'PECAM1', 'COL1A1', 'LUM', 'CD68', 'CD163', 'RGS5', 'PDGFRB', 'CD2', 'IL7R']
sc.pl.umap(adata, color=markers, size=0.2, save='nontumor_feature_plot.png', vmax='p98')

markers = ['ESR1', 'PGR', 'AR']
sc.pl.umap(adata, color=markers, save='hormononal_feature_plot.png', size=0.2, vmax='p98')

markers = ['LAG3', 'PDCD1']
sc.pl.umap(adata, color=markers, save='immune_checkpoint_feature_plot.png', size=0.2, vmax='p98')


# In[ ]:


# Leiden clustering at multiple resolutions and differential gene expression
for resolution in resolutions:
    str_res = str(resolution).replace('.', '_')
    leiden_key = 'leiden' + str_res
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key)
    jpascvi.plot_umap(adata, resolution)
    jpascvi.sc_degs(adata, resolution, use_rep='X_resolVI', plots=['dotplot'])

# Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
jpascvi.cluster_stats(adata, resolutions, scores = ['Calinski-Harabasz', 'Davies-Bouldin'], rep='X_resolVI')


# In[ ]:


adata.write_h5ad(output_dir / 'resolvi_tumor_clustered.h5ad')

