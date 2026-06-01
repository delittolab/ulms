#!/usr/bin/env python
# coding: utf-8

# # Integrating all ULMS samples for the revision
# - Inputs: all preprocessed samples after QC and doublet removal
# - Initial model (before RBC and low-quality cluster removal)

# In[1]:


import os
import sys
import numpy as np
import scanpy as sc
import torch
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata as ad
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi


# In[2]:


torch.cuda.is_available()


# In[3]:


# version control
print("seaborn:", sns.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("scvi:", scvi.__version__)
scvi.settings.seed = 1234
sns.set_theme()
torch.set_float32_matmul_precision("high")


# In[4]:


# Set up input and output directories
CURRENT_DIR = Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent
print(PROJECT_DIR)

DATA_DIR = PROJECT_DIR / 'preprocessed'
print(DATA_DIR)

output_dir = jpascvi.create_output_dir(PROJECT_DIR, 'scvi_low_quality', change_dir=True)


# # Load, concatenate, and prepare anndata for scVI

# In[5]:


# load preprocessed anndatas
adata_list = jpascvi.import_data(DATA_DIR)
adata_list.sort(key=lambda x: np.unique(x.uns['filename']))

for adata in adata_list:
    print(list(np.unique(adata.obs['batch'])))
    print(list(np.unique(adata.obs['sample'])))
    print(adata.uns['filename'])
    print(adata)
    print()  


# In[6]:


len(adata_list)


# In[7]:


# concatenate the samples
adata = ad.concat(adata_list, join="inner")
del adata_list
adata


# In[ ]:


# # Make a raw counts object of concatenated preprocessed anndatas
# adata.write_h5ad('objects/ulms_raw.h5ad')


# In[8]:


adata.layers["counts"] = adata.X.copy() # this layer will contain the raw counts
sc.pp.normalize_total(adata) # normalize X to the median total counts
sc.pp.log1p(adata) # logarithmize X
adata.raw = adata # full dimension normalized logtransformed raw data


# In[9]:


# Calculate HVGs. We use 2000 per the Zappia et al feature selection paper
print(f"Number of genes before HVG selection: {adata.n_vars}")
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=2000,
    layer="counts",
    batch_key="batch",
    subset=True,
)
print(f"Number of genes after HVG selection: {adata.n_vars}")


# In[10]:


# Some cells may have zero HVG counts - this may mess up integration and differential expression calculation by creating a division by zero
print(f"Number of cells in anndata: {adata.n_obs}")
# Make sure to use the raw counts layer
low_counts = adata[adata.layers['counts'].sum(axis=1) < 1]
print(f"Number of cells with zero HVG counts: {low_counts.n_obs}")
# Decided not to remove the few cells with zero HVG counts since we will remove the low-quality cell cluster and RBCs later.


# In[ ]:


# Find neighbors and UMAP prior to integration to get a baseline for batch effect
sc.tl.pca(adata)
sc.pp.neighbors(adata, key_added="X_pca")
sc.tl.umap(adata, min_dist=0.3, neighbors_key="X_pca")
sc.pl.umap(adata, neighbors_key="X_pca", color=["batch", "sample"], ncols=1, save='_unintegrated.png')


# # Train the model

# In[ ]:


# correcting for sample and batch
# Assumed that batch effect is primarily from the batch variable
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch", categorical_covariate_keys=['sample',])
model = scvi.model.SCVI(adata)
print(model)

# Train the vae with early stopping for the default number of epochs
scvi.settings.seed = 1234
model.train(check_val_every_n_epoch=1,
            early_stopping=True,
            early_stopping_patience=20, # how many epochs of no change are tolerated
            early_stopping_monitor="elbo_validation")

# Check training
train_test_results = model.history["elbo_train"]
train_test_results["elbo_validation"] = model.history["elbo_validation"]
train_test_results.plot()
plt.savefig('elbo_plot.png')
plt.close()


# # Extract the embeddings

# In[ ]:


adata.obsm["X_scVI"] = model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scVI", key_added="N_scVI")
sc.tl.umap(adata, min_dist=0.3, neighbors_key="N_scVI")
adata.layers["scvi_normalized"] = model.get_normalized_expression()
# saving the model and anndata now that umap has been computed
model.save(dir_path=output_dir, prefix='scVI', overwrite=True, save_anndata=True)


# # Feature plots

# In[ ]:


jpa_markers = jpascvi.import_markers('/labs/delitto/james/ref/jpa_sc_markers.csv', output_type='dict')
mmk_markers = jpascvi.import_markers('/labs/delitto/james/ref/mmk_sc_markers.csv', output_type='dict')
djd_markers = jpascvi.import_markers('/labs/delitto/ulms_cellbender/ref/markers_4.csv', output_type='dict')


# In[ ]:


sc._settings.ScanpyConfig.figdir = output_dir
sc._settings.ScanpyConfig.autoshow = False
sc._settings.ScanpyConfig.autosave = True

jpascvi.featureplot(adata, mmk_markers, neighbors_key="N_scVI")
jpascvi.featureplot(adata, jpa_markers, neighbors_key="N_scVI")


# # Main loop: clustering

# In[ ]:


resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for resolution in resolutions:
    print("Clustering with resolution " + str(resolution))
    str_res = str(resolution).replace('.', '_')
    leiden_key = "leiden" + str_res
    sc.tl.leiden(adata, neighbors_key="N_scVI", key_added=leiden_key, resolution=resolution, flavor="igraph", n_iterations=2)
    jpascvi.plot_umap(adata, resolution, neighbors_key="N_scVI")
    jpascvi.scvi_degs(adata, model, resolution, djd_markers, rep_key="X_scVI", norm_layer="scvi_normalized")
    jpascvi.sc_degs(adata, resolution, use_rep='X_scVI')

# Save adata with umap and leiden clustering
model.save(dir_path=output_dir, prefix='scVI_clustered', overwrite=True, save_anndata=True)


# In[ ]:


# QC umap
sc.pl.umap(adata, 
           neighbors_key="N_scVI", 
           color=['n_genes_by_counts', 'log1p_n_genes_by_counts', 
                   'total_counts', 'log1p_total_counts', 'n_counts', 
                   'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 
                   'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 
                   'doublet_score', 'doublet',], 
           frameon=False, ncols=4, save='qc_umap.png',)

sc.pl.umap(adata, neighbors_key='N_scVI', color='batch', frameon=False, save='batch.png')
sc.pl.umap(adata, neighbors_key='N_scVI', color='sample', frameon=False, save='sample.png')
sc.pl.umap(adata, neighbors_key='N_scVI', color=['batch', 'sample', 'CALD1', 'CD3E', 'CD68', 'PECAM1'], frameon=False, ncols=2, save='CALD1.png')


# In[ ]:


# calculating clustering statistics
jpascvi.cluster_stats(adata, resolutions)


# # Reload data, reassess quality control, and remove poor quality cell clusters

# In[ ]:


# path = Path(output_dir / 'SCVI_bs_umapped_clusteredadata.h5ad')
# adata = sc.read_h5ad(path)
# adata


# In[ ]:


# sc.pl.umap(adata, neighbors_key='N_scVI', color='leiden0_8', frameon=False, save='res0_8.png')
# sc.pl.umap(adata, neighbors_key='N_scVI', color='leiden0_8', legend_loc="on data", frameon=False, save='res0_8_labeled.png')


# In[ ]:


# # Annotate at resolution 0.8 to separate out RBCs

# leiden_map = {
#     "0" : "Tumor",
#     "1" : "Tumor",
#     "2" : "Tumor",
#     "3" : "Tumor",
#     "4" : "Tumor",
#     "5" : "Tumor",
#     "6" : "Tumor",
#     "7" : "Tumor",
#     "8" : "Tumor",
#     "9" : "RBCs",
#     "10" : "ECs",
#     "11" : "Low_quality",
#     "12" : "Fibroblasts",
#     "13" : "Tumor",
#     "14" : "Bowel",
#     "15" : "Pericytes",
#     "16" : "Tumor",
#     "17" : "Myeloid",
#     "18" : "Myeloid",
#     "19" : "Mast_cells",
#     "20" : "Bowel",
#     "21" : "T_and_NK_cells",
#     "22" : "T_and_NK_cells",
#     "23" : "B_cells",
#     "24" : "T_and_NK_cells",
#     "25" : "T_and_NK_cells",
#     "26" : "T_and_NK_cells",
#     "27" : "T_and_NK_cells",
#     "28" : "T_and_NK_cells",
#     "29" : "Plasma_cells_and_pDCs",
#     "30" : "B_cells",
#     "31" : "Tumor",
#     "32" : "Tumor",
#     "33" : "Neutrophils",
#     "34" : "Tumor",
#     "35" : "Bowel",
#     "36" : "Lung",
# }

# adata.obs['coarse_celltype'] = adata.obs['leiden0_8'].map(leiden_map)


# In[ ]:


# sc.pl.umap(adata, neighbors_key='N_scVI', 
#            color=['coarse_celltype'], 
#            frameon=False, save='coarse_celltype.png')


# # Smooth qc metrics by calculating the median value across a given cluster and plotting that

# In[ ]:


# qc = ['log1p_n_genes_by_counts', 'log1p_total_counts', 'pct_counts_mt',]
# for metric in qc:
#     str_metric = 'median_' + metric
#     adata.obs[str_metric] = adata.obs.groupby('leiden0_8', observed=True)[metric].transform('median')
# print(adata.obs)


# In[ ]:


# sc.pl.umap(adata, neighbors_key='N_scVI', 
#            color=['median_log1p_n_genes_by_counts', 'median_log1p_total_counts', 'median_pct_counts_mt',], 
#            frameon=False, save='qc_median_res0_8.png')


# In[ ]:


# median_genes = adata.obs.groupby('leiden0_8', observed=True)['median_log1p_n_genes_by_counts'].unique().reset_index().astype(float)
# median_genes['leiden0_8'] = median_genes['leiden0_8'].astype(int)
# median_genes


# In[ ]:


# # Create a bar plot for genes by counts
# plt.figure(figsize=(12, 6))
# sns.barplot(x='leiden0_8', y='median_log1p_n_genes_by_counts', data=median_genes)
# plt.title('Median genes by counts for each cluster')
# plt.ylabel('median log1p n genes by counts')
# plt.tight_layout()  # Adjust layout to prevent clipping
# plt.savefig('median_log1p_n_genes_by_counts_barplot.png')
# plt.show()


# In[ ]:


# # Create a box plot for genes by counts
# plt.figure(figsize=(8, 6))
# sns.boxplot(y='median_log1p_n_genes_by_counts', data=median_genes)
# plt.title('Median genes by counts for each cluster')
# plt.ylabel('median log1p n genes by counts')
# plt.tight_layout()  # Adjust layout to prevent clipping
# plt.savefig('median_log1p_n_genes_by_counts_boxplot.png')
# plt.show()


# In[ ]:


# median_counts = adata.obs.groupby('leiden0_5', observed=True)['median_log1p_total_counts'].unique().reset_index().astype(float)
# median_counts['leiden0_5'] = median_genes['leiden0_5'].astype(int)
# median_counts


# In[ ]:


# # Create a bar plot for counts
# plt.figure(figsize=(12, 6))
# sns.barplot(x='leiden0_8', y='median_log1p_total_counts', data=median_counts)
# plt.title('Median total counts for each cluster')
# plt.ylabel('median log1p total counts')
# plt.tight_layout()  # Adjust layout to prevent clipping
# plt.savefig('median_log1p_total_counts_barplot.png')
# plt.show()


# In[ ]:


# # Create a box plot for counts
# plt.figure(figsize=(8, 6))
# sns.boxplot(y='median_log1p_total_counts', data=median_counts)
# plt.title('Median total counts for each cluster')
# plt.ylabel('median log1p total counts')
# plt.tight_layout() # Adjust layout to prevent clipping
# plt.savefig('median_log1p_total_counts_boxplot.png')
# plt.show()


# In[ ]:


# median_pct_mt = adata.obs.groupby('leiden0_8', observed=True)['median_pct_counts_mt'].unique().reset_index().astype(float)
# median_pct_mt['leiden0_8'] = median_pct_mt['leiden0_8'].astype(int)
# median_pct_mt


# In[ ]:


# # Create a bar plot for pct_mt
# plt.figure(figsize=(12, 6))
# sns.barplot(x='leiden0_8', y='median_pct_counts_mt', data=median_pct_mt)
# plt.title('Median percentage of mitochondrial counts for each cluster')
# plt.ylabel('median pct counts mt')
# plt.tight_layout()  # Adjust layout to prevent clipping
# plt.savefig('median_pct_counts_mt_barplot.png')
# plt.show()


# In[ ]:


# # Create a box plot for pct mt
# plt.figure(figsize=(8, 6))
# sns.boxplot(y='median_pct_counts_mt', data=median_pct_mt)
# plt.title('Median percentage of mitochondrial counts for each cluster')
# plt.ylabel('median pct counts mt')
# plt.tight_layout()  # Adjust layout to prevent clipping
# plt.savefig('median_pct_counts_mt_boxplot.png')
# plt.show()


# # Alternatively, remove only low-quality cells and RBCs

# In[ ]:


# path = Path(output_dir / 'SCVI_bs_umapped_clusteredadata.h5ad')
# adata = sc.read_h5ad(path)
# adata


# In[ ]:


# # Annotate at resolution 0.8 to separate out RBCs - this is a copy of the cell above

# leiden_map = {
#     "0" : "Tumor",
#     "1" : "Tumor",
#     "2" : "Tumor",
#     "3" : "Tumor",
#     "4" : "Tumor",
#     "5" : "Tumor",
#     "6" : "Tumor",
#     "7" : "Tumor",
#     "8" : "Tumor",
#     "9" : "RBCs",
#     "10" : "ECs",
#     "11" : "Low_quality",
#     "12" : "Fibroblasts",
#     "13" : "Tumor",
#     "14" : "Bowel",
#     "15" : "Pericytes",
#     "16" : "Tumor",
#     "17" : "Myeloid",
#     "18" : "Myeloid",
#     "19" : "Mast_cells",
#     "20" : "Bowel",
#     "21" : "T_and_NK_cells",
#     "22" : "T_and_NK_cells",
#     "23" : "B_cells",
#     "24" : "T_and_NK_cells",
#     "25" : "T_and_NK_cells",
#     "26" : "T_and_NK_cells",
#     "27" : "T_and_NK_cells",
#     "28" : "T_and_NK_cells",
#     "29" : "Plasma_cells_and_pDCs",
#     "30" : "B_cells",
#     "31" : "Tumor",
#     "32" : "Tumor",
#     "33" : "Neutrophils",
#     "34" : "Tumor",
#     "35" : "Bowel",
#     "36" : "Lung",
# }

# adata.obs['coarse_celltype'] = adata.obs['leiden0_8'].map(leiden_map)


# In[ ]:


# adata = adata[~adata.obs['coarse_celltype'].isin(["RBCs", "Low_quality"])].copy()
# print(adata)


# In[ ]:


# adata_afterqc = adata


# In[ ]:


# # reload ULMS data - preprocessed anndatas
# ulms_adata = sc.read_h5ad('/oak/stanford/groups/longaker/ULMS/redo_analysis/objects/raw.h5ad')
# print(ulms_adata)

# ulms_adata = ulms_adata[adata_afterqc.obs.index].copy()
# print(ulms_adata)


# In[ ]:


# ulms_adata.write_h5ad('ulms_afterqc_noRBC.h5ad')

