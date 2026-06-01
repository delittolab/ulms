#!/usr/bin/env python
# coding: utf-8

# # scVI integration of only the T cells, B cells, Plasma cells, and NK cells for the revision
# - removing sample05 (spleen)
# - removing cluster that looks like tumor doublets

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
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

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

DATA_DIR = PROJECT_DIR / 'objects'
print(DATA_DIR)

output_dir = jpascvi.create_output_dir(PROJECT_DIR, 'lymphoid_wo5_wodoublet', change_dir=True)


# # Subset

# In[5]:


# reload the previously annotated raw object
path = DATA_DIR / "annotated_raw_counts.h5ad"
adata = sc.read_h5ad(path)
adata


# In[6]:


np.unique(adata.obs['celltype'])


# In[7]:


# subset lymphoid cells only
lymphoid_types = ['T_and_NK', 'B', 'Plasma']
adata = adata[adata.obs['celltype'].isin(lymphoid_types)].copy()
adata


# In[8]:


#remove sample05
adata = adata[adata.obs["sample"] != "Sample05"].copy()
adata


# In[9]:


# load the previously clustered lymphoid anndata
path = PROJECT_DIR / "lymphoid_wo5/scVI_clusteredadata.h5ad"
ann = sc.read_h5ad(path)
ann


# In[10]:


sc.pl.umap(ann, color=['leiden0_3', 'CALD1'], groups='1', size=5.0)


# In[11]:


# remove the doublet cluster from the lymphoid adata
ann = ann[ann.obs['leiden0_3'] != '1'].copy()
ann


# In[12]:


adata = adata[ann.obs.index].copy()
adata


# In[13]:


del ann


# In[14]:


counts_by_batch = adata.obs.batch.value_counts()
counts_by_batch = counts_by_batch.to_dict()
print(counts_by_batch)

batch_labels = list(counts_by_batch.keys())
counts = list(counts_by_batch.values())
plt.figure(figsize=(10, 5))
plt.bar(batch_labels, counts)

# Add labels and title
plt.xlabel('Batch')
plt.ylabel('Counts')
plt.title('Counts Per Batch in Lymphoid Cells')
plt.xticks(rotation='vertical')

# Display the plot
plt.show()
plt.savefig('counts_by_batch.png')
plt.close()


# In[15]:


counts_by_sample = adata.obs['sample'].value_counts()
counts_by_sample = counts_by_sample.to_dict()
print(counts_by_sample)

sample_labels = list(counts_by_sample.keys())
counts = list(counts_by_sample.values())
plt.figure(figsize=(10, 5))
plt.bar(sample_labels, counts)

# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Counts')
plt.title('Counts Per Sample in Lymphoid Cells')
plt.xticks(rotation='vertical')

# Display the plot
plt.show()
plt.savefig('counts_by_sample.png')
plt.close()


# # Prepare for scVI

# In[16]:


adata.layers["counts"] = adata.X.copy() # this layer will contain the raw counts
sc.pp.normalize_total(adata) # normalize X to the median total counts
sc.pp.log1p(adata) # logarithmize X
adata.raw = adata # full dimension normalized logtransformed raw data


# In[17]:


# Trying to make a histogram of counts before HVG selection
counts = adata.layers['counts'].sum(axis=1).tolist()
counts = [item for sublist in counts for item in sublist]
print(len(counts))
sns.histplot(counts)
plt.xlim(xmin=0)
plt.xlim(xmax=50000)
plt.xticks(rotation='vertical')
plt.xlabel('Counts per Cell')
plt.ylabel('Frequency')
plt.title('Total Counts Per Cell in Lymphoid Cells')
plt.xticks(rotation='vertical')
plt.show()
plt.savefig('counts_beforehvgs_histogram.png')
plt.close()


# In[18]:


print(f"Number of genes before filtering: {adata.n_vars}")
sc.pp.filter_genes(adata, min_cells=3)
print(f"Number of genes after filtering: {adata.n_vars}")


# In[20]:


print(f"Number of genes before HVG selection: {adata.n_vars}")
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    batch_key="batch",
    n_top_genes=2000,
    subset=True,
    layer="counts",
    span=0.8,
)
print(f"Number of genes after HVG selection: {adata.n_vars}")


# In[21]:


# Trying to make a histogram of counts after HVG selection
counts = adata.layers['counts'].sum(axis=1).tolist()
counts = [item for sublist in counts for item in sublist]
print(len(counts))
sns.histplot(counts)
plt.xlim(xmax=50000)
plt.xlabel('Counts per Cell')
plt.ylabel('Frequency')
plt.title('Total Counts Per Cell in Tumor Cells')
plt.xticks(rotation='vertical')
plt.savefig('counts_afterhvgs_histogram.png')
plt.show()
plt.close()


# In[22]:


# Some cells may have zero HVG counts - this may mess up integration and differential expression calculation by creating a division by zero
print(f"Number of cells in anndata: {adata.n_obs}")
# Make sure to use the raw counts layer
low_counts = adata[adata.layers['counts'].sum(axis=1) < 1]
print(f"Number of cells with zero HVG counts: {low_counts.n_obs}")


# In[ ]:


# Find neighbors and UMAP prior to integration to get a baseline for batch effect
sc.tl.pca(adata)
sc.pp.neighbors(adata, key_added="X_pca")
sc.tl.umap(adata, min_dist=0.3, neighbors_key="X_pca")
sc.pl.umap(adata, neighbors_key="X_pca", color=["batch", "sample"], save='unintegrated.png')


# # scVI integration

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
lymphoid_markers = jpascvi.import_markers('/labs/delitto/james/ref/lymphoid.csv', output_type='dict')


# In[ ]:


sc._settings.ScanpyConfig.figdir = output_dir
sc._settings.ScanpyConfig.autoshow = False
sc._settings.ScanpyConfig.autosave = True

jpascvi.featureplot(adata, mmk_markers, neighbors_key="N_scVI")
jpascvi.featureplot(adata, jpa_markers, neighbors_key="N_scVI")
jpascvi.featureplot(adata, lymphoid_markers, neighbors_key="N_scVI")


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
    jpascvi.sc_degs(adata, resolution, use_rep='X_scVI', plots=['dotplot'])

    # dotplot
    sc.pl.dotplot(adata, lymphoid_markers, groupby=leiden_key, dendrogram=False,
                  swap_axes=True, use_raw=True, standard_scale="var", save=f'lymphoid_dp_{str_res}.png')
    # heatmap
    sc.pl.heatmap(adata, lymphoid_markers, groupby=leiden_key, 
                  standard_scale="var", dendrogram=False, save=f'lymphoid_hm_{str_res}.png')
    # matrix plot
    sc.pl.matrixplot(adata, lymphoid_markers, groupby=leiden_key, standard_scale="var", save=f'lymphoid_mp_{str_res}.png')

# Save adata with umap and leiden clustering
model.save(dir_path=output_dir, prefix='scVI_clustered', overwrite=True, save_anndata=True)


# In[ ]:


jpascvi.cluster_stats(adata, resolutions)


# # Making a raw counts object

# In[ ]:


# # reload the previously clustered and umapped object
# ad_path = output_dir / 'lymphoid_clusteredadata.h5ad'
# adata = sc.read_h5ad(ad_path)
# print(adata)
# print()

# # load the rawdata and filter for lymphoid cells only
# ad_path = '/oak/stanford/groups/longaker/ULMS/analysis_v3/objects/annotated_raw_counts.h5ad'
# rawdata = sc.read_h5ad(ad_path)
# print(rawdata)
# print()

# # subset the raw counts and transfer scVI embedding
# rawdata = rawdata[rawdata.obs['celltype'].isin(['T_and_NK_cells', 'B_cells', 'Plasma_cells'])]
# print(rawdata.obs.index.equals(adata.obs.index)) # check to make sure indices are the same
# print()
# rawdata.uns['N_scVI'] = adata.uns['N_scVI']
# rawdata.obsm['X_scVI'] = adata.obsm['X_scVI']
# rawdata.obsm['X_umap'] = adata.obsm['X_umap']
# rawdata.obsp['N_scVI_connectivities'] = adata.obsp['N_scVI_connectivities']
# rawdata.obsp['N_scVI_distances'] = adata.obsp['N_scVI_distances']
# rawdata.obs['leiden0_1'] = adata.obs.loc[rawdata.obs.index, 'leiden0_1']
# rawdata.obs['leiden0_2'] = adata.obs.loc[rawdata.obs.index, 'leiden0_2']
# rawdata.obs['leiden0_5'] = adata.obs.loc[rawdata.obs.index, 'leiden0_5']
# rawdata.obs['leiden0_6'] = adata.obs.loc[rawdata.obs.index, 'leiden0_6']
# rawdata.obs['leiden0_8'] = adata.obs.loc[rawdata.obs.index, 'leiden0_8']
# rawdata.obs['leiden1_0'] = adata.obs.loc[rawdata.obs.index, 'leiden1_0']
# del adata
# adata = rawdata
# del rawdata
# print(adata)


# In[ ]:


# adata.write_h5ad('/oak/stanford/groups/longaker/ULMS/analysis_v3/objects/lymphoid_raw_counts.h5ad')


# # Cell cycle scoring
# https://nbviewer.org/github/theislab/scanpy_usage/blob/master/180209_cell_cycle/cell_cycle.ipynb
# https://satijalab.org/seurat/articles/cell_cycle_vignette.html#assign-cell-cycle-scores

# In[ ]:


# # reload the previously clustered and umapped object
# ad_path = output_dir / 'lymphoid_clusteredadata.h5ad'
# adata = sc.read_h5ad(ad_path)
# print(adata)
# # need all the genes, not just HVGs
# rawdata = adata.raw.to_adata()
# print(rawdata)
# del adata


# In[ ]:


# # get the cell cycle genes from https://www.science.org/doi/10.1126/science.aad0501
# # download from https://www.dropbox.com/s/3dby3bjsaf5arrw/cell_cycle_vignette_files.zip?dl=1
# cell_cycle_genes = [x.strip() for x in open('/labs/delitto/james/ref/regev_lab_cell_cycle_genes.txt')]
# s_genes = cell_cycle_genes[:43]
# s_genes = [x for x in s_genes if x in rawdata.var_names]
# g2m_genes = cell_cycle_genes[43:]
# g2m_genes = [x for x in g2m_genes if x in rawdata.var_names]
# cell_cycle_genes = [x for x in cell_cycle_genes if x in rawdata.var_names]


# In[ ]:


# # Data is already log-transformed, but we still need to scale data to unit variance and zero mean in order to score genes
# sc.pp.scale(rawdata)


# In[ ]:


# sc.tl.score_genes_cell_cycle(rawdata, s_genes=s_genes, g2m_genes=g2m_genes)
# print(rawdata)


# In[ ]:


# sc.pl.umap(rawdata, neighbors_key='N_scVI', color=['leiden0_2', 'phase', 'S_score', 'G2M_score'], ncols=2, save='cell_cycle.png')


# In[ ]:


# del rawdata


# # Automated annotation with celltypist

# In[ ]:


# import celltypist as ct
# from celltypist import models
# print("celltypist:", ct.__version__)


# celltypist expects anndata normalized to 1e4 and log-transformed with all the genes, and I didn't do that, so let's recreate the anndata

# In[ ]:


# # reload the previously clustered and umapped object
# ad_path = output_dir / 'lymphoid_clusteredadata.h5ad'
# adata = sc.read_h5ad(ad_path)
# print(adata)
# print()

# # load the rawdata and filter for lymphoid cells only
# ad_path = '/oak/stanford/groups/longaker/ULMS/analysis_v3/objects/annotated_raw_counts.h5ad'
# rawdata = sc.read_h5ad(ad_path)
# print(rawdata)
# print()

# # subset the raw counts and transfer scVI embedding
# rawdata = rawdata[rawdata.obs['celltype'].isin(['T_and_NK_cells', 'B_cells', 'Plasma_cells'])]
# print(rawdata.obs.index.equals(adata.obs.index)) # check to make sure indices are the same
# print()
# rawdata.uns['N_scVI'] = adata.uns['N_scVI']
# rawdata.obsm['X_scVI'] = adata.obsm['X_scVI']
# rawdata.obsm['X_umap'] = adata.obsm['X_umap']
# rawdata.obsp['N_scVI_connectivities'] = adata.obsp['N_scVI_connectivities']
# rawdata.obsp['N_scVI_distances'] = adata.obsp['N_scVI_distances']
# rawdata.obs['leiden0_1'] = adata.obs.loc[rawdata.obs.index, 'leiden0_1']
# rawdata.obs['leiden0_2'] = adata.obs.loc[rawdata.obs.index, 'leiden0_2']
# rawdata.obs['leiden0_5'] = adata.obs.loc[rawdata.obs.index, 'leiden0_5']
# rawdata.obs['leiden0_6'] = adata.obs.loc[rawdata.obs.index, 'leiden0_6']
# rawdata.obs['leiden0_8'] = adata.obs.loc[rawdata.obs.index, 'leiden0_8']
# rawdata.obs['leiden1_0'] = adata.obs.loc[rawdata.obs.index, 'leiden1_0']
# del adata
# adata = rawdata
# del rawdata
# print(adata)


# In[ ]:


# adata.layers["counts"] = adata.X.copy() # this layer will contain the raw counts
# sc.pp.normalize_total(adata, target_sum=1e4) # normalize X to the median total counts
# sc.pp.log1p(adata) # logarithmize X
# adata.raw = adata # full dimension normalized logtransformed raw data


# In[ ]:


# model = models.Model.load(model='Immune_All_Low.pkl')
# model


# In[ ]:


# model.cell_types


# In[ ]:


# predictions = ct.annotate(adata, model=model, majority_voting=True, over_clustering='leiden0_6')


# In[ ]:


# predictions.predicted_labels


# In[ ]:


# adata = predictions.to_adata()
# adata.obs


# In[ ]:


# sc.pl.umap(adata, ncols=1,
#            neighbors_key="N_scVI", 
#            color=['majority_voting'], 
#            frameon=False, save='celltypist0_6_mv.png',)

