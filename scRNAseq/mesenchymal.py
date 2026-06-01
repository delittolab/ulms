#!/usr/bin/env python
# coding: utf-8

# # Integration of all mesenchymal cells for the revision
# This version filters out some genes but also increases the span

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

output_dir = jpascvi.create_output_dir(PROJECT_DIR, 'mesenchymal', change_dir=True)


# # Subset

# In[ ]:


# # reload the previously annotated raw object
# path = DATA_DIR / "annotated_raw_counts.h5ad"
# adata = sc.read_h5ad(path)
# adata


# In[ ]:


# np.unique(adata.obs['celltype'])


# In[ ]:


# # subset mesenchymal cells only
# adata = adata[adata.obs['celltype'] == 'Mesenchymal'].copy()
# adata


# In[ ]:


# counts_by_batch = adata.obs.batch.value_counts()
# counts_by_batch = counts_by_batch.to_dict()
# print(counts_by_batch)

# batch_labels = list(counts_by_batch.keys())
# counts = list(counts_by_batch.values())
# plt.figure(figsize=(10, 5))
# plt.bar(batch_labels, counts)

# # Add labels and title
# plt.xlabel('Batch')
# plt.ylabel('Counts')
# plt.title('Counts Per Batch in Mesenchymal Cells')
# plt.xticks(rotation='vertical')

# # Display the plot
# plt.savefig('counts_by_batch.png')


# In[ ]:


# counts_by_sample = adata.obs['sample'].value_counts()
# counts_by_sample = counts_by_sample.to_dict()
# print(counts_by_sample)

# sample_labels = list(counts_by_sample.keys())
# counts = list(counts_by_sample.values())
# plt.figure(figsize=(10, 5))
# plt.bar(sample_labels, counts)

# # Add labels and title
# plt.xlabel('Sample')
# plt.ylabel('Counts')
# plt.title('Counts Per Sample in Mesenchymal Cells')
# plt.xticks(rotation='vertical')

# # Display the plot
# plt.savefig('counts_by_sample.png')


# # Prepare for scVI

# In[ ]:


# print(f"Number of genes before filtering: {adata.n_vars}")
# sc.pp.filter_genes(adata, min_cells=3)
# print(f"Number of genes after filtering: {adata.n_vars}")


# In[ ]:


# adata.layers["counts"] = adata.X.copy() # this layer will contain the raw counts
# sc.pp.normalize_total(adata) # normalize X to the median total counts
# sc.pp.log1p(adata) # logarithmize X
# adata.raw = adata # full dimension normalized logtransformed raw data


# In[ ]:


# # increasing the span to stop the skmisc loess function from erroring out
# print(f"Number of genes before HVG selection: {adata.n_vars}")
# sc.pp.highly_variable_genes(
#     adata,
#     flavor="seurat_v3",
#     n_top_genes=2000,
#     batch_key="batch",
#     subset=True,
#     layer="counts",
#     span=0.4
# )
# print(f"Number of genes after HVG selection: {adata.n_vars}")


# In[ ]:


# # Some cells may have zero HVG counts - this may mess up integration and differential expression calculation by creating a division by zero
# print(f"Number of cells in anndata: {adata.n_obs}")
# # Make sure to use the raw counts layer
# low_counts = adata[adata.layers['counts'].sum(axis=1) < 1]
# print(f"Number of cells with zero HVG counts: {low_counts.n_obs}")


# In[ ]:


# # Find neighbors and UMAP prior to integration to get a baseline for batch effect
# sc.tl.pca(adata)
# sc.pp.neighbors(adata, key_added="X_pca")
# sc.tl.umap(adata, min_dist=0.3, neighbors_key="X_pca")
# sc.pl.umap(adata, neighbors_key="X_pca", color=["batch", "sample"], ncols=1, save='_unintegrated.png')


# # Train the scVI model

# In[ ]:


# # correcting for sample and batch
# # Assumed that batch effect is primarily from the batch variable
# scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch", categorical_covariate_keys=['sample',])
# model = scvi.model.SCVI(adata)
# print(model)

# # Train the vae with early stopping for the default number of epochs
# scvi.settings.seed = 1234
# model.train(check_val_every_n_epoch=1,
#             early_stopping=True,
#             early_stopping_patience=20, # how many epochs of no change are tolerated
#             early_stopping_monitor="elbo_validation")

# # Check training
# train_test_results = model.history["elbo_train"]
# train_test_results["elbo_validation"] = model.history["elbo_validation"]
# train_test_results.plot()
# plt.savefig('elbo_plot.png')
# plt.close()


# In[ ]:


# adata.obsm["X_scVI"] = model.get_latent_representation()
# sc.pp.neighbors(adata, use_rep="X_scVI", key_added="N_scVI")
# sc.tl.umap(adata, min_dist=0.3, neighbors_key="N_scVI")
# adata.layers["scvi_normalized"] = model.get_normalized_expression()
# # saving the model and anndata now that umap has been computed
# model.save(dir_path=output_dir, prefix='scVI', overwrite=True, save_anndata=True)


# # Feature plots

# In[ ]:


jpa_markers = jpascvi.import_markers('/labs/delitto/james/ref/jpa_sc_markers.csv', output_type='dict')
mmk_markers = jpascvi.import_markers('/labs/delitto/james/ref/mmk_sc_markers.csv', output_type='dict')
djd_markers = jpascvi.import_markers('/labs/delitto/ulms_cellbender/ref/markers_4.csv', output_type='dict')


# In[ ]:


sc._settings.ScanpyConfig.figdir = output_dir
sc._settings.ScanpyConfig.autoshow = False
sc._settings.ScanpyConfig.autosave = True


# In[ ]:


# jpascvi.featureplot(adata, mmk_markers, neighbors_key="N_scVI")
# jpascvi.featureplot(adata, jpa_markers, neighbors_key="N_scVI")


# In[ ]:


# # QC umap
# sc.pl.umap(adata, 
#            neighbors_key="N_scVI", 
#            color=['n_genes_by_counts', 'log1p_n_genes_by_counts', 
#                    'total_counts', 'log1p_total_counts', 'n_counts', 
#                    'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 
#                    'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 
#                    'doublet_score', 'doublet',], 
#            frameon=False, ncols=4, save='qc_umap.png',)

# sc.pl.umap(adata, neighbors_key='N_scVI', color='batch', frameon=False, save='batch.png')
# sc.pl.umap(adata, neighbors_key='N_scVI', color='sample', frameon=False, save='sample.png')
# sc.pl.umap(adata, neighbors_key='N_scVI', color=['batch', 'sample', 'CALD1', 'LUM', 'RGS5', 'PECAM1'], frameon=False, ncols=2, save='CALD1.png')


# # Main loop: clustering

# In[ ]:


# resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# for resolution in resolutions:
#     print("Clustering with resolution " + str(resolution))
#     str_res = str(resolution).replace('.', '_')
#     leiden_key = "leiden" + str_res
#     sc.tl.leiden(adata, neighbors_key="N_scVI", key_added=leiden_key, resolution=resolution, flavor="igraph", n_iterations=2)
#     jpascvi.plot_umap(adata, resolution, neighbors_key="N_scVI")
#     jpascvi.scvi_degs(adata, model, resolution, djd_markers, rep_key="X_scVI", norm_layer="scvi_normalized")
#     jpascvi.sc_degs(adata, resolution, use_rep='X_scVI', plots=['dotplot'])

# # Save adata with umap and leiden clustering
# model.save(dir_path=output_dir, prefix='scVI_clustered', overwrite=True, save_anndata=True)


# In[ ]:


# jpascvi.cluster_stats(adata, resolutions)


# # Adding resolutions
# - Trying to isolate the adipocytes and any RBCs

# In[5]:


ad_path = Path(output_dir / 'scVI_clusteredadata.h5ad')
adata = sc.read_h5ad(ad_path)
print(adata)
model = scvi.model.SCVI.load(dir_path=output_dir, prefix='scVI_clustered', adata=adata)
print(model)


# In[ ]:


resolutions = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
for resolution in resolutions:
    print("Clustering with resolution " + str(resolution))
    str_res = str(resolution).replace('.', '_')
    leiden_key = "leiden" + str_res
    sc.tl.leiden(adata, neighbors_key="N_scVI", key_added=leiden_key, resolution=resolution, flavor="igraph", n_iterations=2)
    jpascvi.plot_umap(adata, resolution, neighbors_key="N_scVI")
    jpascvi.scvi_degs(adata, model, resolution, djd_markers, rep_key="X_scVI", norm_layer="scvi_normalized")
    jpascvi.sc_degs(adata, resolution, use_rep='X_scVI', plots=['dotplot'])

# Save adata with umap and leiden clustering
model.save(dir_path=output_dir, prefix='scVI_clustered_highres', overwrite=True, save_anndata=True)

resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
               1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
               2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
jpascvi.cluster_stats(adata, resolutions)

adipo_genes = ['ADIPOQ', 'PLIN1', 'LPL', 'FABP4', 'PPARG', 'LEP']
sc.pl.umap(adata, neighbors_key='N_scVI', color=adipo_genes, save='Adipocyte_fp.png')


# # Fibroblasts
# Trying to use the CAF score from https://www.nature.com/articles/s41467-024-46504-4

# In[ ]:


# broz = jpascvi.import_markers('/labs/delitto/james/ref/caf_broz.csv')
# buechler = jpascvi.import_markers('/labs/delitto/james/ref/fibroblast_buechler.csv')

# sc._settings.ScanpyConfig.figdir = output_dir
# sc._settings.ScanpyConfig.autoshow = False
# sc._settings.ScanpyConfig.autosave = True

# jpascvi.featureplot(adata, broz, neighbors_key="N_scVI")
# jpascvi.featureplot(adata, buechler, neighbors_key="N_scVI")

# for resolution in resolutions:
#     str_res = str(resolution).replace('.', '_')
#     leiden_key = "leiden" + str_res

#     # dotplot
#     sc.pl.dotplot(adata, broz, groupby=leiden_key, dendrogram=False,
#                   swap_axes=True, use_raw=True, standard_scale="var", save=f'broz_dp_{str_res}.png')
#     sc.pl.dotplot(adata, buechler, groupby=leiden_key, dendrogram=False,
#                   swap_axes=True, use_raw=True, standard_scale="var", save=f'buechler_dp_{str_res}.png')
    
#     # heatmap
#     # make sure you have saved the normalized expression to this layer already
#     sc.pl.heatmap(adata, broz, groupby=leiden_key, 
#                   standard_scale="var", dendrogram=False, save=f'broz_hm_{str_res}.png')
#     sc.pl.heatmap(adata, buechler, groupby=leiden_key, 
#                   standard_scale="var", dendrogram=False, save=f'buechler_hm_{str_res}.png')
    
#     # matrix plot
#     sc.pl.matrixplot(adata, broz, groupby=leiden_key, standard_scale="var", save=f'broz_mp_{str_res}.png')
#     sc.pl.matrixplot(adata, buechler, groupby=leiden_key, standard_scale="var", save=f'buechler_mp_{str_res}.png')


# # Pericytes
# ## Vanlandewijck https://www.nature.com/articles/nature25739
# ## Sziraki https://onlinelibrary.wiley.com/doi/full/10.1111/nan.12942
# Note: ended up not using this one since they didn't match any particular cluster
# ## Van Splunder https://pubmed.ncbi.nlm.nih.gov/37474376
# ## Nee https://www.nature.com/articles/s41588-023-01298-x

# In[ ]:


# vanlandewijck = jpascvi.import_markers('/labs/delitto/james/ref/pericyte_vanlandewijck.csv')
# vansplunder = jpascvi.import_markers('/labs/delitto/james/ref/pericyte_vansplunder.csv')
# nee = jpascvi.import_markers('/labs/delitto/james/ref/pericyte_nee.csv')

# jpascvi.featureplot(adata, vanlandewijck, neighbors_key="N_scVI")
# jpascvi.featureplot(adata, vansplunder, neighbors_key="N_scVI")
# jpascvi.featureplot(adata, nee, neighbors_key="N_scVI")

# for resolution in resolutions:
#     str_res = str(resolution).replace('.', '_')
#     leiden_key = "leiden" + str_res

#     # dotplot
#     sc.pl.dotplot(adata, vanlandewijck, groupby=leiden_key, dendrogram=False,
#                   swap_axes=True, use_raw=True, standard_scale="var", save=f'vanlandewijck_dp_{str_res}.png')
#     sc.pl.dotplot(adata, vansplunder, groupby=leiden_key, dendrogram=False,
#                   swap_axes=True, use_raw=True, standard_scale="var", save=f'vansplunder_dp_{str_res}.png')
#     sc.pl.dotplot(adata, nee, groupby=leiden_key, dendrogram=False,
#                   swap_axes=True, use_raw=True, standard_scale="var", save=f'nee_dp_{str_res}.png')
    
#     # heatmap
#     sc.pl.heatmap(adata, vanlandewijck, groupby=leiden_key, 
#                   standard_scale="var", dendrogram=False, save=f'vanlandewijck_hm_{str_res}.png')
#     sc.pl.heatmap(adata, vansplunder, groupby=leiden_key, 
#                   standard_scale="var", dendrogram=False, save=f'vansplunder_hm_{str_res}.png')
#     sc.pl.heatmap(adata, nee, groupby=leiden_key, 
#                   standard_scale="var", dendrogram=False, save=f'nee_hm_{str_res}.png')
    
#     # matrix plot
#     sc.pl.matrixplot(adata, vanlandewijck, groupby=leiden_key, standard_scale="var", save=f'vanlandewijck_mp_{str_res}.png')
#     sc.pl.matrixplot(adata, vansplunder, groupby=leiden_key, standard_scale="var", save=f'vansplunder_mp_{str_res}.png')
#     sc.pl.matrixplot(adata, nee, groupby=leiden_key, standard_scale="var", save=f'nee_mp_{str_res}.png')


# In[ ]:


# # score genes based on the above fibroblast and pericyte markers
# # first line of code deals with nested lists
# broz_values = [gene for sublist in broz.values() for gene in sublist]
# sc.tl.score_genes(adata, broz_values, score_name='Broz')
# sc.pl.umap(adata, color=['celltype', 'Broz'], save='broz_score_umap.png')

# buechler_values = [gene for sublist in buechler.values() for gene in sublist]
# sc.tl.score_genes(adata, buechler_values, score_name='Buechler')
# sc.pl.umap(adata, color=['celltype', 'Buechler'], save='buechler_score_umap.png')

# vanlandewijck_values = [gene for sublist in vanlandewijck.values() for gene in sublist]
# sc.tl.score_genes(adata, vanlandewijck_values, score_name='Vanlandewijck')
# sc.pl.umap(adata, color=['celltype', 'Vanlandewijck'], save='vanlandewijck_score_umap.png')

# vansplunder_values = [gene for sublist in vansplunder.values() for gene in sublist]
# sc.tl.score_genes(adata, vansplunder_values, score_name='vanSplunder')
# sc.pl.umap(adata, color=['celltype', 'vanSplunder'], save='vansplunder_score_umap.png')

# nee_values = [gene for sublist in nee.values() for gene in sublist]
# sc.tl.score_genes(adata, nee_values, score_name='Nee')
# sc.pl.umap(adata, color=['celltype', 'Nee'], save='nee_score_umap.png')


# # Reload and umap the original celltypes

# In[ ]:


# ad_path = Path(output_dir / 'scVI_clusteredadata.h5ad')
# adata = sc.read_h5ad(ad_path)
# adata


# In[ ]:


# sc.pl.umap(adata, neighbors_key='N_scVI', color='celltype', save='celltype_umap.png')


# # CNV scores UMAP
# Start by importing the anndata with the cnv scores and transferring them over

# In[ ]:


# ad_path = Path(PROJECT_DIR / 'cnv/rawdata_with_cnv.h5ad')
# cnv_adata = sc.read_h5ad(ad_path)
# print(cnv_adata)
# print(cnv_adata.obs)


# In[ ]:


# adata.obs['cnv_score'] = cnv_adata.obs.loc[adata.obs.index, 'cnv_score']
# adata.obs['cnv_score_norm'] = adata.obs['cnv_score'] / np.median(adata.obs['cnv_score'])
# adata.obs['cnv_score_log1p'] = np.log1p(adata.obs['cnv_score_norm'])
# sc.pl.umap(adata, neighbors_key='N_scVI', color=['celltype', 'cnv_score', 'cnv_score_norm', 'cnv_score_log1p'], ncols=2, save='celltype_cnvscore.png')


# In[ ]:


# del cnv_adata


# # Annotation of the mesenchymal subset

# In[ ]:


# ad_path = Path(output_dir / 'scVI_clusteredadata.h5ad')
# adata = sc.read_h5ad(ad_path)
# adata


# In[ ]:


# # Annotate at resolution 0.7
# leiden_key = 'leiden0_7'
# leiden_map = {
#     "0": "Tumor", 
#     "1": "Tumor", 
#     "2": "Tumor", 
#     "3": "Tumor", 
#     "4": "Tumor", 
#     "5": "Tumor", 
#     "6": "Tumor", 
#     "7": "Tumor", 
#     "8": "Tumor", 
#     "9": "Fibroblast", 
#     "10": "Tumor",
#     "11": "Tumor", 
#     "12": "Tumor", 
#     "13": "Tumor", 
#     "14": "Tumor", 
#     "15": "Tumor", 
#     "16": "Tumor",
#     "17": "Pericyte",
#     "18": "Tumor", 
#     "19": "Tumor", 
#     "20": "Tumor", 
# }
# adata.obs['celltype'] = adata.obs[leiden_key].map(leiden_map)
# sc.pl.umap(adata, neighbors_key='N_scVI', color=['celltype'], frameon=False, save='celltype.png')


# In[ ]:


# path = Path(output_dir / 'mesenchymal_annotated.h5ad')
# adata.write_h5ad(path)


# In[ ]:




