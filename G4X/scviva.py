#!/usr/bin/env python
# coding: utf-8

# # Using the scVIVA model for ULMS G4X dataset
# - Using coarse cell type annotations from resolVI

# In[1]:


import os
import sys
import numpy as np
import scanpy as sc
import torch
import scvi
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi


# In[2]:


torch.cuda.is_available()


# In[3]:


# version control
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("scvi:", scvi.__version__)

plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.interactive = False
plt.ioff()
sc.settings.autoshow = False

sc.settings.n_jobs = -1  # Use all available cores
scvi.settings.seed = 1234
torch.set_float32_matmul_precision("high")


# In[4]:


CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

output_dir = jpascvi.create_output_dir(PARENT_DIR, 'scviva', change_dir=True)
DATA_DIR = PARENT_DIR / 'annotation'
print(f"DATA_DIR is: {DATA_DIR}")


# In[5]:


jpa_markers = jpascvi.import_markers((PARENT_DIR / 'ref/jpa_g4x_breast_panel.csv'), output_type='dict')
jpa_markers = {key: value for key, value in jpa_markers.items() if key != 'Plasma_cell'} # JCHAIN and IGHG1 not in this segmentation run
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


# # Import data and reformat for scVIVA

# In[10]:


# This is the coarsely annotated anndata from resolVI
adata = sc.read_h5ad(DATA_DIR / 'coarse_celltype.h5ad')
adata


# In[11]:


cols_to_drop = ['_indices', '_scvi_batch', '_scvi_ind_x', '_scvi_labels', 
                'leiden0_1', 'leiden0_2', 'leiden0_3', 'leiden0_4', 'leiden0_5', 
                'leiden0_6', 'leiden0_7', 'leiden0_8', 'leiden0_9', 'leiden1_0', 
                'leiden1_1', 'leiden1_2', 'leiden1_3', 'leiden1_4', 'leiden1_5']
adata.obs.drop(cols_to_drop, axis='columns', inplace=True, errors='ignore')

uns_to_drop = ['_scvi_manager_uuid', '_scvi_uuid', 
               'dendrogram_leiden0_1', 'dendrogram_leiden0_2', 'dendrogram_leiden0_3', 'dendrogram_leiden0_4', 
               'dendrogram_leiden0_5', 'dendrogram_leiden0_6', 'dendrogram_leiden0_7', 'dendrogram_leiden0_8', 
               'dendrogram_leiden0_9', 'dendrogram_leiden1_0', 'dendrogram_leiden1_1', 'dendrogram_leiden1_2', 
               'dendrogram_leiden1_3', 'dendrogram_leiden1_4', 'dendrogram_leiden1_5', 
               'leiden0_1', 'leiden0_1_colors', 'leiden0_2', 'leiden0_2_colors', 'leiden0_3', 'leiden0_3_colors', 
               'leiden0_4', 'leiden0_4_colors', 'leiden0_5', 'leiden0_5_colors', 'leiden0_6', 'leiden0_6_colors', 
               'leiden0_7', 'leiden0_7_colors', 'leiden0_8', 'leiden0_8_colors', 'leiden0_9', 'leiden0_9_colors', 
               'leiden1_0', 'leiden1_0_colors', 'leiden1_1', 'leiden1_1_colors', 'leiden1_2', 'leiden1_2_colors', 
               'leiden1_3', 'leiden1_3_colors', 'leiden1_4', 'leiden1_4_colors', 'leiden1_5', 'leiden1_5_colors', 
               'log1p', 'neighbors', 'rank_genes_groups', 'umap']
for key in uns_to_drop:
    if key in adata.uns:
        adata.uns.pop(key, None)
    
adata


# In[12]:


setup_kwargs = {
    "sample_key": "Section",  # column in adata.obs that contains the individual slide ID
    "labels_key": "coarse_celltype",  # column in adata.obs that contains the cell type labels
    "cell_coordinates_key": "X_spatial",  # spatial coordinates key in adata.obsm
    "expression_embedding_key": "X_resolVI",  # expression embedding key in adata.obsm
}


# In[13]:


scvi.external.SCVIVA.preprocessing_anndata(
    adata,
    k_nn=20,  # number of nearest neighbors for spatial graph construction
    **setup_kwargs,
)


# In[14]:


scvi.external.SCVIVA.setup_anndata(
    adata,
    layer="counts",  # adata layer that contains the raw counts
    batch_key="Patient",  # column in adata.obs that contains the batch covariate
    **setup_kwargs,
)


# # Train the model

# In[ ]:


nichevae = scvi.external.SCVIVA(adata)
nichevae


# In[ ]:


# Train the vae
scvi.settings.seed = 1234
nichevae.train(
    max_epochs=600,
    early_stopping=True,
    check_val_every_n_epoch=1,
)


# In[ ]:


# Check training
try:
    print(nichevae.history.keys())
    
    plt.plot(nichevae.history["elbo_train"], label="train")
    plt.plot(nichevae.history["elbo_validation"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.title("scVIVA Model ELBO During Training")
    plt.savefig('scviva_elbo_plot.png')
    plt.close()
    
    plt.plot(nichevae.history["niche_compo_validation"])
    plt.xlabel("Epoch")
    plt.ylabel("Niche Component Validation")
    plt.title("scVIVA Model Niche Component Validation")
    plt.savefig('scviva_niche_component_validation.png')
    plt.close()
    
    plt.plot(nichevae.history["niche_reconst_validation"])
    plt.xlabel("Epoch")
    plt.ylabel("Niche Reconstruction Validation")
    plt.title("scVIVA Niche Reconstruction Validation")
    plt.savefig('scviva_niche_recon_validation.png')
    plt.close()
    
    plt.plot(nichevae.history["kl_local_validation"])
    plt.xlabel("Epoch")
    plt.ylabel("KL Local Validation")
    plt.title("scVIVA KL Local Validation")
    plt.savefig('scviva_kl_local_validation.png')
    plt.close()
    
    plt.plot(nichevae.history["reconstruction_loss_validation"])
    plt.xlabel("Epoch")
    plt.ylabel("Recon Loss Validation")
    plt.title("scVIVA Recon Loss Validation")
    plt.savefig('scviva_recon_loss_validation.png')
    plt.close()

except Exception as e:
    print(e)


# In[ ]:


adata.obsm["X_scVIVA"] = nichevae.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scVIVA", n_neighbors=30)
sc.tl.umap(adata, min_dist=0.3)


# In[ ]:


# saving the model and anndata now that umap has been computed
nichevae.save(dir_path=output_dir, prefix='scviva', overwrite=True, save_anndata=True)


# # Clustering and Differential Expression

# In[ ]:


sc.pl.umap(adata, color='coarse_celltype', frameon=False, save='coarse_celltype.png')
sc.pl.umap(adata, color=['Section', 'Patient'], frameon=False, save='section_and_patient.png')
sc.pl.umap(adata, color=['component', 'volume', 'surface_area', 
                         'n_genes_by_counts', 'log1p_n_genes_by_counts', 
                         'total_counts', 'log1p_total_counts', 'n_counts', 'n_genes',], 
           frameon=False, save='qc.png')


# In[ ]:


jpascvi.featureplot(adata, jpa_markers, save='scviva.png', use_raw=False, vmax='p98') # False to allow scaled values to be plotted from adata.X


# In[ ]:


# Leiden clustering at multiple resolutions and differential gene expression
for resolution in resolutions:
    str_res = str(resolution).replace('.', '_')
    leiden_key = 'leiden' + str_res
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key)
    jpascvi.plot_umap(adata, resolution)
    jpascvi.sc_degs(adata, resolution, use_rep='X_scVIVA')

# Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
jpascvi.cluster_stats(adata, resolutions, scores = ['Calinski-Harabasz', 'Davies-Bouldin'], rep='X_scVIVA')


# In[ ]:


sc.pl.umap(adata, 
           color=['MYH11', 'MYLK', 'DES', 'TAGLN', 'ESR1', 'COL1A1', 'SDC1', 'SOX10'], 
           size=0.2, 
           save='ulms_feature_plot.png', 
           vmax='p98')

sc.pl.umap(adata, 
           color=['VEGFA', 'SLC2A1',], 
           size=0.2, 
           save='ischemia_feature_plot.png', 
           vmax='p98')

sc.pl.umap(adata, 
           color=['VWF', 'PECAM1', 'COL1A1', 'LUM', 'CD68', 'CD163', 'RGS5', 'PDGFRB', 'CD2', 'IL7R'], 
           size=0.2, 
           save='nontumor_feature_plot.png', 
           vmax='p98')

sc.pl.umap(adata, 
           color=['ESR1', 'PGR', 'AR'], 
           save='hormononal_feature_plot.png', 
           size=0.2, 
           vmax='p98')

sc.pl.umap(adata, 
           color=['LAG3', 'PDCD1'], 
           save='immune_checkpoint_feature_plot.png', 
           size=0.2, 
           vmax='p98')


# In[ ]:


adata.write_h5ad('scviva_clustered.h5ad')


# # Plotting the corrected and normalized counts from resolVI in the scVIVA UMAP

# In[ ]:


sc.pl.umap(
    adata,
    color=["total_counts", "true_proportion", "diffusion_proportion", "background_proportion"],
    save="resolvi_corr.png"
)


# In[ ]:


jpascvi.featureplot(adata, jpa_markers, save='corr_resolvi.png', layer="generated_expression", vmax='p98')


# In[ ]:


jpascvi.featureplot(adata, jpa_markers, save='norm_resolvi.png', layer="X_normalized_resolVI", vmax='p98')

