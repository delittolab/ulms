#!/usr/bin/env python
# coding: utf-8

# # Using the resolvi model for ULMS G4X dataset
# - Unsupervised using the ulms_g4x_qc outputs

# In[ ]:


import os
import sys
import numpy as np
import scanpy as sc
import torch
import scvi
import pandas as pd
import anndata as ad
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi


# In[ ]:


torch.cuda.is_available()


# In[ ]:


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


# In[ ]:


CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

output_dir = jpascvi.create_output_dir(PARENT_DIR, 'resolvi', change_dir=True)
DATA_DIR = os.path.join(PARENT_DIR, 'qc')
print("DATA_DIR is: " + DATA_DIR)


# In[ ]:


jpa_markers = jpascvi.import_markers((PARENT_DIR / 'ref/jpa_g4x_breast_panel.csv'), output_type='dict')
jpa_markers = {key: value for key, value in jpa_markers.items() if key != 'Plasma_cell'} # JCHAIN and IGHG1 not in this segmentation run
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


# # Import data and reformat

# In[ ]:


# # import all preprocessed section anndatas and sort by section ID
# adata_list = jpascvi.import_from_subdirs(DATA_DIR, label='preprocessed')
# adata_list.sort(key=lambda x: x.uns['section'])
# adata_list


# In[ ]:


# len(adata_list)


# In[ ]:


# for adata in adata_list:
#     print(adata.uns['section'])


# In[ ]:


# for i, adata in enumerate(adata_list):
#     adata.obsm['X_spatial'] = adata.obsm.pop('spatial')
#     adata_list[i] = adata


# In[ ]:


# # concatenate the samples
# adata = ad.concat(adata_list, join="inner")
# del adata_list
# adata


# In[ ]:


# samples = pd.read_csv(os.path.join(PARENT_DIR, 'ref/g4x_ulms_sections.csv'))
# samples


# In[ ]:


# adata.obs.rename(columns={'section' : 'Section'}, inplace=True)
# adata.obs = pd.merge(adata.obs, samples, how='left', on='Section')
# adata.obs


# In[ ]:


# adata.layers["counts"] = adata.X.copy() # this layer will contain the raw counts
# sc.pp.normalize_total(adata) # normalize X to the median total counts
# sc.pp.log1p(adata) # logarithmize X
# adata.raw = adata # full dimension normalized data
# adata


# In[ ]:


# # Check if any cells have less than 10 counts - this may mess up integration
# print(f"Number of cells in anndata: {adata.n_obs}")
# # Make sure to use the raw counts layer
# low_counts = adata[adata.layers['counts'].sum(axis=1) < 10]
# print(f"Number of cells with low counts: {low_counts.n_obs}")


# # Train the model

# In[ ]:


# # set up anndata to compute spatial neighbors within each batch
# scvi.external.RESOLVI.setup_anndata(adata, layer="counts", batch_key='Section')
# resolvi_model = scvi.external.RESOLVI(adata)


# In[ ]:


# # Train the vae
# scvi.settings.seed = 1234
# resolvi_model.train(max_epochs=100)


# In[ ]:


# adata.obsm["X_resolVI"] = resolvi_model.get_latent_representation(adata)
# sc.pp.neighbors(adata, use_rep="X_resolVI")
# sc.tl.umap(adata, min_dist=0.3)


# In[ ]:


# # saving the model and anndata now that umap has been computed
# resolvi_model.save(dir_path=output_dir, prefix='resolvi', overwrite=True, save_anndata=True)


# # Clustering and Differential Expression

# In[ ]:


# sc.pl.umap(adata, color=['Section', 'Patient'], frameon=False, save='section_and_patient.png')
# sc.pl.umap(adata, color=['component', 'volume', 'surface_area', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'n_counts', 'n_genes',], frameon=False, save='qc.png')


# In[ ]:


# sc.pp.scale(adata, max_value=10) # scale to unit variance and zero mean and clip any gene expression above 10 std deviations


# In[ ]:


# jpascvi.featureplot(adata, jpa_markers, save='resolvi.png', use_raw=False) # False to allow scaled values to be plotted from adata.X


# In[ ]:


# # Leiden clustering at multiple resolutions and differential gene expression
# for resolution in resolutions:
#     str_res = str(resolution).replace('.', '_')
#     leiden_key = 'leiden' + str_res
#     sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key)
#     jpascvi.plot_umap(adata, resolution)
#     jpascvi.sc_degs(adata, resolution, use_rep='X_resolVI')

# # Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
# jpascvi.cluster_stats(adata, resolutions, scores = ['Calinski-Harabasz', 'Davies-Bouldin'], rep='X_resolVI')


# In[ ]:


# sc.pl.umap(adata, color=['MYH11', 'MYLK', 'DES', 'TAGLN', 'ESR1', 'COL1A1', 'SDC1', 'SOX10'], size=0.2, save='ulms_feature_plot.png')
# sc.pl.umap(adata, color=['VEGFA', 'SLC2A1',], size=0.2, save='ischemia_feature_plot.png')
# sc.pl.umap(adata, color=['VWF', 'PECAM1', 'COL1A1', 'LUM', 'CD68', 'CD163', 'RGS5', 'PDGFRB', 'CD2', 'IL7R'], size=0.2, save='nontumor_feature_plot.png')
# sc.pl.umap(adata, color=['ESR1', 'PGR', 'AR'], save='hormononal_feature_plot.png', size=0.2)
# sc.pl.umap(adata, color=['LAG3', 'PDCD1'], save='immune_checkpoint_feature_plot.png', size=0.2)


# In[ ]:


# adata.write_h5ad('resolvi_clustered.h5ad')


# # Checkpoint: Reload the data

# In[ ]:


adata = sc.read_h5ad('resolvi_clustered.h5ad')
adata


# In[ ]:


resolvi_model = scvi.external.RESOLVI.load(output_dir, prefix='resolvi', adata=adata)
resolvi_model


# # Checking the model training

# In[ ]:


resolvi_model.history['elbo_train']


# In[ ]:


plt.plot(resolvi_model.history["elbo_train"])
plt.xlabel("Epoch")
plt.ylabel("Training ELBO")
plt.title("resolVI Model ELBO During Training")
plt.savefig('resolvi_elbo_plot.png')


# # Getting the corrected counts
# https://discourse.scverse.org/t/best-practices-for-downstream-analysis-of-resolvi-corrected-data/3906/2

# In[ ]:


samples_corr = resolvi_model.sample_posterior(
    model=resolvi_model.module.model_corrected,
    return_sites=["px_rate"],
    summary_fun={"post_sample_q50": np.median},
    num_samples=3,
    summary_frequency=30,
)
samples_corr = pd.DataFrame(samples_corr).T


# In[ ]:


samples = resolvi_model.sample_posterior(
    model=resolvi_model.module.model_residuals,
    return_sites=["mixture_proportions"],
    summary_fun={"post_sample_means": np.mean},
    num_samples=3,
    summary_frequency=100,
)
samples = pd.DataFrame(samples).T


# In[ ]:


adata.obs[["true_proportion", "diffusion_proportion", "background_proportion"]] = samples.loc[
    "post_sample_means", "mixture_proportions"
]


# In[ ]:


sc.pl.umap(
    adata,
    color=["total_counts", "true_proportion", "diffusion_proportion", "background_proportion"],
    save="resolvi_corr.png"
)


# In[ ]:


print("Adding the generated expression matrix to layers")
adata.layers["generated_expression"] = samples_corr.loc["post_sample_q50", "px_rate"]


# In[ ]:


jpascvi.featureplot(adata, jpa_markers, save='corr_resolvi.png', layer="generated_expression", vmax='p98')


# In[ ]:


# print("Transforming the generated expression matrix")
# # Store the current scaled lognormalized counts in X away in a layer
# adata.layers["scaled"] = adata.X
# # Normalize
# sc.pp.normalize_total(adata, layer="generated_expression", target_sum=1e4)
# # Square root transformation
# if scipy.sparse.issparse(adata.layers['generated_expression']):
#     adata.layers['generated_expression'] = adata.layers['generated_expression'].toarray()
# adata.layers['generated_expression'] = np.sqrt(adata.layers['generated_expression'])


# In[ ]:


print("Adding the normalized expression matrix to layers")
adata.layers["X_normalized_resolVI"] = resolvi_model.get_normalized_expression()


# In[ ]:


jpascvi.featureplot(adata, jpa_markers, save='norm_resolvi.png', layer="X_normalized_resolVI")


# In[ ]:


adata.write_h5ad('resolvi_clustered_corr.h5ad')

