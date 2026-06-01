# # scANVI to integrate ULMS G4X dataset into ULMS scRNAseq reference
# - now running on just the tumor subset
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/hub/query_hlca_knn.html

# Built on top of ULMS_scANVI_tumor_allgenes
# Using knn classifier to predict cell types instead of scANVI
# kernel-weighted k-nearest neighbors with soft voting

# SET UP DEPENDENCIES

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
import pynndescent
import numba

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi

print(f"\nRunning script: {Path(__file__).name}\n")

print("Is CUDA available?", torch.cuda.is_available())

# version control
print("\nPackage versions:")
print("torch:", torch.__version__)
print("anndata:", ad.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("scvi:", scvi.__version__)
print("pynndescent:", pynndescent.__version__)

mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.rcParams['axes.facecolor'] = 'white'
plt.ioff()
sc.settings.autoshow = False
sc.settings.n_jobs = -1  # Use all available cores
SEED = 1234
scvi.settings.seed = SEED
torch.set_float32_matmul_precision("high")

# SET UP DIRECTORIES

CURRENT_DIR = Path.cwd()
MULTIMODAL_DIR = CURRENT_DIR.parent
print(MULTIMODAL_DIR)

INPUT_MASTER_DIR = MULTIMODAL_DIR / 'scANVI_tumor_allgenes'

# Making an output directory using the pathlib package
OUTPUT_MASTER_DIR = jpascvi.create_output_dir(INPUT_MASTER_DIR, 'knn', change_figdir=True)

# PARAMETERS

SCANVI_LATENT_KEY = "X_scANVI"
KNN_PREDICTIONS_KEY = "knn_pred"
CT_KEY = 'tumor_subtype'

# Load scANVI reference anndata that has the scANVI model embedding saved in adata.obsm
SCANVI_REF_DIR = INPUT_MASTER_DIR / 'scanvi_ref'
adata_ref = sc.read_h5ad(SCANVI_REF_DIR / 'scanvi_ref_adata.h5ad')
print(adata_ref)

# Learn a neighbors index using PyNNDescent, an approximate neighbors technique, on the scANVI reference embedding
# We will later use this as a classifier
print("Calculating the neighbors index on the scANVI reference embedding...")
X_train = adata_ref.obsm[SCANVI_LATENT_KEY]
ref_nn_index = pynndescent.NNDescent(X_train)
ref_nn_index.prepare()

# Load the scANVI query anndata that has the scANVI model embedding saved in adata.obsm
SCANVI_QUERY_DIR = INPUT_MASTER_DIR / 'scanvi_query'
adata_query = sc.read_h5ad(SCANVI_QUERY_DIR / 'scanvi_query_adata.h5ad')
print(adata_query)

# get the embedding
query_emb = ad.AnnData(adata_query.obsm[SCANVI_LATENT_KEY])
query_emb.obs_names = adata_query.obs_names

# find the nearest neighbors
ref_neighbors, ref_distances = ref_nn_index.query(query_emb.X)

# convert distances to affinities
stds = np.std(ref_distances, axis=1) # std of distances for each query
stds = (2.0 / stds) ** 2 # sigma^2 = (2/std)^2 creates a per-query scaling factor 
stds = stds.reshape(-1, 1)
ref_distances_tilda = np.exp(-np.true_divide(ref_distances, stds)) # apply Gaussian RBF kernel w = exp(-d^2/sigma^2)
weights = ref_distances_tilda / np.sum(ref_distances_tilda, axis=1, keepdims=True) # normalize - divide by row sum so that each query's weights sum to 1

# check
if np.any(np.isnan(weights)):
    print("Warning: NaN values in weights detected")
    weights = np.nan_to_num(weights, nan=0.0)

# Weighted voting
@numba.njit # just-in-time compilation - much faster for numerical operations
def weighted_prediction(weights, ref_cats):
    """
    Get highest weight category. Predict category based on neighbor votes.
    """
    N = len(weights)
    predictions = np.zeros((N,), dtype=ref_cats.dtype)
    uncertainty = np.zeros((N,))
    for i in range(N):
        obs_weights = weights[i]
        obs_cats = ref_cats[i]
        best_prob = 0
        for c in np.unique(obs_cats):
            cand_prob = np.sum(obs_weights[obs_cats == c])
            if cand_prob > best_prob: # pick winner
                best_prob = cand_prob
                predictions[i] = c
                uncertainty[i] = max(1 - best_prob, 0)

    return predictions, uncertainty


ref_cats = adata_ref.obs[CT_KEY].cat.codes.to_numpy()[ref_neighbors]
p, u = weighted_prediction(weights, ref_cats)
p = np.asarray(adata_ref.obs[CT_KEY].cat.categories)[p]
query_emb.obs[CT_KEY + "_pred"], query_emb.obs[CT_KEY + "_uncertainty"] = p, u

# Filter predictions based on uncertainty threshold
UNCERTAINTY_THRESHOLD = 0.8 # this is intentionally quite high
mask = query_emb.obs[CT_KEY + "_uncertainty"] > UNCERTAINTY_THRESHOLD
print(f"{CT_KEY}: {sum(mask) / len(mask)} unknown")
query_emb.obs.loc[mask, CT_KEY + "_pred"] = "Unknown"

# check
print(f"\nPrediction Summary:")
print(f"Total queries: {len(query_emb)}")
print(f"Unknown predictions: {sum(mask)} ({sum(mask)/len(mask)*100:.1f}%)")
print(f"Mean uncertainty: {u.mean():.3f}")


# Combine embeddings
ref_emb = ad.AnnData(X_train, obs=adata_ref.obs)
print(ref_emb)
print(query_emb)
combined_emb = ad.concat([ref_emb, query_emb], join='outer')

# Visualize embeddings and predictions
combined_emb.obsm[SCANVI_LATENT_KEY] = combined_emb.X
print('Calculating nearest neighbors on the combined embedding...')
sc.pp.neighbors(combined_emb, use_rep=SCANVI_LATENT_KEY)
print('Calculating UMAP...')
sc.tl.umap(combined_emb, min_dist=0.3)

color = CT_KEY + "_uncertainty"
sc.pl.umap(combined_emb, color=color, frameon=False, save=f'combined_emb_{color}.png')
sc.pl.umap(combined_emb, color=color, frameon=False, save=f'combined_emb_{color}.pdf')
color = CT_KEY + "_pred"
sc.pl.umap(combined_emb, color=color, frameon=False, save=f'combined_emb_{color}.png')
sc.pl.umap(combined_emb, color=color, frameon=False, save=f'combined_emb_{color}.pdf')

# Save results
print('\nSaving annotated combined embedding...')
combined_emb.write_h5ad(OUTPUT_MASTER_DIR / 'combined_emb_with_knn_pred.h5ad')