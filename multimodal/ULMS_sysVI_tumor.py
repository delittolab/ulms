# # SysVI to integrate ULMS G4X dataset into ULMS scRNAseq reference. Just tumor cells.
# https://discourse.scverse.org/t/conduct-scanvi-after-running-sysvi-on-multi-platform-data/3745
# https://pubmed.ncbi.nlm.nih.gov/41168710/
# https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/scrna/sysVI.html
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/hub/query_hlca_knn.html#download-query-data

# Built on top of ULMS_scANVI_tumor_allgenes
# Using knn classifier to predict cell types instead of scANVI
# kernel-weighted k-nearest neighbors with soft voting

# SET UP DEPENDENCIES

import sys
import numpy as np
import scanpy as sc
import torch
import scvi
from scvi.external import SysVI
import pandas as pd
import anndata as ad
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pynndescent
import numba
import scipy.sparse as sp

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

# PARAMETERS
SEED = 1
CCLW = 25 # cycle-consistency loss weight. Increase this for more batch correction
# The conditions key specify the covariates over which to integrate your samples
# batch is the sequencing batch (scRNAseq) 
# section (G4X, like in resolVI)
# sample is the consistent patient sample numbering (G4X and scRNAseq)
# assay is scRNAseq or spatial
TECH_KEY = 'assay'
CONDITION_KEYS = ['batch', 'sample', 'section']
CT_KEY = 'tumor_subtype'
N_TOP_GENES = 10000 # number of highly variables genes in the scRNAseq data
MIN_COUNTS = 10 # remove scRNAseq cells with fewer than this many counts after subsetting to the G4X gene list, which will be the genes used for integration
MAX_EPOCHS = 50
BATCH_SIZE = 1024 # https://github.com/scverse/scvi-tools/issues/2726
REF_KEY = 'is_reference'  # boolean: True for reference, False for query

mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.rcParams['axes.facecolor'] = 'white'
plt.ioff()
sc.settings.autoshow = False
sc.settings.n_jobs = -1  # Use all available cores
scvi.settings.seed = SEED
torch.set_float32_matmul_precision("high")
scvi.settings.batch_size = BATCH_SIZE

# SET UP DIRECTORIES

CURRENT_DIR = Path.cwd()
MULTIMODAL_DIR = CURRENT_DIR.parent
print(MULTIMODAL_DIR)
G4X_DIR = MULTIMODAL_DIR.parent / 'G4X'
print(G4X_DIR)
SCRNASEQ_DIR = MULTIMODAL_DIR.parent / 'scRNAseq'
print(SCRNASEQ_DIR)
# Making an output directory using the pathlib package
OUTPUT_MASTER_DIR = jpascvi.create_output_dir(MULTIMODAL_DIR, 'SysVI', change_figdir=True)

# LOAD THE SCRNASEQ REFERENCE DATA

# # Import scRNAseq data, which will be the reference data
print("\nLoading scRNAseq anndata")
data_dir = SCRNASEQ_DIR / 'objects'
adata_ref = sc.read_h5ad(data_dir / 'annotated_raw_counts.h5ad')
print(adata_ref)

# Reformat the scRNAseq anndata for training, subsetting to only those genes present in the G4X anndata
print("\nReformatting scRNAseq anndata")
adata_ref.obs[TECH_KEY] = 'scRNAseq'
adata_ref.obs.rename(columns={'celltype' : 'cell_type'}, inplace=True)
adata_ref = adata_ref[adata_ref.obs['cell_type'] != 'RBC'].copy()
adata_ref.obs['cell_type'] = adata_ref.obs['cell_type'].astype('category')
adata_ref.obs['cell_type'] = adata_ref.obs['cell_type'].cat.remove_unused_categories()
adata_ref.obs['sample'] = adata_ref.obs['sample'].str.replace('Sample', '')
adata_ref.obs['sample'] = adata_ref.obs['sample'].astype('category')
adata_ref.obs['section'] = 'scRNAseq'
adata_ref.obs['section'] = adata_ref.obs['section'].astype('category')
adata_ref.obs[TECH_KEY] = adata_ref.obs[TECH_KEY].astype('category')
adata_ref.obs['batch'] = adata_ref.obs['batch'].astype('category')
print(adata_ref)

# Subset the reference to just the tumor cells for this analysis
print("\nSubsetting scRNAseq reference to just tumor cells")
adata_ref = adata_ref[adata_ref.obs['cell_type'] == 'Tumor'].copy()
print(adata_ref)

# Load the tumor subtype annotations for the scRNAseq reference data
print("\nLoading scRNAseq reference annotations")
data_dir = SCRNASEQ_DIR / 'objects'
print(data_dir)
ann = sc.read_h5ad(data_dir / 'tumor_annotated.h5ad')
ann.obs[CT_KEY] = ann.obs['celltype'] # renaming
adata_ref.obs[CT_KEY] = ann.obs.loc[adata_ref.obs.index, CT_KEY]
adata_ref.obs[CT_KEY] = adata_ref.obs[CT_KEY].astype('category')
print(np.unique(adata_ref.obs[CT_KEY]))
del ann

# Calculate highly variable genes - this takes raw counts
# Keep increasing the span by 0.1 until it runs
sc.pp.filter_genes(adata_ref, min_cells=3)
span = 0.3
while True:
    try:
        sc.pp.highly_variable_genes(adata_ref,
                                    n_top_genes=N_TOP_GENES,
                                    flavor='seurat_v3',
                                    batch_key='batch',
                                    subset=True,
                                    span=span)
        print(f"HVG completed with span={span}")
        break
    except Exception as e:
        print(f"HVG failed at span={span}: {e}")
        span += 0.1
        if span > 0.9:
            raise
        print(f"Retrying HVG with span={span}")

# Log normalize data and save raw counts in a layer, as recommended for scvi-tools
adata_ref.layers["counts"] = adata_ref.X.copy() # this layer will contain the raw counts
sc.pp.normalize_total(adata_ref, target_sum=1e4) # SysVI recommends normalizing to a fixed number
sc.pp.log1p(adata_ref) # logarithmize X
adata_ref.raw = adata_ref # full dimension normalized logtransformed raw data

# LOAD THE G4X QUERY DATA

# # Import G4X data, which will eventually be the query data
print("\nLoading G4X anndata")
data_dir = G4X_DIR / 'objects'
print(data_dir)
adata_g4x = sc.read_h5ad(data_dir / 'g4x_raw_counts.h5ad')
print(adata_g4x)

# reformat the G4X anndata
print("\nReformatting G4X anndata")
adata_g4x.obs_names = adata_g4x.obs['cell_name']
adata_g4x.obs[TECH_KEY] = 'spatial'
adata_g4x.obs.rename(columns={'Sample' : 'sample', 'Section' : 'section'}, inplace=True)
adata_g4x.obs['batch'] = 'g4x'
adata_g4x.obs['batch'] = adata_g4x.obs['batch'].astype('category')
adata_g4x.obs['sample'] = adata_g4x.obs['sample'].astype('category')
adata_g4x.obs['section'] = adata_g4x.obs['section'].astype('category')
adata_g4x.obs[TECH_KEY] = adata_g4x.obs[TECH_KEY].astype('category')
print(adata_g4x)

# add in the celltype annotations
print("\nLoading G4X annotations")
data_dir = G4X_DIR / 'annotation'
print(data_dir)
g4x_ann = sc.read_h5ad(data_dir / 'scviva_celltype.h5ad')
g4x_ann.obs_names = g4x_ann.obs['cell_name']
print(g4x_ann)
adata_g4x.obs['cell_type'] = g4x_ann.obs.loc[adata_g4x.obs.index, 'celltype']
print(np.unique(adata_g4x.obs['cell_type']))
print(adata_g4x)
del g4x_ann

# subset for tumor cells
print("\nSubsetting for tumor cells")
adata_g4x = adata_g4x[adata_g4x.obs['cell_type'] == 'Tumor'].copy()
print(adata_g4x)

# Set the tumor subtypes to unknown
adata_g4x.obs[CT_KEY] = 'Unknown'
adata_g4x.obs[CT_KEY] = adata_g4x.obs[CT_KEY].astype('category')

# Log normalize data and save raw counts in a layer, as recommended for scvi-tools
adata_g4x.layers["counts"] = adata_g4x.X.copy() # this layer will contain the raw counts
sc.pp.normalize_total(adata_g4x, target_sum=1e4) # SysVI recommends normalizing to a fixed number
sc.pp.log1p(adata_g4x) # logarithmize X
adata_g4x.raw = adata_g4x # full dimension normalized logtransformed raw data

# subset to the intersection of both gene lists, which should be N_TOP_GENES
common_genes = adata_ref.var_names.intersection(adata_g4x.var_names).tolist()
adata_ref = adata_ref[:, common_genes].copy()
adata_g4x = adata_g4x[:, common_genes].copy()

print("Here are the genes that will be used in the model:")
print(*common_genes)

# Removing low-quality cells

print("scRNAseq adata: any cells with low counts after subsetting and filtering?")
# Some cells may have low HVG counts - this may mess up integration and differential expression calculation by creating a division by zero
print(f"Number of cells in anndata: {adata_ref.n_obs}")
# Make sure to use the raw counts layer - in this case just adata.X since there are no layers
low_counts = adata_ref[adata_ref.X.sum(axis=1) < MIN_COUNTS]
print(f"Number of cells with low HVG counts: {low_counts.n_obs}")
adata_ref = adata_ref[adata_ref.X.sum(axis=1) >= MIN_COUNTS].copy()
print('Anndata after filtering out cells with low counts:')
print(adata_ref)

print("G4X adata: any cells with low counts after subsetting and filtering?")
# Some cells may have low HVG counts - this may mess up integration and differential expression calculation by creating a division by zero
print(f"Number of cells in anndata: {adata_g4x.n_obs}")
# Make sure to use the raw counts
low_counts = adata_g4x[adata_g4x.X.sum(axis=1) < MIN_COUNTS]
print(f"Number of cells with low HVG counts: {low_counts.n_obs}")
adata_g4x = adata_g4x[adata_g4x.X.sum(axis=1) >= MIN_COUNTS].copy()
print('Anndata after filtering out cells with low counts:')
print(adata_g4x)


# Prepare for integration
print("\nFinal scRNAseq anndata:")
print(adata_ref)
print(np.unique(adata_ref.obs['cell_type']))
print("\nFinal G4X anndata:")
print(adata_g4x)
print(np.unique(adata_g4x.obs['cell_type']))

# concatenation
adata = ad.concat([adata_ref, adata_g4x], join='inner')
print(adata)
del adata_ref
del adata_g4x

# Densify the counts layer to speed up training
if sp.issparse(adata.layers["counts"]):
    adata.layers["counts"] = adata.layers["counts"].toarray()
    print("Densified counts layer")

# Set up the anndata and model
SysVI.setup_anndata(adata=adata, 
                    layer="counts", 
                    batch_key=TECH_KEY, 
                    categorical_covariate_keys=CONDITION_KEYS)
model = SysVI(
    adata=adata,
    embed_categorical_covariates=True,
)

# Train the vae with early stopping for the default number of epochs
scvi.settings.seed = SEED
model.train(max_epochs=MAX_EPOCHS, 
            batch_size=BATCH_SIZE,
            check_val_every_n_epoch=1,
            plan_kwargs={
                "z_distance_cycle_weight": CCLW
                }
            )

# Check if batch size is actually used
try:
    print(f"Training batch size used: {model.trainer.train_dataloader.batch_size}")
except Exception as e:
    print(e)

# Plot losses
# The plotting code below was specifically adapted to the above-specified model and its training
# If changing the model or training the plotting functions may need to be adapted accordingly
# Make detailed plot after N epochs
epochs_detail_plot = 50
# Losses to plot
losses = [
    "reconstruction_loss_train",
    "kl_local_train",
    "cycle_loss_train",
]
try:
    fig, axs = plt.subplots(2, len(losses), figsize=(len(losses) * 3, 4))
    for ax_i, l_train in enumerate(losses):
        l_val = l_train.replace("_train", "_validation")
        l_name = l_train.replace("_train", "")
        # Change idx of epochs to start with 1
        l_val_values = model.trainer.logger.history[l_val].copy()
        l_val_values.index = l_val_values.index + 1
        l_train_values = model.trainer.logger.history[l_train].copy()
        l_train_values.index = l_train_values.index + 1
        for l_values, c, alpha, dp in [
            (l_train_values, "tab:blue", 1, epochs_detail_plot),
            (l_val_values, "tab:orange", 0.5, epochs_detail_plot),
        ]:
            axs[0, ax_i].plot(l_values.index, l_values.values.ravel(), c=c, alpha=alpha)
            axs[0, ax_i].set_title(l_name)
            axs[1, ax_i].plot(l_values.index[dp:], l_values.values.ravel()[dp:], c=c, alpha=alpha)
    fig.tight_layout()
    fig_path = OUTPUT_MASTER_DIR / 'sysvi_elbo_plot.png'
    fig.savefig(fig_path)
    plt.close()
except Exception as e:
    print(e)

# Get embedding - save it into X of new AnnData
embed = model.get_latent_representation(adata=adata)
embed = sc.AnnData(embed, obs=adata.obs)
embed.obs[REF_KEY] = embed.obs[TECH_KEY] == 'scRNAseq'

print("Calculating neighbors and UMAP...")
sc.pp.neighbors(embed, use_rep='X')
sc.tl.umap(embed, min_dist=0.3, random_state=SEED)

# Obs columns to color by
colors = [TECH_KEY, CT_KEY] + CONDITION_KEYS
# One plot per obs column used for coloring
for c in colors:
    sc.pl.umap(
        embed,
        color=c,
        frameon=False,
        save=f'{c}_sysvi.png'
    )
    sc.pl.umap(
        embed,
        color=c,
        frameon=False,
        save=f'{c}_sysvi.pdf'
    )

# save the embedding
embed.write_h5ad(OUTPUT_MASTER_DIR / 'SysVI_embedding.h5ad')
print("SysVI embedding saved to ", OUTPUT_MASTER_DIR)


# KNN CLASSIFIER TO PREDICT CELL TYPES FROM SYSVI EMBEDDING

print(f"Reference cells: {sum(embed.obs[REF_KEY])}")
print(f"Query cells: {sum(~embed.obs[REF_KEY])}")

# Separate reference and query based on mask
ref_mask = embed.obs[REF_KEY]
query_mask = ~ref_mask

X_ref = embed.X[ref_mask]  # Extract reference embedding
X_query = embed.X[query_mask]  # Extract query embedding

# Learn a neighbors index using PyNNDescent, an approximate neighbors technique, on the scANVI reference embedding
# We will later use this as a classifier
# Build k-NN index on reference subset
print("Building k-NN index on reference cells...")
ref_nn_index = pynndescent.NNDescent(X_ref)
ref_nn_index.prepare()

# Query the reference neighbors from query cells
print("Finding nearest reference neighbors for query cells...")
ref_neighbors, ref_distances = ref_nn_index.query(X_query)

# Convert distances to affinities
stds = np.std(ref_distances, axis=1)
stds = (2.0 / stds) ** 2
stds = stds.reshape(-1, 1)
ref_distances_tilda = np.exp(-np.true_divide(ref_distances, stds))
weights = ref_distances_tilda / np.sum(ref_distances_tilda, axis=1, keepdims=True)

# Weighted voting function
@numba.njit
def weighted_prediction(weights, ref_cats):
    """Get highest weight category. Predict category based on neighbor votes."""
    N = len(weights)
    predictions = np.zeros((N,), dtype=ref_cats.dtype)
    uncertainty = np.zeros((N,))
    for i in range(N):
        obs_weights = weights[i]
        obs_cats = ref_cats[i]
        best_prob = 0
        for c in np.unique(obs_cats):
            cand_prob = np.sum(obs_weights[obs_cats == c])
            if cand_prob > best_prob:
                best_prob = cand_prob
                predictions[i] = c
                uncertainty[i] = max(1 - best_prob, 0)
    return predictions, uncertainty

# Get reference categories for the neighbors
ref_labels = embed.obs.loc[ref_mask, CT_KEY]
ref_cats = ref_labels.cat.codes.to_numpy()[ref_neighbors]

# Make predictions
p, u = weighted_prediction(weights, ref_cats)
p = np.asarray(ref_labels.cat.categories)[p]

# Add predictions back to the joint object
embed.obs[CT_KEY + "_pred"] = embed.obs[CT_KEY].astype(str).copy()  # Initialize with actual labels
embed.obs[CT_KEY + "_uncertainty"] = 0.0  # Initialize uncertainty

# Overwrite query cells with predictions
embed.obs.loc[query_mask, CT_KEY + "_pred"] = p
embed.obs.loc[query_mask, CT_KEY + "_uncertainty"] = u

# Filter low-confidence predictions
UNCERTAINTY_THRESHOLD = 0.8
uncertain_mask = (query_mask) & (embed.obs[CT_KEY + "_uncertainty"] > UNCERTAINTY_THRESHOLD)
print(f"\nPrediction Summary:")
print(f"Total query cells: {sum(query_mask)}")
print(f"Unknown predictions: {sum(uncertain_mask)} ({sum(uncertain_mask)/sum(query_mask)*100:.1f}%)")
print(f"Mean uncertainty: {embed.obs.loc[query_mask, CT_KEY + '_uncertainty'].mean():.3f}")
embed.obs.loc[uncertain_mask, CT_KEY + "_pred"] = "Unknown"

# Save results
print('\nSaving annotated embedding...')
embed.write_h5ad(OUTPUT_MASTER_DIR / 'embed_with_knn_predictions.h5ad')

# Visualize using existing UMAP
# No need to recalculate - you already have it in X_umap
print('\nCreating UMAP visualizations...')

# Plot reference vs query
sc.pl.umap(embed, color=REF_KEY, frameon=False, 
           save='_ref_vs_query.png', use_raw=False)
# Plot predictions
sc.pl.umap(embed, color=CT_KEY + "_pred", frameon=False, 
           save='_predictions.png', use_raw=False)
# Plot uncertainty
sc.pl.umap(embed, color=CT_KEY + "_uncertainty", frameon=False, 
           save='_uncertainty.png', use_raw=False, cmap='viridis')
# Optional: Plot actual vs predicted for reference cells (validation)
sc.pl.umap(embed, color=[CT_KEY, CT_KEY + "_pred"], frameon=False, 
           save='_actual_vs_pred.png', use_raw=False)