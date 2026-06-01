# # scANVI to integrate ULMS G4X dataset into ULMS scRNAseq reference
# - now running on just the tumor subset
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/scarches_scvi_tools.html
# https://discourse.scverse.org/t/increase-scvi-integration-speed/1772/5
# https://discourse.scverse.org/t/scvi-tools-label-transfer-accuracy/1503

# Training reference without subsetting to G4X genes

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
import scipy.sparse as sp
from scipy.stats import entropy

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
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

G4X_DIR = PARENT_DIR.parent / 'G4X'
print(G4X_DIR)

SCRNASEQ_DIR = PARENT_DIR.parent / 'scRNAseq'
print(SCRNASEQ_DIR)

# Making an output directory using the pathlib package
OUTPUT_MASTER_DIR = jpascvi.create_output_dir(PARENT_DIR, 'scANVI_tumor_allgenes')

# USER-DEFINED PARAMETERS

early_stopping_kwargs = {
    'check_val_every_n_epoch': 1,
    'early_stopping': True,
    'early_stopping_patience': 20, # how many epochs of no change are tolerated
    'early_stopping_monitor': "elbo_validation"
}
# The conditions key specify the covariates over which to integrate your samples
# batch is the sequencing batch (scRNAseq) 
# section (G4X, like in resolVI)
# sample is the consistent patient sample numbering (G4X and scRNAseq)
# assay is scRNAseq or spatial
tech_key = 'assay'
condition_keys = ['batch', 'sample', 'section']
cell_type_key = 'tumor_subtype'
SCRNASEQ_MIN_COUNTS = 10 # remove scRNAseq cells with fewer than this many counts after subsetting to the G4X gene list, which will be the genes used for integration
SCVI_MAX_EPOCHS = 50
SCANVI_MAX_EPOCHS = 50
REF_MAP_MAX_EPOCHS = 100
SCVI_LATENT_KEY = "X_scVI"
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTIONS_KEY = "scanvi_pred"
# https://github.com/scverse/scvi-tools/issues/2726
QUERY_TRAIN_BATCH_SIZE = 512
# SCANVI_N_SAMPLES_PER_LABEL = 1000 # trying to increase the representation of rare cell types for better integration

# LOAD THE G4X QUERY DATA

# # Import G4X data, which will eventually be the query data
print("\nLoading G4X anndata")
data_dir = G4X_DIR / 'objects'
print(data_dir)
g4x_adata = sc.read_h5ad(data_dir / 'g4x_raw_counts.h5ad')
print(g4x_adata)

# reformat the G4X anndata
print("\nReformatting G4X anndata")
g4x_adata.obs_names = g4x_adata.obs['cell_name']
g4x_adata.obs[tech_key] = 'spatial'
g4x_adata.obs.rename(columns={'Sample' : 'sample', 'Section' : 'section'}, inplace=True)
g4x_adata.obs['cell_type'] = 'Unknown'
g4x_adata.obs['batch'] = 'g4x'
g4x_adata.obs['batch'] = g4x_adata.obs['batch'].astype('category')
g4x_adata.obs['sample'] = g4x_adata.obs['sample'].astype('category')
g4x_adata.obs['section'] = g4x_adata.obs['section'].astype('category')
g4x_adata.obs[tech_key] = g4x_adata.obs[tech_key].astype('category')
adata_query = g4x_adata
del g4x_adata
print(adata_query)

# add in the celltype annotations
print("\nLoading G4X annotations")
data_dir = G4X_DIR / 'annotation'
print(data_dir)
g4x_ann = sc.read_h5ad(data_dir / 'scviva_celltype.h5ad')
g4x_ann.obs_names = g4x_ann.obs['cell_name']
print(g4x_ann)
adata_query.obs['cell_type'] = g4x_ann.obs.loc[adata_query.obs.index, 'celltype']
print(adata_query.obs['cell_type'])
print(np.unique(adata_query.obs['cell_type']))
adata_query = adata_query[adata_query.obs['cell_type'] != 'Necrosis'].copy()
adata_query.obs['cell_type'] = adata_query.obs['cell_type'].cat.remove_unused_categories()
adata_query.obs['cell_type'] = adata_query.obs['cell_type'].cat.rename_categories(
    {'Macrophage': 'Myeloid'}
)
print(np.unique(adata_query.obs['cell_type']))
print(adata_query)
del g4x_ann

# subset for tumor cells
print("\nSubsetting for tumor cells")
adata_query = adata_query[adata_query.obs['cell_type'] == 'Tumor'].copy()
print(adata_query)

# Set the tumor subtypes to unknown
adata_query.obs[cell_type_key] = 'Unknown'
adata_query.obs[cell_type_key] = adata_query.obs[cell_type_key].astype('category')

# LOAD THE SCRNASEQ REFERENCE DATA

# # Import scRNAseq data, which will be the reference data
print("\nLoading scRNAseq anndata")
data_dir = SCRNASEQ_DIR / 'objects'
print(data_dir)
adata_ref = ad.read_h5ad(data_dir / 'annotated_raw_counts.h5ad')
print(adata_ref)

# Reformat the scRNAseq anndata for training, subsetting to only those genes present in the G4X anndata
print("\nReformatting scRNAseq anndata")
adata_ref.obs[tech_key] = 'scRNAseq'
adata_ref.obs.rename(columns={'celltype' : 'cell_type'}, inplace=True)
adata_ref = adata_ref[adata_ref.obs['cell_type'] != 'RBC'].copy()
adata_ref.obs['cell_type'] = adata_ref.obs['cell_type'].astype('category')
adata_ref.obs['cell_type'] = adata_ref.obs['cell_type'].cat.remove_unused_categories()
adata_ref.obs['sample'] = adata_ref.obs['sample'].str.replace('Sample', '')
adata_ref.obs['sample'] = adata_ref.obs['sample'].astype('category')
adata_ref.obs['section'] = 'scRNAseq'
adata_ref.obs['section'] = adata_ref.obs['section'].astype('category')
adata_ref.obs[tech_key] = adata_ref.obs[tech_key].astype('category')
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
ann = ad.read_h5ad(data_dir / 'tumor_annotated.h5ad')
ann.obs[cell_type_key] = ann.obs['celltype'] # renaming
adata_ref.obs[cell_type_key] = ann.obs.loc[adata_ref.obs.index, cell_type_key]
adata_ref.obs[cell_type_key] = adata_ref.obs[cell_type_key].astype('category')
print(np.unique(adata_ref.obs[cell_type_key]))
del ann

# Prepare for scVI/scANVI
print("\nFinal scRNAseq anndata:")
print(adata_ref)
print(np.unique(adata_ref.obs['cell_type']))
print("\nFinal G4X anndata:")
print(adata_query)
print(np.unique(adata_query.obs['cell_type']))

# Log normalize data and save raw counts in a layer, as recommended for scvi-tools
adata_ref.layers["counts"] = adata_ref.X.copy() # this layer will contain the raw counts
sc.pp.normalize_total(adata_ref) # normalize X to the median total counts
sc.pp.log1p(adata_ref) # logarithmize X
adata_ref.raw = adata_ref # full dimension normalized logtransformed raw data


# SCVI TRAINING OF THE REFERENCE DATA
print("\nTraining scVI reference model\n")
scvi_ref_dir = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scvi_ref', change_figdir=True)

# # Train the reference scVI model on fully labeled scRNAseq data
# batch_key=tech_key is a single value here but necessary for scArches 
# to handle the new 'spatial' batch during query mapping
scvi.model.SCVI.setup_anndata(adata_ref, 
                              layer="counts", 
                              batch_key=tech_key, 
                              categorical_covariate_keys=condition_keys)
# Using custom parameters found to work well in scArches
scvi_ref = scvi.model.SCVI(adata_ref, 
                           gene_likelihood="nb", # choosing this since spatial will be less sparse
                           use_layer_norm="both", # scArches found that using layer norm in both the encoder and decoder worked best for integration
                           use_batch_norm="none", # scArches found that not using batch norm in the decoder worked best for integration
                           encode_covariates=True, # Necesary for scArches to map new batches in the encoder
                           dropout_rate=0.2,
                           n_layers=2,)
print(scvi_ref)

# Train the vae with early stopping for the default number of epochs
scvi.settings.seed = 1234
scvi_ref.train(max_epochs=SCVI_MAX_EPOCHS, **early_stopping_kwargs)
save_path = scvi_ref_dir / 'scvi_ref_elbo_plot.png'
jpascvi.check_training(scvi_ref, save=save_path)

# Get the latent representation of the reference data
adata_ref.obsm[SCVI_LATENT_KEY] = scvi_ref.get_latent_representation()
sc.pp.neighbors(adata_ref, use_rep=SCVI_LATENT_KEY)
sc.tl.umap(adata_ref, min_dist=0.3, random_state=SEED)

# Visual check of integration
sc.pl.umap(
    adata_ref,
    color=cell_type_key,
    show=False,
    frameon=False,
    save=f'{cell_type_key}.png'
)
sc.pl.umap(
    adata_ref,
    color='sample',
    show=False,
    frameon=False,
    save='sample.png'
)
sc.pl.umap(
    adata_ref,
    color='batch',
    show=False,
    frameon=False,
    save='batch.png'
)
# save the model and anndata with the latent representation
scvi_ref.save(scvi_ref_dir, prefix="scvi_ref_", save_anndata=True, overwrite=True)
print("scVI reference model saved to ", scvi_ref_dir)


# TRAINING THE SCANVI REFERENCE MODEL
print("\nTraining scANVI reference model\n")
scanvi_ref_dir = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scanvi_ref', change_figdir=True)

scanvi_ref = scvi.model.SCANVI.from_scvi_model(
    scvi_ref,
    adata=adata_ref,
    labels_key=cell_type_key,
    unlabeled_category="Unknown",
)
scvi.settings.seed = 1234
scanvi_ref.train(max_epochs=SCANVI_MAX_EPOCHS, **early_stopping_kwargs)
save_path = scanvi_ref_dir / 'scanvi_ref_elbo_plot.png'
jpascvi.check_training(scanvi_ref, save=save_path)

# Get the latent representation of the reference data
adata_ref.obsm[SCANVI_LATENT_KEY] = scanvi_ref.get_latent_representation()
sc.pp.neighbors(adata_ref, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(adata_ref, min_dist=0.3, random_state=SEED)

# Visual check of integration
sc.pl.umap(
    adata_ref,
    color=cell_type_key,
    show=False,
    frameon=False,
    save=f'{cell_type_key}.png'
)
sc.pl.umap(
    adata_ref,
    color='sample',
    show=False,
    frameon=False,
    save='sample.png'
)
sc.pl.umap(
    adata_ref,
    color='batch',
    show=False,
    frameon=False,
    save='batch.png'
)
# save the model and anndata with the latent representation
scanvi_ref.save(scanvi_ref_dir, prefix="scanvi_ref_", save_anndata=True, overwrite=True)
print("scANVI reference model saved to ", scanvi_ref_dir)

# clean up memory before training the query model
del scvi_ref
torch.cuda.empty_cache()


# # REFERENCE MAPPING OF QUERY G4X DATA ONTO AN INTEGRATED SCRNASEQ REFERENCE ATLAS USING SCANVI
print("\nMapping query G4X data onto reference atlas with scANVI\n")
scanvi_query_dir = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scanvi_query', change_figdir=True)

# Log normalize data and save raw counts in a layer, as recommended for scvi-tools
adata_query.layers["counts"] = adata_query.X.copy() # this layer will contain the raw counts
# Densify the counts layer to speed up training
if sp.issparse(adata_query.layers["counts"]):
    adata_query.layers["counts"] = adata_query.layers["counts"].toarray()
    print("Densified counts layer")
sc.pp.normalize_total(adata_query) # normalize X to the median total counts
sc.pp.log1p(adata_query) # logarithmize X
adata_query.raw = adata_query # full dimension normalized logtransformed raw data

# Prepare the query anndata by reordering the genes and padding any missing genes with zeros
scvi.model.SCANVI.prepare_query_anndata(adata_query, scanvi_ref)
# Online update of reference model using the scArches algorithm
scanvi_query = scvi.model.SCANVI.load_query_data(adata_query, scanvi_ref)

# Train the model
# Weight decay of 0.0 ensures the latent representation of the reference cells remains the same
scvi.settings.seed = 1234
scanvi_query.train(**early_stopping_kwargs, 
                   max_epochs=REF_MAP_MAX_EPOCHS, 
                   plan_kwargs={"weight_decay": 0.0},
                   batch_size=QUERY_TRAIN_BATCH_SIZE,
                   )
save_path = scanvi_query_dir / 'scanvi_query_elbo_plot.png'
jpascvi.check_training(scanvi_query, save=save_path)

# Get the latent space
adata_query.obsm[SCANVI_LATENT_KEY] = scanvi_query.get_latent_representation()
sc.pp.neighbors(adata_query, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(adata_query, min_dist=0.3, random_state=SEED)

# Visual check of integration
sc.pl.umap(
    adata_query,
    color=cell_type_key,
    show=False,
    frameon=False,
    save=f'{cell_type_key}.png'
)
sc.pl.umap(
    adata_query,
    color='sample',
    show=False,
    frameon=False,
    save='sample.png'
)
sc.pl.umap(
    adata_query,
    color='section',
    show=False,
    frameon=False,
    save='section.png'
)
# save the model and anndata with the latent representation
scanvi_query.save(scanvi_query_dir, prefix="scanvi_query_", save_anndata=True, overwrite=True)
print("scANVI query model saved to ", scanvi_query_dir)

# SCANVI CELL TYPE PREDICTIONS
print("\nGetting scANVI cell type predictions for query G4X data\n")
# Get the cell type predictions for the query data
adata_query.obs[SCANVI_PREDICTIONS_KEY] = scanvi_query.predict()

sc.pl.umap(
    adata_query,
    color=SCANVI_PREDICTIONS_KEY,
    show=False,
    frameon=False,
    save=f'{SCANVI_PREDICTIONS_KEY}.png'
)
adata_path = scanvi_query_dir / 'adata_query.h5ad'
adata_query.write_h5ad(adata_path)

# ANALYZING THE QUERY AND REFERENCE TOGETHER
print("\nAnalyzing query and reference together\n")
SCANVI_FULL_DIR = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scanvi_full', change_figdir=True)

adata = ad.concat([adata_query, adata_ref], join='inner')
print(adata)

# soft predictions gives you a probability distribution with confidence score
prob_df = scanvi_query.predict(adata, soft=True)
adata.obs[SCANVI_PREDICTIONS_KEY] = prob_df.idxmax(axis=1)
adata.obs['scANVI_confidence'] = prob_df.max(axis=1)
adata.obs['scANVI_entropy'] = entropy(prob_df.values, axis=1)

prob_df.to_csv(SCANVI_FULL_DIR / 'full_predictions.csv')

# Summary
print(f"High confidence (≥0.5): {(adata.obs['scANVI_confidence'] >= 0.5).sum()} cells")
print(f"Low confidence (<0.5): {(adata.obs['scANVI_confidence'] < 0.5).sum()} cells")

sc.pp.neighbors(adata, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(adata, min_dist=0.3, random_state=SEED)

sc.pl.umap(
    adata,
    color=tech_key,
    show=False,
    frameon=False,
    save=f'{tech_key}.png'
)
sc.pl.umap(
    adata,
    color='sample',
    show=False,
    frameon=False,
    save='sample.png'
)
sc.pl.umap(
    adata,
    color='batch',
    show=False,
    frameon=False,
    save='batch.png'
)
sc.pl.umap(
    adata,
    color='section',
    show=False,
    frameon=False,
    save='section.png'
)
sc.pl.umap(
    adata,
    color=cell_type_key,
    show=False,
    frameon=False,
    save=f'{cell_type_key}.png'
)
sc.pl.umap(
    adata,
    color=SCANVI_PREDICTIONS_KEY,
    show=False,
    frameon=False,
    save=f'{SCANVI_PREDICTIONS_KEY}.png'
)

adata_path = SCANVI_FULL_DIR / 'scanvi_full_adata.h5ad'
adata.write_h5ad(adata_path)
print("Full anndata with query and reference together saved to ", adata_path)
