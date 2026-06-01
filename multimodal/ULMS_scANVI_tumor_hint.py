# # scANVI to integrate ULMS G4X dataset into ULMS scRNAseq reference
# now running on just the tumor subset
# HVGs only
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/scarches_scvi_tools.html
# https://discourse.scverse.org/t/increase-scvi-integration-speed/1772/5
# https://discourse.scverse.org/t/scvi-tools-label-transfer-accuracy/1503

# TOP 10 expressed genes per cluster in the scRNAseq plus hints

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
OUTPUT_MASTER_DIR = jpascvi.create_output_dir(PARENT_DIR, 'scANVI_tumor_hint', change_figdir=True)

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
N_TOP_GENES = 10 # Number of top differentially expressed genes per cluster that will be kept as features in the model
MIN_COUNTS = 10 # remove scRNAseq cells with fewer than this many counts after subsetting to the G4X gene list, which will be the genes used for integration
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
adata_g4x = g4x_adata
del g4x_adata
print(adata_g4x)

# add in the tumor_subtype annotations
print("\nLoading G4X annotations")
data_dir = G4X_DIR / 'scviva_tumor'
print(data_dir)
g4x_ann = sc.read_h5ad(data_dir / 'tumor_annotated.h5ad')
g4x_ann.obs_names = g4x_ann.obs['cell_name']
print(g4x_ann)
adata_g4x = adata_g4x[g4x_ann.obs.index, :].copy() # subset for tumor cells
adata_g4x.obs[cell_type_key] = g4x_ann.obs.loc[adata_g4x.obs.index, cell_type_key]
print(adata_g4x.obs[cell_type_key])
print(np.unique(adata_g4x.obs[cell_type_key]))
print(adata_g4x)
del g4x_ann

# Mask all tumor subtypes except the ones you will use to provide a hint to scANVI
keep_subtypes = ['ESR1 PGR AR Tumor']  # easy to extend this list
adata_g4x.obs[cell_type_key] = (
    adata_g4x.obs[cell_type_key]
    .astype('str')
    .where(adata_g4x.obs[cell_type_key].isin(keep_subtypes), other='Unknown')
    .astype('category')
)
adata_g4x.obs[cell_type_key] = adata_g4x.obs[cell_type_key].cat.remove_unused_categories()
print(adata_g4x.obs[cell_type_key])

# Log normalize data and save raw counts in a layer, as recommended for scvi-tools
adata_g4x.layers["counts"] = adata_g4x.X.copy() # this layer will contain the raw counts
sc.pp.normalize_total(adata_g4x) # normalize X to the median total counts
sc.pp.log1p(adata_g4x) # logarithmize X

# LOAD THE SCRNASEQ REFERENCE DATA

# # Import scRNAseq data, which will be the reference data
print("\nLoading scRNAseq anndata")
data_dir = SCRNASEQ_DIR / 'objects'
print(data_dir)
adata_ref = ad.read_h5ad(data_dir / 'annotated_raw_counts.h5ad')
print(adata_ref)

# Reformat the scRNAseq anndata for training
print("\nReformatting scRNAseq anndata")
adata_ref.obs[tech_key] = 'scRNAseq'
adata_ref.obs.rename(columns={'celltype' : 'cell_type'}, inplace=True)
adata_ref.obs['cell_type'] = adata_ref.obs['cell_type'].astype('category')
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
adata_ref.obs[cell_type_key] = 'Unknown' # initialize with unknown
adata_ref.obs[cell_type_key] = ann.obs.loc[adata_ref.obs.index, cell_type_key]
del ann

# Combine the ESR PGR AR cells to make it the same as G4X
tumor_map = {
    'ESR1+ cells': 'ESR1 PGR AR Tumor',
    'AR+/PGR+ cells': 'ESR1 PGR AR Tumor'
}
adata_ref.obs[cell_type_key] = (
    adata_ref.obs[cell_type_key]
    .replace(tumor_map) # keeps the other scRNAseq annotations as they are
    .astype('category')
)
adata_ref.obs[cell_type_key] = adata_ref.obs[cell_type_key].cat.remove_unused_categories()
print(adata_ref.obs[cell_type_key])


# Log normalize data and save raw counts in a layer, as recommended for scvi-tools
adata_ref.layers["counts"] = adata_ref.X.copy() # this layer will contain the raw counts
sc.pp.normalize_total(adata_ref) # normalize X to the median total counts
sc.pp.log1p(adata_ref) # logarithmize X
# subset to the intersection of both gene lists
common_genes = adata_ref.var_names.intersection(adata_g4x.var_names).tolist()
adata_ref = adata_ref[:, common_genes].copy()
adata_g4x = adata_g4x[:, common_genes].copy()
adata_ref.raw = adata_ref # normalized logtransformed raw data but only the G4X genes

# FEATURE SELECTION

# calculate the top differentially expressed genes in each scRNAseq tumor cluster
# Must have log normalized first in adata.raw
sc.tl.rank_genes_groups(adata_ref, groupby=cell_type_key, method="wilcoxon")
rank_genes_filename = 'scrnaseq_tumor_subtype_top_genes.png'
sc.pl.rank_genes_groups_dotplot(adata_ref, groupby=cell_type_key, standard_scale="var", n_genes=N_TOP_GENES, save=rank_genes_filename)
rank_genes_filename = 'scrnaseq_tumor_subtype_top_genes.pdf'
sc.pl.rank_genes_groups_dotplot(adata_ref, groupby=cell_type_key, standard_scale="var", n_genes=N_TOP_GENES, save=rank_genes_filename)

# saving the dataframe with all the degs
de_df = sc.get.rank_genes_groups_df(adata_ref, group=None)
csv_path = OUTPUT_MASTER_DIR / 'scrnaseq_tumor_subtype_all_degs.csv'
de_df.to_csv(csv_path, index=False)
# filter for top degs for each cluster before saving the dataframe
cats = adata_ref.obs[cell_type_key].cat.categories
topgenes = pd.DataFrame()
for c in cats:
    de_filt = (de_df[de_df['group'] == c]
           .query('logfoldchanges > 0 and pvals_adj < 0.05')
           .sort_values('scores', kind='mergesort', ascending=False)
           .head(N_TOP_GENES))
    topgenes = pd.concat([topgenes, de_filt], axis=0)
# write degs df - the filtered one
csv_path = OUTPUT_MASTER_DIR / f'scrnaseq_tumor_subtype_top{N_TOP_GENES}_degs.csv'
topgenes.to_csv(csv_path)

if len(topgenes) == 0:
    raise ValueError("No significant DEGs found after filtering.")
# Subset to the final gene list
topgene_list = topgenes['names'].unique().tolist()
adata_ref = adata_ref[:, topgene_list].copy()
adata_g4x = adata_g4x[:, topgene_list].copy()

print("scRNAseq adata: any cells with low counts after subsetting and filtering?")
# Some cells may have low counts - this may mess up integration and differential expression calculation by creating a division by zero
print(f"Number of cells in anndata: {adata_ref.n_obs}")
# Make sure to use the raw counts layer - in this case just adata.X since there are no layers
low_counts = adata_ref[adata_ref.X.sum(axis=1) < MIN_COUNTS]
print(f"Number of cells with low counts: {low_counts.n_obs}")
adata_ref = adata_ref[adata_ref.X.sum(axis=1) >= MIN_COUNTS].copy()
print('Anndata after filtering out cells with low counts:')
print(adata_ref)

print("G4X adata: any cells with low counts after subsetting and filtering?")
# Some cells may have low counts - this may mess up integration and differential expression calculation by creating a division by zero
print(f"Number of cells in anndata: {adata_g4x.n_obs}")
# Make sure to use the raw counts
low_counts = adata_g4x[adata_g4x.X.sum(axis=1) < MIN_COUNTS]
print(f"Number of cells with low counts: {low_counts.n_obs}")
adata_g4x = adata_g4x[adata_g4x.X.sum(axis=1) >= MIN_COUNTS].copy()
print('Anndata after filtering out cells with low counts:')
print(adata_g4x)

assert all(g in adata_g4x.var_names for g in adata_ref.var_names), "Gene mismatch!"
print("Here are the genes that will be used in the model:")
print(*adata_g4x.var_names.to_list())

# Prepare for scVI/scANVI
print("\nFinal scRNAseq anndata:")
print(adata_ref)
print(np.unique(adata_ref.obs['cell_type']))
del adata_ref.raw
adata_ref.raw = adata_ref # normalized logtransformed raw data but only the final gene list
print("\nFinal G4X anndata:")
print(adata_g4x)
print(np.unique(adata_g4x.obs['cell_type']))
adata_g4x.raw = adata_g4x # normalized logtransformed raw data but only the final gene list




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

# Densify the counts layer to speed up training
if sp.issparse(adata_g4x.layers["counts"]):
    adata_g4x.layers["counts"] = adata_g4x.layers["counts"].toarray()
    print("Densified counts layer")

# Prepare the query anndata by reordering the genes and padding any missing genes with zeros
scvi.model.SCANVI.prepare_query_anndata(adata_g4x, scanvi_ref)
# Online update of reference model using the scArches algorithm
scanvi_query = scvi.model.SCANVI.load_query_data(adata_g4x, scanvi_ref)

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
adata_g4x.obsm[SCANVI_LATENT_KEY] = scanvi_query.get_latent_representation()
sc.pp.neighbors(adata_g4x, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(adata_g4x, min_dist=0.3, random_state=SEED)

# Visual check of integration
sc.pl.umap(
    adata_g4x,
    color=cell_type_key,
    show=False,
    frameon=False,
    save=f'{cell_type_key}.png'
)
sc.pl.umap(
    adata_g4x,
    color='sample',
    show=False,
    frameon=False,
    save='sample.png'
)
sc.pl.umap(
    adata_g4x,
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
adata_g4x.obs[SCANVI_PREDICTIONS_KEY] = scanvi_query.predict()

sc.pl.umap(
    adata_g4x,
    color=SCANVI_PREDICTIONS_KEY,
    show=False,
    frameon=False,
    save=f'{SCANVI_PREDICTIONS_KEY}.png'
)
adata_path = scanvi_query_dir / 'adata_g4x.h5ad'
adata_g4x.write_h5ad(adata_path)

# ANALYZING THE QUERY AND REFERENCE TOGETHER
print("\nAnalyzing query and reference together\n")
SCANVI_FULL_DIR = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scanvi_full', change_figdir=True)

adata = ad.concat([adata_g4x, adata_ref], join='inner')
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
sc.pl.umap(
    adata,
    color='scANVI_confidence',
    show=False,
    frameon=False,
    save='scANVI_confidence.png'
)
sc.pl.umap(
    adata,
    color='scANVI_entropy',
    show=False,
    frameon=False,
    save='scANVI_entropy.png'
)

adata_path = SCANVI_FULL_DIR / 'scanvi_full_adata.h5ad'
adata.write_h5ad(adata_path)
print("Full anndata with query and reference together saved to ", adata_path)
