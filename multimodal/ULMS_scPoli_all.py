#!/usr/bin/env python
# coding: utf-8

# # scPoli to integrate ULMS G4X dataset into ULMS scRNAseq reference
# - decided to run this on all the cells
# - https://docs.scarches.org/en/latest/scpoli_surgery_pipeline.html

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
import seaborn as sns
from scarches.models.scpoli import scPoli
from sklearn.metrics import classification_report
import h5py
from anndata._io.specs import read_elem
from pytorch_lightning.callbacks import Callback

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi

print(f"Running script: {Path(__file__).name}")

print(torch.cuda.is_available())

# version control
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("seaborn:", sns.__version__)
print("scvi:", scvi.__version__)

mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.rcParams['axes.facecolor'] = 'white'
plt.interactive = False
plt.ioff()
sc.settings.autoshow = False

sc.settings.n_jobs = -1  # Use all available cores
scvi.settings.seed = 1234
torch.set_float32_matmul_precision("high")

current_dir = Path.cwd()
parent_dir = current_dir.parent
print(parent_dir)

# Making an output directory using the pathlib package
output_dir = jpascvi.create_output_dir(parent_dir, 'scPoli_all', change_dir=True)

early_stopping_kwargs = {
    "early_stopping_metric" : "val_prototype_loss",
    "mode" : "min",
    "threshold" : 0,
    "patience" : 20,
    "reduce_lr" : True,
    "lr_patience" : 13,
    "lr_factor" : 0.1,
}
# The conditions key specify the covariates over which to integrate your samples
# batch is the sequencing batch (scRNAseq) 
# section (G4X, like in resolVI)
# sample is the consistent patient sample numbering (G4X and scRNAseq)
# assay is scRNAseq or spatial
condition_keys = ['batch', 'sample', 'section', 'assay']
cell_type_key = 'cell_type'
SOURCE_ETA = 2
QUERY_ETA = 5

class MetricNameFinder(Callback):
    '''
    Store the scPoli training metrics
    '''
    def __init__(self):
        super().__init__()
        self.found_metrics = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.found_metrics.append(list(trainer.callback_metrics.keys()))
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.found_metrics.append(list(trainer.callback_metrics.keys()))


# # Import G4X data, which will eventually be the query data
print("\nLoading G4X anndata")
data_dir = parent_dir.parent / 'G4X/objects'
print(data_dir)
g4x_adata = sc.read_h5ad(data_dir / 'g4x_raw_counts.h5ad')
print(g4x_adata)

# reformat the G4X anndata
print("\nReformatting G4X anndata")
g4x_adata.obs_names = g4x_adata.obs['cell_name']
g4x_adata.obs['assay'] = 'spatial'
g4x_adata.obs.rename(columns={'Sample' : 'sample', 'Section' : 'section'}, inplace=True)
g4x_adata.obs['cell_type'] = 'Unknown'
g4x_adata.obs['batch'] = 'g4x'
g4x_adata.obs['batch'] = g4x_adata.obs['batch'].astype('category')
gene_list = g4x_adata.var_names.tolist()
target_adata = g4x_adata
del g4x_adata
# need to convert datatype to avoid training issues
target_adata.X = target_adata.X.astype(np.float32)
print(target_adata)

# add in the celltype annotations
print("\nLoading G4X annotations")
data_dir = parent_dir.parent / 'G4X/annotation'
print(data_dir)
# for anndata compatibility issues, you can read in elements of anndata
# https://github.com/scverse/anndata/issues/436
with h5py.File(data_dir / 'scviva_celltype.h5ad') as f:
    mtx = read_elem(f["X"])
    obs = read_elem(f["obs"])
    var = read_elem(f["var"])
    # uns = read_elem(f["uns"]) # can comment this out if there are issues
    obsm = read_elem(f["obsm"])
    obsp = read_elem(f["obsp"])
    counts = read_elem(f["layers/counts"])
g4x_ann = ad.AnnData(
    X=counts,
    obs=obs,
    var=var,
    obsm=obsm,
    obsp=obsp,
)
g4x_ann.obs_names = g4x_ann.obs['cell_name']
print(g4x_ann)

target_adata.obs['cell_type'] = g4x_ann.obs.loc[target_adata.obs.index, 'celltype']
print(target_adata.obs['cell_type'])

print(np.unique(target_adata.obs['cell_type']))
target_adata = target_adata[target_adata.obs['cell_type'] != 'Necrosis'].copy()
target_adata.obs['cell_type'] = target_adata.obs['cell_type'].cat.rename_categories(
    {'Macrophage': 'Myeloid'}
)
print(np.unique(target_adata.obs['cell_type']))
print(target_adata)
del g4x_ann


# # Import scRNAseq data, which will be the reference data
print("\nLoading scRNAseq anndata")
data_dir = parent_dir.parent / 'scRNAseq/objects'
print(data_dir)
adata = ad.read_h5ad(data_dir / 'annotated_raw_counts.h5ad')
print(adata)

# Reformat the scRNAseq anndata for training, subsetting to only those genes present in the G4X anndata
print("\nReformatting scRNAseq anndata")
adata.obs['assay'] = 'scRNAseq'
adata.obs.rename(columns={'celltype' : 'cell_type'}, inplace=True)
adata.obs['sample'] = adata.obs['sample'].str.replace('Sample', '')
adata.obs['sample'] = adata.obs['sample'].astype('category')
adata.obs['section'] = 'scRNAseq'
adata.obs['section'] = adata.obs['section'].astype('category')
# subset to the G4X gene list
gene_list = [gene for gene in gene_list if gene in adata.var_names.tolist()]
adata = adata[:, gene_list].copy()
print(adata)
source_adata = adata
del adata
source_adata.X = source_adata.X.astype(np.float32)
print(source_adata)

# Note: make sure the cell_type categories are named the same thing in source and target
# e.g. not T_cells and T_cell
print(source_adata.obs['cell_type'].cat.categories)
# Intersection
print(set(target_adata.obs['cell_type'].cat.categories) & set(source_adata.obs['cell_type'].cat.categories))
# Union
print(set(target_adata.obs['cell_type'].cat.categories) | set(source_adata.obs['cell_type'].cat.categories))

# in case there are any genes in the G4X that are not in the scRNAseq, though that is unlikely
target_adata = target_adata[:, source_adata.var_names.tolist()].copy()
print(target_adata)

print("\nFinal scRNAseq anndata:")
print(source_adata)
print("Here are the types of the batch variables:")
for col in source_adata.obs.columns:
    if col in condition_keys:
        print(f"{col}: {source_adata.obs[col].dtype}")
print("Let's check for NaN in each of the batch variables:")
for col in source_adata.obs.columns:
    if col in condition_keys:
        print(f"{col}: {source_adata.obs[col].isnull().sum()} NaN values")

print("\nFinal G4X anndata:")
print(target_adata)
print("Here are the types of the batch variables:")
for col in target_adata.obs.columns:
    if col in condition_keys:
        print(f"{col}: {target_adata.obs[col].dtype}")
print("Let's check for NaN in each of the batch variables:")
for col in target_adata.obs.columns:
    if col in condition_keys:
        print(f"{col}: {target_adata.obs[col].isnull().sum()} NaN values")


# # Train the reference scPoli model on fully labeled scRNAseq data

source_dir = jpascvi.create_output_dir(output_dir, 'scpoli_model', change_dir=True)

print("\nFinal scRNAseq anndata:")
print(source_adata)
print("\nFinal G4X anndata:")
print(target_adata)

finder = MetricNameFinder()

scpoli_model = scPoli(
    adata=source_adata,
    condition_keys=condition_keys,
    cell_type_keys=cell_type_key,
    recon_loss='nb',
)
scpoli_model.train(
    n_epochs=50,
    pretraining_epochs=40, # recommended pretraining/training epoch ratio of 80-90%
    early_stopping_kwargs=early_stopping_kwargs,
    eta=SOURCE_ETA, # the weight of the prototype loss, i.e. higher value means more emphasis on putting cell types together
    callbacks=[finder]
)
try:
    scpoli_model.save(source_dir, overwrite=True, save_anndata=True)
except Exception as e:
    print(e)

# Inspect AFTER training completes
print("Here are the available metrics:")
print(finder.found_metrics)


# # Reference mapping of query G4X data into integrated scRNAseq reference atlas 

target_dir = jpascvi.create_output_dir(output_dir, 'scpoli_query', change_dir=True)

finder = MetricNameFinder()

scpoli_query = scPoli.load_query_data(
    adata=target_adata,
    reference_model=scpoli_model,
    labeled_indices=[],
)
scpoli_query.train(
    n_epochs=50,
    pretraining_epochs=40,
    eta=QUERY_ETA, # the weight of the prototype loss, i.e. higher value means more emphasis on putting cell types together
    callbacks=[finder]
)
try:
    scpoli_query.save(target_dir, overwrite=True, save_anndata=True)
except Exception as e:
    print(e)

# Inspect AFTER training completes
print("Here are the available metrics:")
print(finder.found_metrics)


# # Inspect the results
jpascvi.create_output_dir(output_dir, 'scpoli_results', change_dir=True)
results_dict = scpoli_query.classify(target_adata, scale_uncertainties=True)

for i in range(len(cell_type_key)):
    preds = results_dict[cell_type_key]["preds"]
    results_dict[cell_type_key]["uncert"]
    classification_df = pd.DataFrame(
        classification_report(
            y_true=target_adata.obs[cell_type_key],
            y_pred=preds,
            output_dict=True,
        )
    ).transpose()
print(classification_df)
classification_df.to_csv('scpoli_all_classification_df.csv')

# Get latent representation of reference data
scpoli_query.model.eval()
data_latent_source = scpoli_query.get_latent(source_adata, mean=True)
adata_latent_source = sc.AnnData(data_latent_source)
adata_latent_source.obs = source_adata.obs.copy()

# Get latent representation of query data
data_latent= scpoli_query.get_latent(target_adata, mean=True)
adata_latent = sc.AnnData(data_latent)
adata_latent.obs = target_adata.obs.copy()

# Get label annotations
adata_latent.obs['cell_type_pred'] = results_dict['cell_type']['preds'].tolist()
adata_latent.obs['cell_type_uncert'] = results_dict['cell_type']['uncert'].tolist()
adata_latent.obs['classifier_outcome'] = (adata_latent.obs['cell_type_pred'] == adata_latent.obs['cell_type'])

# Get prototypes
labeled_prototypes = scpoli_query.get_prototypes_info()
labeled_prototypes.obs['assay'] = 'labeled_prototype'
unlabeled_prototypes = scpoli_query.get_prototypes_info(prototype_set='unlabeled')
unlabeled_prototypes.obs['assay'] = 'unlabeled_prototype'

# Concatenate the anndatas
adata_latent_full = ad.concat(
    [adata_latent_source, adata_latent, labeled_prototypes, unlabeled_prototypes],
    join='inner'
)
adata_latent_full.obs.loc[adata_latent_full.obs['assay'] == 'scRNAseq', 'cell_type_pred'] = np.nan
sc.pp.neighbors(adata_latent_full)
sc.tl.umap(adata_latent_full)

# Save the full anndata
print(adata_latent_full)
adata_latent_full.write_h5ad('scpoli_all.h5ad')

# Get adata without prototypes
adata_no_prototypes = adata_latent_full[adata_latent_full.obs['assay'].isin(['scRNAseq', 'spatial'])]
try:
    sc.pl.umap(
        adata_no_prototypes,
        color='assay',
        show=False,
        frameon=False,
        save='assay.png'
    )
    sc.pl.umap(
        adata_no_prototypes,
        color='cell_type_pred',
        show=False,
        frameon=False,
        save='cell_type_pred.png'
    )
    sc.pl.umap(
        adata_no_prototypes,
        color='cell_type_uncert',
        show=False,
        frameon=False,
        cmap='magma',
        vmax=1, 
        save='cell_type_uncert.png'
    )
    sc.pl.umap(
        adata_no_prototypes,
        color='classifier_outcome',
        show=False,
        frameon=False,
        save='classifier_outcome.png'
    )
    sc.pl.umap(
        adata_no_prototypes,
        color='cell_type',
        show=False,
        frameon=False,
        save='cell_type.png'
    )
    sc.pl.umap(
        adata_no_prototypes,
        color='sample',
        show=False,
        frameon=False,
        save='sample.png'
    )
    sc.pl.umap(
        adata_no_prototypes,
        color='batch',
        show=False,
        frameon=False,
        save='batch.png'
    )
    sc.pl.umap(
        adata_no_prototypes,
        color='section',
        show=False,
        frameon=False,
        save='section.png'
    )
except Exception as e:
    print(e)


# Examine the prototypes
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
adata_labeled_prototypes = adata_latent_full[adata_latent_full.obs['assay'] == 'labeled_prototype']
adata_unlabeled_prototypes = adata_latent_full[adata_latent_full.obs['assay'] == 'unlabeled_prototype']
adata_labeled_prototypes.obs['cell_type_pred'] = adata_labeled_prototypes.obs['cell_type_pred'].astype('category')
adata_unlabeled_prototypes.obs['cell_type_pred'] = adata_unlabeled_prototypes.obs['cell_type_pred'].astype('category')
adata_unlabeled_prototypes.obs['cell_type'] = adata_unlabeled_prototypes.obs['cell_type'].astype('category')

sc.pl.umap(
    adata_no_prototypes,
    alpha=0.2,
    show=False,
    ax=ax
)
ax.legend([])
# plot labeled prototypes
sc.pl.umap(
    adata_labeled_prototypes,
    size=200,
    color=f'{cell_type_key}_pred',
    ax=ax,
    show=False,
    frameon=False,
)
cell_types = adata_labeled_prototypes.obs[f'{cell_type_key}_pred'].cat.categories
color_ct = adata_labeled_prototypes.uns[f'{cell_type_key}_pred_colors']
color_dict = dict(zip(cell_types, color_ct))
# plot labeled prototypes
sc.pl.umap(
    adata_unlabeled_prototypes,
    size=100,
    color=f'{cell_type_key}_pred',
    palette=color_dict,
    ax=ax,
    show=False,
    frameon=False,
    alpha=0.5,
)
sc.pl.umap(
    adata_unlabeled_prototypes,
    size=0,
    color=cell_type_key,
    #palette=color_dict,
    frameon=False,
    show=False,
    ax=ax,
    legend_loc='on data',
    legend_fontsize=5,
)
ax.set_title('Landmarks')
h, l = ax.get_legend_handles_labels()
ax.legend().remove()
ax.legend(handles=h[:13], labels= l[:13], frameon=False, bbox_to_anchor=(1, 1))
fig.tight_layout()
plt.savefig('prototypes.png')
plt.close()

