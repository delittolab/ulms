# Integrate ULMS scRNAseq and G4X data using harmony

# SET UP

# Set up the environment and import necessary libraries
import sys
import numpy as np
import scanpy as sc
import anndata as ad
import harmonypy as hm
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi

print(Path(__file__).name)

# version control
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("matplotlib:", mpl.__version__)
print("seaborn:", sns.__version__)
print("anndata:", ad.__version__)
print("scanpy:", sc.__version__)
print("harmonypy:", hm.__version__)
sc.settings.n_jobs = -1
SEED = 1234
np.random.seed(SEED)

# PARAMETERS
# vars_use:  the variables to use for batch correction. In this case we want to correct for both the assay (G4X vs scRNAseq) and the batch
# theta: diversity clustering penalty parameter, which controls how much harmony will try to mix cells from different batches together. 
# Higher values will result in more aggressive batch correction.
# We want to use a higher theta for the assay variable since we expect more technical differences between G4X and scRNAseq than between different samples or sections
SCRNASEQ_MIN_COUNTS = 10 # remove scRNAseq cells with fewer than this many counts after subsetting to the G4X gene list, which will be the genes used for integration
VARS_USE = ["assay", "batch", "sample", "section"]
THETA = [4, 2, 2, 2]
MAX_ITER_HARMONY = 20

# SET UP DIRECTORIES

CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

G4X_DIR = PARENT_DIR.parent / 'G4X'
print(G4X_DIR)

SCRNASEQ_DIR = PARENT_DIR.parent / 'scRNAseq'
print(SCRNASEQ_DIR)

# Making an output directory using the pathlib package
OUTPUT_MASTER_DIR = jpascvi.create_output_dir(PARENT_DIR, 'harmony_all', change_dir=True)

# FUNCTIONS

def plot_scaled_counts_per_cell(adata, dataset_name, output_dir, bins=50, color='tab:blue'):
    """Save a histogram of counts per cell after scaling."""
    cell_counts = np.array(adata.X.sum(axis=1)).flatten()
    sns.histplot(cell_counts, bins=bins, color=color)
    plt.title(f'{dataset_name} scaled counts per cell')
    plt.xlabel('Scaled total counts')
    plt.ylabel('Number of cells')
    plt.tight_layout()
    filename = f"{dataset_name.lower()}_scaled_counts_per_cell.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


# G4X

# # Import G4X data, which will eventually be the query data
print("\nLoading G4X anndata")
data_dir = G4X_DIR / 'objects'
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
print(g4x_adata)

print(g4x_adata.n_vars)
gene_list = g4x_adata.var_names.tolist()

# add in the celltype annotations
print("\nLoading G4X annotations")
data_dir = G4X_DIR / 'annotation'
print(data_dir)
g4x_ann = sc.read_h5ad(data_dir / 'scviva_celltype.h5ad')
g4x_ann.obs_names = g4x_ann.obs['cell_name']
print(g4x_ann)
g4x_adata.obs['cell_type'] = g4x_ann.obs.loc[g4x_adata.obs.index, 'celltype']
print(g4x_adata.obs['cell_type'])
print(np.unique(g4x_adata.obs['cell_type']))
g4x_adata = g4x_adata[g4x_adata.obs['cell_type'] != 'Necrosis'].copy()
g4x_adata.obs['cell_type'] = g4x_adata.obs['cell_type'].cat.remove_unused_categories()
g4x_adata.obs['cell_type'] = g4x_adata.obs['cell_type'].cat.rename_categories(
    {'Macrophage': 'Myeloid'}
)
print(np.unique(g4x_adata.obs['cell_type']))
print(g4x_adata)
del g4x_ann

# Normalize, plot, log transform, and scale the G4X anndata independently before concatenation with the scRNAseq anndata
g4x_adata.layers['counts'] = g4x_adata.X.copy()  # raw counts layer
sc.pp.normalize_total(g4x_adata, target_sum=1e4)  # normalize counts per cell to a total of 10,000
sc.pp.log1p(g4x_adata)  # logarithmize X within each dataset
g4x_adata.layers['lognorm'] = g4x_adata.X.copy()  # log normalized counts layer
sc.pp.scale(g4x_adata, max_value=10)  # scale X to unit variance and zero mean within each dataset
plot_scaled_counts_per_cell(
    g4x_adata,
    dataset_name='G4X',
    output_dir=OUTPUT_MASTER_DIR,
    color='tab:blue'
)

# SCRNASEQ

# # Import scRNAseq data, which will be the reference data
print("\nLoading scRNAseq anndata")
data_dir = SCRNASEQ_DIR / 'objects'
print(data_dir)
adata = ad.read_h5ad(data_dir / 'annotated_raw_counts.h5ad')
print(adata)

# Reformat the scRNAseq anndata for training, subsetting to only those genes present in the G4X anndata
print("\nReformatting scRNAseq anndata")
adata.obs['assay'] = 'scRNAseq'
adata.obs.rename(columns={'celltype' : 'cell_type'}, inplace=True)
adata = adata[adata.obs['cell_type'] != 'RBC'].copy()
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()
adata.obs['sample'] = adata.obs['sample'].str.replace('Sample', '')
adata.obs['sample'] = adata.obs['sample'].astype('category')
adata.obs['section'] = 'scRNAseq'
adata.obs['section'] = adata.obs['section'].astype('category')
# subset to the G4X gene list
ref_genes = set(adata.var_names)
# gene_list is the G4X gene list from above. Now filter for only those genes that are also in the scRNAseq reference, which will be the genes used for integration
gene_list = [gene for gene in gene_list if gene in ref_genes]
adata = adata[:, gene_list].copy()
scrnaseq_adata = adata
del adata
print(scrnaseq_adata)

# Remove cells with low counts after subsetting to the G4X gene list, which will be the genes used for integration
print("\nFiltering scRNAseq cells with low counts after subsetting to G4X gene list")
sc.pp.filter_cells(scrnaseq_adata, min_counts=SCRNASEQ_MIN_COUNTS)
print(scrnaseq_adata)


# Normalize, plot, log transform, and scale the scRNAseq anndata independently before concatenation with the G4X anndata
scrnaseq_adata.layers['counts'] = scrnaseq_adata.X.copy()  # raw counts layer
sc.pp.normalize_total(scrnaseq_adata, target_sum=1e4)  # normalize counts per cell to a total of 10,000
sc.pp.log1p(scrnaseq_adata)  # logarithmize X within each dataset
scrnaseq_adata.layers['lognorm'] = scrnaseq_adata.X.copy()  # log normalized counts layer
sc.pp.scale(scrnaseq_adata, max_value=10)  # scale X to unit variance and zero mean within each dataset
plot_scaled_counts_per_cell(
    scrnaseq_adata,
    dataset_name='scRNAseq',
    output_dir=OUTPUT_MASTER_DIR,
    color='tab:green'
)

# Note: make sure the cell_type categories are named the same thing in reference and query
# e.g. not T_cells and T_cell
print(scrnaseq_adata.obs['cell_type'].cat.categories)
# Intersection
print(set(g4x_adata.obs['cell_type'].cat.categories) & set(scrnaseq_adata.obs['cell_type'].cat.categories))
# Union
print(set(g4x_adata.obs['cell_type'].cat.categories) | set(scrnaseq_adata.obs['cell_type'].cat.categories))

# in case there are any genes in the G4X that are not in the scRNAseq, though that is unlikely
assert all(g in g4x_adata.var_names for g in scrnaseq_adata.var_names), "Gene mismatch!"
print(g4x_adata)
g4x_adata = g4x_adata[:, scrnaseq_adata.var_names.tolist()].copy()
print(g4x_adata)

# CONCATENATE

print("\nHere are the G4X and scRNAseq anndata objects before concatenation:")
print("G4X anndata:")
print(g4x_adata)
print("scRNAseq anndata:")
print(scrnaseq_adata)

# Concatenate the G4X and scRNAseq anndata objects after independent preprocessing
adata = ad.concat([g4x_adata, scrnaseq_adata], join='inner')

g4x_adata.obs['batch'] = g4x_adata.obs['batch'].astype('category')
g4x_adata.obs['sample'] = g4x_adata.obs['sample'].astype('category')
g4x_adata.obs['section'] = g4x_adata.obs['section'].astype('category')
g4x_adata.obs['assay'] = g4x_adata.obs['assay'].astype('category')

print("Concatenated anndata:")
print(adata)
print("Here are the types of the batch variables:")
for col in adata.obs.columns:
    if col in VARS_USE:
        print(f"{col}: {adata.obs[col].dtype}")

print("Let's check for NaN in each of the batch variables:")
for col in adata.obs.columns:
    if col in VARS_USE:
        print(f"{col}: {adata.obs[col].isnull().sum()} NaN values")

# PCs

# Calculating principal components for harmony on the combined shared feature space
print("\nCalculating PCs for harmony on the combined shared feature space...")
sc.pp.pca(adata)

# Get PCs from the AnnData object
pcs = adata.obsm['X_pca']
print(pcs.shape)  # (n_cells, n_pcs)

# HARMONY

# Run Harmony on the PCA embedding
harmony_out = hm.run_harmony(data_mat=pcs, 
                             meta_data=adata.obs, 
                             vars_use=VARS_USE, 
                             theta=THETA, 
                             max_iter_harmony=MAX_ITER_HARMONY,
                             random_state=SEED)

# Store corrected PCs back in the AnnData object
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr

# Use harmonized PCs for downstream analysis
print("Calculating neighbors and UMAP using harmonized PCs...")
sc.pp.neighbors(adata, use_rep='X_pca_harmony')
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color='cell_type',
    show=False,
    frameon=False,
    save='cell_type.png'
)
sc.pl.umap(
    adata,
    color='assay',
    show=False,
    frameon=False,
    save='assay.png'
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

# Save the harmonized anndata object
adata.write_h5ad(OUTPUT_MASTER_DIR / 'harmony_all.h5ad')






