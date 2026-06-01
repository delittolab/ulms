# Matching up ULMS tumor subtypes with ULMS scVIVA clusters using the scANVI prediction model with a majority voting approach

# SET UP DEPENDENCIES

import sys
import numpy as np
import scanpy as sc
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi

print(f"\nRunning script: {Path(__file__).name}\n")

# version control
print("\nPackage versions:")
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)

mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.rcParams['axes.facecolor'] = 'white'
plt.ioff()
sc.settings.autoshow = False
sc.settings.n_jobs = -1  # Use all available cores

# SET UP DIRECTORIES

CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

G4X_DIR = PARENT_DIR.parent / 'G4X'
print(G4X_DIR)

SCANVI_DIR = PARENT_DIR.parent / 'multimodal/scANVI_tumor_allgenes'

SCANVI_FULL_DIR = jpascvi.create_output_dir(SCANVI_DIR, 'scanvi_full') # where scANVI full adata with predictions is saved
OUTPUT_MASTER_DIR = jpascvi.create_output_dir(SCANVI_DIR, 'scanvi_tumor_corr', change_figdir=True) # new subdirectory for outputs

# USER PARAMETERS
cell_type_key = 'tumor_subtype' # key in the G4X anndata obs where the tumor subtype annotations will be stored
leiden_key = 'leiden1_5' # leiden clustering key in the G4X data to use for majority voting
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTIONS_KEY = "scanvi_pred"

# FUNCTIONS

def majority_vote_with_confidence(adata, 
                                  cluster_col='leiden', 
                                  prediction_col='predicted_cell_type', 
                                  result_col='cell_type_majority', 
                                  confidence_col='majority_confidence'
                                  ):
    """
    Majority voting with confidence (proportion of cells agreeing with majority).
    """
    cluster_stats = []
    
    for cluster, group in adata.obs.groupby(cluster_col):
        value_counts = group[prediction_col].value_counts()
        majority_type = value_counts.idxmax()
        confidence = value_counts.iloc[0] / len(group)  # proportion of majority class
        cluster_stats.append({
            'cluster': cluster,
            'majority_type': majority_type,
            'confidence': confidence,
            'n_cells': len(group)
        })
    
    stats_df = pd.DataFrame(cluster_stats).set_index('cluster')
    
    # Map back to cells
    adata.obs[result_col] = adata.obs[cluster_col].map(stats_df['majority_type'])
    adata.obs[confidence_col] = adata.obs[cluster_col].map(stats_df['confidence']).astype(float)
    
    # Print summary
    print("Majority Vote Summary:")
    print(stats_df.to_string())
    
    return adata, stats_df


# LOAD THE POST-SCANVI ADATA WITH CELL TYPE PREDICTIONS
print(f"\nLoading post-scanvi anndata with cell type predictions from {SCANVI_FULL_DIR}")
adata_full = sc.read_h5ad(SCANVI_FULL_DIR / 'scanvi_full_adata.h5ad')
print(adata_full)

# Load the G4X leiden clusters (from the original G4X analysis)
print(f"\nLoading clustered G4X object from {G4X_DIR}")
data_dir = G4X_DIR / 'scviva_tumor'
adata = sc.read_h5ad(data_dir / 'scviva_tumor_clustered.h5ad')
adata.obs_names = adata.obs['cell_name']
adata.obs[cell_type_key] = 'Unknown'

# Transfer the predicted cell types for the tumor cells
common_idx = adata.obs.index.intersection(adata_full.obs.index)
adata.obs.loc[common_idx, SCANVI_PREDICTIONS_KEY] = adata_full.obs.loc[common_idx, SCANVI_PREDICTIONS_KEY]
print(adata.obs[SCANVI_PREDICTIONS_KEY].isna().sum())  # how many NaN (cells that were not in the common index)?
adata = adata[adata.obs[SCANVI_PREDICTIONS_KEY].notna()].copy()
adata.obs[SCANVI_PREDICTIONS_KEY] = adata.obs[SCANVI_PREDICTIONS_KEY].astype('category')
print(adata.obs[SCANVI_PREDICTIONS_KEY].cat.categories)

# Print final G4X anndata
print("Final G4X anndata:")
print(adata)


adata, stats = majority_vote_with_confidence(adata, 
                                             prediction_col=SCANVI_PREDICTIONS_KEY, 
                                             cluster_col=leiden_key, 
                                             result_col='cell_type_majority', 
                                             confidence_col='majority_confidence')

stats.to_csv(OUTPUT_MASTER_DIR / f'majority_vote_summary_{leiden_key}.csv')

# Visualize
sc.pl.umap(adata, color=SCANVI_PREDICTIONS_KEY, save='_predicted_cell_type.png', frameon=False)
sc.pl.umap(adata, color=SCANVI_PREDICTIONS_KEY, save='_predicted_cell_type.pdf', frameon=False)
sc.pl.umap(adata, color='cell_type_majority', save='_cell_type_majority.png', frameon=False)
sc.pl.umap(adata, color='cell_type_majority', save='_cell_type_majority.pdf', frameon=False)
sc.pl.umap(adata, color='majority_confidence', save='_majority_confidence.png', frameon=False, cmap='inferno')
sc.pl.umap(adata, color='majority_confidence', save='_majority_confidence.pdf', frameon=False, cmap='inferno')

adata.write_h5ad(OUTPUT_MASTER_DIR / 'scviva_tumor_scanvi_pred.h5ad')

# Also load the resolvi adata to take a look at where these predicted subtypes fall on the tumor subset umap

print("\nLoading resolvi anndata")
RESOLVI_DIR = G4X_DIR / 'resolvi_tumor'
resolvi_adata = sc.read_h5ad(RESOLVI_DIR / 'resolvi_tumor.h5ad')
print(resolvi_adata)
resolvi_adata.obs_names = resolvi_adata.obs['cell_name']
# Transfer the predicted cell types for the tumor cells
resolvi_adata.obs[SCANVI_PREDICTIONS_KEY] = adata_full.obs.loc[resolvi_adata.obs.index, SCANVI_PREDICTIONS_KEY].values
np.unique(resolvi_adata.obs[SCANVI_PREDICTIONS_KEY])

# Visualize
sc.pl.umap(resolvi_adata, color=SCANVI_PREDICTIONS_KEY, save='_resolvi_predicted_cell_type.png', frameon=False)
sc.pl.umap(resolvi_adata, color=SCANVI_PREDICTIONS_KEY, save='_resolvi_predicted_cell_type.pdf', frameon=False)

resolvi_adata.write_h5ad(OUTPUT_MASTER_DIR / 'resolvi_tumor_scanvi_pred.h5ad')