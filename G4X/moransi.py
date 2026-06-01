# Moran's I score on individual sections and across all sections

# DEPENDENCIES
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import seaborn as sns

plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.ioff()
sc.settings.autoshow = False

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpasq

# version control
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("squidpy:", sq.__version__)
print("seaborn:", sns.__version__)

# DIRECTORY SET UP

CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

OUTPUT_MASTER_DIR = jpasq.create_output_dir(PARENT_DIR, 'moransi', change_dir=True)
ANNDATA_DIR = PARENT_DIR / 'objects'
print(ANNDATA_DIR)

# USER PARAMETERS
ct_key = 'ct'
SEED = 1234
N_JOBS = int(os.getenv('SLURM_CPUS_PER_TASK', os.cpu_count()))


# LOAD DATA

all_cells = sc.read_h5ad(ANNDATA_DIR / 'g4x_all_raw.h5ad')
print(all_cells)

# order sections properly. This is how they will print on the page.
sections = [
    'B01', 'C01', 'D01', 'E01', 
    'F01', 'G01', 'H01', 'A02', 
    'B02', 'C02', 'E02', 'F02', 
    'G02', 'H02', 'A03', 'B03', 
    'C03', 'D03', 'E03', 'F03', 
    'G03', 'H03', 'A04', 'B04', 
    'C04', 'D04',
]

# MAIN PROCESSING LOOP
full_df = pd.DataFrame()
for idx, section in enumerate(sections):
    print(f"Processing {section} ({idx+1}/{len(sections)})")
    # Process section
    adata = all_cells[all_cells.obs['Section'] == section].copy()
    adata.obsm['spatial'] = adata.obsm.pop('X_spatial')
    # Calculate Moran's I score for all genes
    sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial')
    sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=N_JOBS)
    # Extract and combine the results into one dataframe, with rows as section-specific genes
    df = adata.uns['moranI']
    df['section'] = section
    df['gene'] = df.index
    print(df.head(10))
    df.to_csv(f'{section}.csv')
    full_df = pd.concat([full_df, df], ignore_index=True)

print(full_df.head())
print(full_df.columns)
print(full_df.shape)

# PLOTTING

# Convert section to categorical with your desired order
full_df['section'] = pd.Categorical(full_df['section'], categories=sections, ordered=True)

# Then pivot using the 'gene' column:
heatmap_data = full_df.pivot_table(
    index='gene',
    columns='section',
    values='I'
)
heatmap_data.to_csv('moransi_heatmap_data.csv')

pval_data = full_df.pivot_table(
    index='gene',
    columns='section',
    values='pval_norm_fdr_bh'  # or whatever p-value column squidpy uses
)
pval_data.to_csv('moransi_pval_data.csv')

significance_threshold = 0.05
heatmap_data_masked = heatmap_data.copy()
heatmap_data_masked[pval_data >= significance_threshold] = np.nan

# 1) heatmap

plt.figure(figsize=(14, 20))
ax = sns.heatmap(heatmap_data_masked,
                 cmap='RdBu_r',
                 center=0,
                 vmin=-1,
                 vmax=1,
                 xticklabels=True,
                 yticklabels=False,
                 # This makes NaN values grey
                 cbar_kws={'label': "Moran's I (p < 0.05)"},
                 mask=False)  # Don't use mask parameter, we set NaN instead

# Add grey color for NaN (non-significant) values
ax.set_facecolor('lightgrey')  # Background shows through NaN values

plt.xlabel('Tissue Section')
plt.ylabel('Genes')
plt.title("Spatial Autocorrelation (Moran's I)")
plt.tight_layout()
plt.savefig('morans_i_with_grey_nonsig.png', dpi=300, bbox_inches='tight')
plt.savefig('morans_i_with_grey_nonsig.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 2) clustermap

heatmap_data_for_clustermap = heatmap_data_masked.fillna(0)

g = sns.clustermap(heatmap_data_for_clustermap,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1,
                   vmax=1,
                   figsize=(12, 20),
                   cbar_kws={'label': "Moran's I (p < 0.05)"},
                   yticklabels=False,
                   xticklabels=True,
                   method='ward',           # Clustering method
                   metric='euclidean',      # Distance metric
                   row_cluster=True,        # Cluster genes (rows)
                   col_cluster=False)       # DON'T cluster sections to preserve your order

# Add labels
g.ax_heatmap.set_xlabel('Tissue Section')
g.ax_heatmap.set_ylabel('Genes')

plt.suptitle("Spatial Autocorrelation (Moran's I) - Clustered", y=0.995)
plt.savefig('morans_i_clustermap.png', dpi=300, bbox_inches='tight')
plt.savefig('morans_i_clustermap.pdf', dpi=300, bbox_inches='tight')
plt.close()


# PLOTTING SELECTED GENES

# Print a heatmap of selected genes from key programs
genes = [
    'ESR1', 'PGR', 'AR',
    'CHI3L1', 'NCAM1', 
    'MYH11', 'TAGLN', 'ACTA2', 'MYLK',
    'VEGFA', 'SLC2A1',
    'HLA-DRA', 'CD74', 'WARS1', 'STAT1',
    'MKI67', 'TOP2A', 'RRM2', 
    'COL1A1', 'POSTN', 'SDC1', 'PDGFRB'
    ]
# filter the significant-masked heatmap for those genes only
heatmap_data_masked = heatmap_data_masked.loc[genes]

# 3) Heatmap of selected genes

plt.figure(figsize=(14, 20))
ax = sns.heatmap(heatmap_data_masked,
                 cmap='RdBu_r',
                 center=0,
                 vmin=-1,
                 vmax=1,
                 xticklabels=True,
                 yticklabels=True,
                 # This makes NaN values grey
                 cbar_kws={'label': "Moran's I (p < 0.05)"},
                 mask=False)  # Don't use mask parameter, we set NaN instead

# Add grey color for NaN (non-significant) values
ax.set_facecolor('lightgrey')  # Background shows through NaN values

plt.xlabel('Tissue Section')
plt.ylabel('Genes')
plt.title("Spatial Autocorrelation (Moran's I)")
plt.tight_layout()
plt.savefig('morans_i_selected_genes.png', dpi=300, bbox_inches='tight')
plt.savefig('morans_i_selected_genes.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 4) Cluster map of selected genes
heatmap_data_for_clustermap = heatmap_data_masked.fillna(0)

g = sns.clustermap(heatmap_data_for_clustermap,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1,
                   vmax=1,
                   figsize=(12, 20),
                   cbar_kws={'label': "Moran's I (p < 0.05)"},
                   yticklabels=True,
                   xticklabels=True,
                   method='ward',           # Clustering method
                   metric='euclidean',      # Distance metric
                   row_cluster=True,        # Cluster genes (rows)
                   col_cluster=False)       # DON'T cluster sections to preserve your order

# Set grey background for NaN values
g.ax_heatmap.set_facecolor('lightgrey')

# Add labels
g.ax_heatmap.set_xlabel('Tissue Section')
g.ax_heatmap.set_ylabel('Genes')

plt.suptitle("Spatial Autocorrelation (Moran's I) - Clustered", y=0.995)
plt.savefig('morans_i_clustermap_selected_genes.png', dpi=300, bbox_inches='tight')
plt.savefig('morans_i_clustermap_selected_genes.pdf', dpi=300, bbox_inches='tight')
plt.close()


# After creating the plot, you might want to add summary stats:
print(f"\nSummary Statistics:")
print(f"Total genes: {len(heatmap_data)}")
print(f"Genes with ≥1 significant section: {(~heatmap_data_masked.isna()).any(axis=1).sum()}")
print(f"Mean significant Moran's I: {heatmap_data_masked.mean().mean():.3f}")
print(f"Median sections per gene with significance: {(~heatmap_data_masked.isna()).sum(axis=1).median():.0f}")
