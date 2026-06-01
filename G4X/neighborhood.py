import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import seaborn as sns

sc.settings.n_jobs = -1  # Use all available cores
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

CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

OUTPUT_MASTER_DIR = jpasq.create_output_dir(PARENT_DIR, 'neighborhood', change_dir=True)
DATA_DIR = PARENT_DIR.parent.parent / 'G4X' / 'G4X_raw'
print(DATA_DIR)
ANNDATA_DIR = PARENT_DIR / 'objects'
print(ANNDATA_DIR)

# SET UP
ct_key = 'ct'
SEED = 1234
PCT_THRESHOLD = 0.01   # Cells must make up 1% of the section to be included in neighborhood analysis

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

# Color palette for celltypes
# Make sure the celltype colors line up
ct_categories = all_cells.obs[ct_key].cat.categories.to_list()
ct_colors = all_cells.uns[f'{ct_key}_colors']
ct_palette = dict(zip(ct_categories, ct_colors))
print(ct_palette)

n_sections_per_row = 3  # How many sections per row (each section has 2 plots)
n_cols = n_sections_per_row * 2  # Total columns (2 plots × sections per row)
n_rows = int(np.ceil(len(sections) / n_sections_per_row))

fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
gs = mpl.gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.25)

for idx, section in enumerate(sections):
    print(f"Processing {section} ({idx+1}/{len(sections)})")
    
    # Process section
    adata = all_cells[all_cells.obs['Section'] == section].copy()
    adata.obsm['spatial'] = adata.obsm.pop('X_spatial')
    # Remove low count cell types
    type_counts = adata.obs[ct_key].value_counts()
    total_cells = len(adata)
    min_cells = total_cells * PCT_THRESHOLD
    top_types = type_counts[type_counts >= min_cells].index.tolist()
    adata = adata[adata.obs[ct_key].isin(top_types)]
    print(f"  Kept {len(top_types)} types, {len(adata)}/{total_cells} cells")
    
    sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial')
    sq.gr.nhood_enrichment(adata, cluster_key=ct_key, connectivity_key='spatial', seed=SEED)
    
    # ===== SAVE INDIVIDUAL PLOT (with both spatial and heatmap) =====
    fig_ind, (ax_spatial, ax_heatmap) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Spatial scatter plot
    sq.pl.spatial_scatter(adata, 
                          shape=None, 
                          size=0.5, 
                          library_id='spatial', 
                          color=ct_key, 
                          ax=ax_spatial, 
                          legend_loc=None,
                          title='',
                          frameon=False)
    
    sq.pl.nhood_enrichment(adata, 
                          cluster_key=ct_key,
                          method='average',
                          cmap='magma',
                          vmin=-5,
                          vmax=5,
                          ax=ax_heatmap,
                          title=f'{section}\n({len(adata)} cells)'
                          )
    
    for axis in fig_ind.get_axes():
        if axis != ax_heatmap and axis != ax_spatial:  # Skip main axes
            axis.set_ylabel('')  # Remove label from colorbar axes
    
    fig_ind.tight_layout()
    fig_ind.savefig(f'{section}.png', dpi=300, bbox_inches='tight')
    fig_ind.savefig(f'{section}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig_ind)
    
    # ===== ADD TO COMBINED PLOT =====
    # Calculate row and column positions
    row = idx // n_sections_per_row
    section_col = idx % n_sections_per_row
    
    # Spatial plot goes in first column of the pair
    col_spatial = section_col * 2
    ax_spatial = fig.add_subplot(gs[row, col_spatial])
    
    sq.pl.spatial_scatter(adata, 
                          shape=None, 
                          size=0.5, 
                          library_id='spatial', 
                          color=ct_key, 
                          ax=ax_spatial, 
                          legend_loc=None, 
                          title='',
                          frameon=False)
    ax_spatial.set_title(f"{section}", fontsize=9, fontweight='bold')
    
    # Remove legend
    legend = ax_spatial.get_legend()
    if legend:
        legend.remove()
    # Turn off axis labels
    ax_spatial.tick_params(axis='x', labelbottom=False)
    ax_spatial.tick_params(axis='y', labelleft=False)
    ax_spatial.set_xlabel('')
    ax_spatial.set_ylabel('')
    
    # Heatmap goes in second column of the pair
    col_heatmap = section_col * 2 + 1
    ax_heatmap = fig.add_subplot(gs[row, col_heatmap])
    sq.pl.nhood_enrichment(adata, 
                        cluster_key=ct_key,
                        method='average',
                        cmap='magma',
                        vmin=-5,
                        vmax=5,
                        ax=ax_heatmap,
                        title=f'{section}\n({len(adata)} cells)'
                        )
    # Remove ylabel from all child axes (colorbars)
    for child_ax in fig_ind.get_axes():
        child_ax.set_ylabel('')
    for child_ax in fig.get_axes():
        child_ax.set_ylabel('')

plt.savefig('all_sections_nhood.png', dpi=300, bbox_inches='tight')
plt.savefig('all_sections_nhood.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\n✓ Saved {len(sections)} individual plots to {OUTPUT_MASTER_DIR}")
print(f"✓ Saved combined plot to {OUTPUT_MASTER_DIR}")
