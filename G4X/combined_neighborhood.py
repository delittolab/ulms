# Combining the neighborhood results from all sections into one figure
# This averages the Z-score results from the sections that come from the same patient
# Then uses Stouffer's method to combine the Z scores across patients

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

def combine_spatial_sections_nested(scores, sections, all_cells, patient_col='Patient'):
    """
    Combine neighborhood enrichment accounting for patient-level nesting to avoid violating the independence assumption
    Parameters:
    -----------
    scores : dict
        Dictionary of section -> DataFrame of z-scores
    sections : list
        List of section names
    all_cells : AnnData
        Your full AnnData object with patient information
    patient_col : str
        Column name in all_cells.obs with patient IDs (default: 'Patient')
    """
    
    # Create section to patient mapping from your data
    section_to_patient = {}
    for section in sections:
        # Get patient ID for this section
        section_cells = all_cells[all_cells.obs['Section'] == section]
        patient = section_cells.obs[patient_col].iloc[0]  # Should be same for all cells in section
        section_to_patient[section] = patient
    
    print(f"Found {len(set(section_to_patient.values()))} unique patients")
    print(f"Sections per patient: {pd.Series(section_to_patient).value_counts().to_dict()}")
    
    # Get all cell types
    all_cell_types = scores[sections[0]].index.tolist()
    n_types = len(all_cell_types)
    
    # First stage: Aggregate within patients
    patients = sorted(set(section_to_patient.values()))
    patient_scores = {}
    
    for patient in patients:
        # Get sections for this patient
        patient_sections = [s for s in sections if section_to_patient[s] == patient]
        print(f"  Patient {patient}: {len(patient_sections)} sections - {patient_sections}")
        
        # Combine sections within patient using simple average
        patient_combined = np.zeros((n_types, n_types))
        patient_n = np.zeros((n_types, n_types))
        
        for i in range(n_types):
            for j in range(n_types):
                vals = []
                for section in patient_sections:
                    val = scores[section].iloc[i, j]
                    if not np.isnan(val):
                        vals.append(val)
                
                if len(vals) > 0:
                    patient_combined[i, j] = np.mean(vals)  # Average within patient
                    patient_n[i, j] = len(vals)
                else:
                    patient_combined[i, j] = np.nan
        
        patient_scores[patient] = pd.DataFrame(patient_combined, 
                                               index=all_cell_types, 
                                               columns=all_cell_types)
    
    # Second stage: Combine across patients using Stouffer's method
    combined = np.zeros((n_types, n_types))
    n_patients_contributing = np.zeros((n_types, n_types))
    variability = np.zeros((n_types, n_types))
    
    for i in range(n_types):
        for j in range(n_types):
            patient_vals = []
            
            for patient in patients:
                val = patient_scores[patient].iloc[i, j]
                if not np.isnan(val):
                    patient_vals.append(val)
            
            if len(patient_vals) > 0:
                n_patients_contributing[i, j] = len(patient_vals)
                variability[i, j] = np.std(patient_vals)
                
                # Stouffer's method across PATIENTS (now independent)
                combined[i, j] = np.sum(patient_vals) / np.sqrt(len(patient_vals))
            else:
                combined[i, j] = np.nan
    
    result = {
        'zscore': pd.DataFrame(combined, index=all_cell_types, columns=all_cell_types),
        'n_patients': pd.DataFrame(n_patients_contributing, index=all_cell_types, columns=all_cell_types),
        'variability': pd.DataFrame(variability, index=all_cell_types, columns=all_cell_types),
        'patient_scores': patient_scores  # Individual patient results
    }
    
    return result


# Main processing loop calculating the neighbors for each section
scores = {}

for idx, section in enumerate(sections):
    print(f"Processing {section} ({idx+1}/{len(sections)})")
    
    # Process section
    adata = all_cells[all_cells.obs['Section'] == section].copy()
    adata.obsm['spatial'] = adata.obsm.pop('X_spatial')
    
    # Filter low count types
    type_counts = adata.obs[ct_key].value_counts()
    total_cells = len(adata)
    min_cells = total_cells * PCT_THRESHOLD
    top_types = type_counts[type_counts >= min_cells].index.tolist()
    adata = adata[adata.obs[ct_key].isin(top_types)]
    
    # Compute enrichment
    sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial')
    sq.gr.nhood_enrichment(adata, cluster_key=ct_key, connectivity_key='spatial', seed=SEED)
    
    # Get categorical order and z-scores
    categories = adata.obs[ct_key].cat.categories.tolist()
    zscore_matrix = adata.uns[f'{ct_key}_nhood_enrichment']['zscore']
    zscore_df = pd.DataFrame(zscore_matrix, index=categories, columns=categories)
    
    # Align to all cell types (add NaN for missing)
    all_cell_types = all_cells.obs[ct_key].cat.categories.tolist()
    for ct in all_cell_types:
        if ct not in categories:
            zscore_df.loc[ct] = np.nan
            zscore_df[ct] = np.nan
    zscore_df = zscore_df.reindex(index=all_cell_types, columns=all_cell_types)
    
    # Store
    scores[section] = zscore_df

# After loop: Combine accounting for patient structure
print("\n" + "="*50)
print("Combining sections accounting for patient nesting")
print("="*50)

result = combine_spatial_sections_nested(scores, sections, all_cells, patient_col='Patient')



# PLOTTING

# After combining
print("\n" + "="*50)
print("Preparing results for plotting")
print("="*50)

# Diagnostics
n_nan = result['zscore'].isna().sum().sum()
n_total = result['zscore'].size
pct_nan = 100 * n_nan / n_total

print(f"NaN values: {n_nan}/{n_total} ({pct_nan:.1f}%)")

# Check patient coverage
n_patients_total = len(set(all_cells.obs['Patient']))
n_patients_diagonal = np.diag(result['n_patients'].values)

print("\nCell type coverage:")
for ct, n in zip(all_cell_types, n_patients_diagonal):
    print(f"  {ct:20s}: {n:.0f}/{n_patients_total} patients")

# Strategy: Replace NaN with 0, but warn about high NaN percentage
if pct_nan > 50:
    print(f"\n⚠️  Warning: {pct_nan:.1f}% of values are NaN!")
    print("This suggests many cell type pairs don't co-occur across patients")
    print("Consider filtering or interpreting results carefully")

# Replace NaN for plotting
zscore_for_plotting = np.nan_to_num(result['zscore'].values, nan=0.0)

# Store for squidpy
all_cells.uns[f'{ct_key}_nhood_enrichment'] = {
    'zscore': zscore_for_plotting
}

# Save detailed results
all_cells.uns[f'{ct_key}_nhood_enrichment_detailed'] = result

# Export CSVs with NaN info
result['zscore'].to_csv('combined_zscore_with_nan.csv')  # Keep NaN
pd.DataFrame(zscore_for_plotting, 
             index=all_cell_types, 
             columns=all_cell_types).to_csv('combined_zscore_for_plotting.csv')
result['n_patients'].to_csv('n_patients_contributing.csv')

print("\n✓ Saved CSV files")

# Save h5ad
all_cells.write_h5ad('all_cells_with_neighborhood.h5ad')
print("✓ Saved h5ad")

# Plot
print("\nGenerating plots...")
try:
    sq.pl.nhood_enrichment(all_cells, cluster_key=ct_key, method='average',
                          cmap='magma', vmin=-5, vmax=5,
                          save='all_sections_nhood_nested.png', dpi=300)
    plt.close()
    print("✓ Saved PNG")
    
    sq.pl.nhood_enrichment(all_cells, cluster_key=ct_key, method='average',
                          cmap='magma', vmin=-5, vmax=5,
                          save='all_sections_nhood_nested.pdf', dpi=300)
    plt.close()
    print("✓ Saved PDF")
    
except Exception as e:
    print(f"⚠️  Squidpy plotting failed: {e}")
    print("Creating manual plot...")
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot with NaN masked
    masked_data = np.ma.masked_invalid(result['zscore'].values)
    
    im = ax.imshow(masked_data, cmap='magma', vmin=-5, vmax=5, aspect='auto')
    ax.set_xticks(range(len(all_cell_types)))
    ax.set_yticks(range(len(all_cell_types)))
    ax.set_xticklabels(all_cell_types, rotation=90)
    ax.set_yticklabels(all_cell_types)
    
    plt.colorbar(im, ax=ax, label='Z-score')
    ax.set_title('Combined Neighborhood Enrichment\n(Stouffer\'s Method, Patient-Nested)')
    plt.tight_layout()
    plt.savefig('all_sections_nhood_nested_manual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved manual plot")

# Print diagnostics
print("\n" + "="*50)
print("DIAGNOSTICS")
print("="*50)

print("\nNumber of patients contributing per cell type pair:")
print(result['n_patients'].describe())

print("\nVariability across patients:")
print(result['variability'].describe())

# Additional useful diagnostics
print("\nCell type pairs with highest variability:")
variability_long = result['variability'].stack().sort_values(ascending=False)
print(variability_long.head(10))

print("\nCell type pairs with least patient support:")
n_patients_long = result['n_patients'].stack().sort_values(ascending=True)
print(n_patients_long[n_patients_long > 0].head(10))

print("\n✓ Analysis complete!")
print(f"Output directory: {OUTPUT_MASTER_DIR}")