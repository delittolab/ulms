#!/usr/bin/env python
# coding: utf-8

# # For the revision: decoupler analysis of the ULMS tumor subset

# # Set up

# In[ ]:


import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
from pathlib import Path
import matplotlib as mpl
import decoupler as dc
import seaborn as sns
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats

mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
sc.set_figure_params(dpi_save=300)


# In[ ]:


def create_output_dir(master_dir, sub_dir_name, change_dir=False):
    '''
    Create an output directory as a subdirectory 'sub_dir_name' string within a parent directory master_dir
    Will not overwrite the files within that directory
    2025-05-28 moved to functions file
    2025-08-29 added change_dir
    '''
    output_dir = master_dir / sub_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Created output directory {output_dir}')
    if change_dir:
        os.chdir(output_dir)
        sc.settings.figdir = output_dir
        print(f'Default output directory changed to {output_dir}')
    return output_dir


# In[ ]:


# Set up input and output directories
CURRENT_DIR = Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent
print(PROJECT_DIR)

DATA_DIR = PROJECT_DIR / 'objects'
print(DATA_DIR)

OUTPUT_DIR = create_output_dir(PROJECT_DIR, 'decoupler', change_dir=True)


# # Load the data

# In[ ]:


ad_path = os.path.join(DATA_DIR, 'tumor_subset_raw.h5ad')
adata = sc.read_h5ad(ad_path)
adata


# In[ ]:


# Save count data, then lognormalize and save in adata.raw
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()
# scale the data to ensure no single feature dominates the downstream results
sc.pp.scale(adata, max_value=10)
adata.raw = adata
adata


# In[ ]:


adata.obs['leiden0_2'].cat.categories


# In[ ]:


annotation_map = {
    "0" : "ESR1+ cells",
    "1" : "IFN signaling cells",
    "2" : "Mesenchyme-like cells",
    "3" : "Cycling cells",
    "4" : "AR+/PGR+ cells",
    "5" : "Ischemic cells",
    "6" : "SMC-like cells",
    "7" : "Neuron-like cells",
    "8" : "ATRX+/DAXX- cells"
}
adata.obs['annotation'] = adata.obs['leiden0_2'].map(annotation_map)
# Set the order of the categories
celltype_order = [
    "AR+/PGR+ cells",
    "ESR1+ cells",
    "SMC-like cells",
    "IFN signaling cells",
    "Cycling cells",
    "Ischemic cells",
    "Mesenchyme-like cells",
    "Neuron-like cells",
    "ATRX+/DAXX- cells"
]
# Make sure the color map is consistent
celltype_colors = {
    "AR+/PGR+ cells" : '#55a868',
    "ESR1+ cells" : '#4c72b0',
    "SMC-like cells" : '#da8bc3',
    "IFN signaling cells" : '#c44e52',
    "Cycling cells" : '#937860',
    "Ischemic cells" : '#8172b3',
    "Mesenchyme-like cells" : '#dd8452',
    "Neuron-like cells" : '#8c8c8c',
    "ATRX+/DAXX- cells" : '#ccb974',
}
adata.obs["annotation"] = pd.Categorical(adata.obs["annotation"], categories=celltype_order, ordered=True)
adata.uns['annotation_colors'] = [celltype_colors[celltype] for celltype in celltype_order]


# In[ ]:


# Also load the annotated anndata and transfer the umap embedding
ad_path = os.path.join(DATA_DIR, 'tumor_annotated.h5ad')
ann = sc.read_h5ad(ad_path)
ann


# In[ ]:


# check if indices are the same
print(adata.obs.index.equals(ann.obs.index))
print(adata.obs.index.identical(ann.obs.index))


# In[ ]:


# transfer the labels
adata.uns['N_scVI'] = ann.uns['N_scVI']
adata.obsm['X_scVI'] = ann.obsm['X_scVI']
adata.obsm['X_umap'] = ann.obsm['X_umap']
adata.obsp['N_scVI_connectivities'] = ann.obsp['N_scVI_connectivities']
adata.obsp['N_scVI_distances'] = ann.obsp['N_scVI_distances']


# In[ ]:


del ann


# # CollecTRI

# In[ ]:


collectri = dc.op.collectri(organism="human")
collectri


# In[ ]:


dc.mt.ulm(data=adata, net=collectri)


# In[ ]:


score = dc.pp.get_obsm(adata=adata, key="score_ulm")
score


# In[ ]:


tf = "MYC"
sc.pl.umap(score, color=[tf, "annotation"], cmap="RdBu_r", vcenter=0, title=[f"{tf} score", "annotation"])
sc.pl.violin(score, keys=[tf], groupby="annotation", rotation=90, ylabel=f"{tf} score")


# In[ ]:


sc.pl.umap(adata, color=[tf, "annotation"], title=[f"{tf} expression", "annotation"])
sc.pl.violin(adata, keys=[tf], groupby="annotation", rotation=90, ylabel=f"{tf} expression")


# In[ ]:


df = dc.tl.rankby_group(adata=score, groupby="annotation", reference="rest", method="wilcoxon")
df = df[df["stat"] > 0]
df


# In[ ]:


df.to_csv('collectri.csv', index=False)
score.write_h5ad('collectri_score.h5ad')


# In[ ]:


n_markers = 10
source_markers = (df.groupby("group").apply(lambda g: g.drop_duplicates(subset="name").head(n_markers)["name"].tolist()).to_dict())
# reorder
source_markers = dict(sorted(source_markers.items(), key=lambda item: celltype_order.index(item[0])))
source_markers


# In[ ]:


sc.pl.matrixplot(
    adata=score,
    var_names=source_markers,
    groupby="annotation",
    dendrogram=False,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap="Blues",
    save=f'ulms_tfs_{n_markers}.png',
)
sc.pl.matrixplot(
    adata=score,
    var_names=source_markers,
    groupby="annotation",
    dendrogram=False,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap="Blues",
    save=f'ulms_tfs_{n_markers}.pdf',
)


# In[ ]:


n_markers = 5
source_markers = (df.groupby("group").apply(lambda g: g.drop_duplicates(subset="name").head(n_markers)["name"].tolist()).to_dict())
# reorder
source_markers = dict(sorted(source_markers.items(), key=lambda item: celltype_order.index(item[0])))
source_markers


# In[ ]:


sc.pl.matrixplot(
    adata=score,
    var_names=source_markers,
    groupby="annotation",
    dendrogram=False,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap="Blues",
    save=f'ulms_tfs_{n_markers}.png',
)
sc.pl.matrixplot(
    adata=score,
    var_names=source_markers,
    groupby="annotation",
    dendrogram=False,
    standard_scale="var",
    colorbar_title="Z-scaled scores",
    cmap="Blues",
    save=f'ulms_tfs_{n_markers}.pdf',
)


# In[ ]:


tf = 'MYC'
sc.pl.violin(score, keys=tf, groupby="annotation", rotation=90, save=f'violin_{tf}.png')
sc.pl.violin(score, keys=tf, groupby="annotation", rotation=90, save=f'violin_{tf}.pdf')


# In[ ]:


# Flower plots of the top 5 TFs in each cluster
for annotation in celltype_order:

    sources = source_markers[annotation]
    adata_subset = adata[adata.obs["annotation"] == annotation].to_df().mean(0).to_frame().T
    score_subset = score[score.obs["annotation"] == annotation].to_df().mean(0).to_frame().T
    
    plt_name = f'fp_{annotation.replace("/", "_").replace(" ", "")}_{"_".join(sources)}.png'
    dc.pl.network(
        data=adata_subset,
        score=score_subset,
        net=collectri,
        sources=sources,
        targets=10,
        size_node=10,
        figsize=(5, 5),
        s_cmap="Reds",
        t_cmap="Reds",
        save=plt_name,
    )
    plt_name = f'fp_{annotation.replace("/", "_").replace(" ", "")}_{"_".join(sources)}.pdf'
    dc.pl.network(
        data=adata_subset,
        score=score_subset,
        net=collectri,
        sources=sources,
        targets=10,
        size_node=10,
        figsize=(5, 5),
        s_cmap="Reds",
        t_cmap="Reds",
        save=plt_name,
    )


# # Reactome

# In[ ]:


msigdb = dc.op.resource(name='MSigDB', organism='human')
np.unique(msigdb['collection'])


# In[ ]:


network_name = 'reactome_pathways'
prefix = 'REACTOME_'
suffix = '_PATHWAY'

net = msigdb[msigdb['collection'] == network_name]
net = net.rename(columns={"geneset": "source", "genesymbol": "target"})
net = net.drop_duplicates(subset=['source', 'target'])
net = net[['target', 'source']]
net['source'] = net['source'].str.replace(prefix, '') # remove prefix
net['source'] = net['source'].str.replace(suffix, '') # remove suffix
net


# In[ ]:


dc.mt.ulm(data=adata, net=net)


# In[ ]:


score = dc.pp.get_obsm(adata=adata, key="score_ulm")
score


# In[ ]:


df = dc.tl.rankby_group(
    adata=score,
    groupby='annotation',
    reference='rest',
    method='wilcoxon',
)
df = df[df['stat'] > 0]
df


# In[ ]:


df.to_csv(f'{network_name}_scores.csv', index=False)
score.write_h5ad(f'{network_name}_score.h5ad')


# In[ ]:


n_markers = 5
source_markers = (df.groupby("group").apply(lambda g: g.drop_duplicates(subset="name").head(n_markers)["name"].tolist()).to_dict())
source_markers = {key: source_markers[key] for key in celltype_order} # reorder keys to match the preferred cell type order
source_markers


# In[ ]:


mp = sc.pl.matrixplot(
    adata=score,
    var_names=source_markers,
    groupby='annotation',
    dendrogram=False,
    standard_scale='var',
    colorbar_title='Z-scaled scores',
    cmap='Blues', 
    swap_axes=True,
    show=False,
    return_fig=True,
)
mp.savefig(f'top10_{network_name}_mp.png', dpi=300)
mp.savefig(f'top10_{network_name}_mp.pdf', dpi=300)


# In[ ]:


# Looking at the IFN signaling cells
sc.pl.dotplot(adata, groupby='annotation', var_names=['STAT1', 'STAT3', 'CXCL10', 'ISG15', 'IFIT1', 'MX1', 'IRF1', 'SOCS3', 'BCL2', 'MMP7', 'MYC'])

