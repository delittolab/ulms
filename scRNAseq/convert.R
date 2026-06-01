library(dplyr)
library(ggplot2)
library(Seurat, lib.loc = "/scg/apps/software/r/4.3.3/lib") # make sure it's Seurat v5
library(Matrix)

setwd("/oak/stanford/groups/longaker/ULMS/revision/scRNAseq/objects/for_seurat/")

# Create Seurat object from anndata ############################################
# Courtesy of Delitto
# This should have all the genes, not just highly variable genes ###############

# Count Matrix
counts <- ReadMtx(
  mtx = "raw_counts.mtx",
  features = "genes.tsv",
  cells = "barcodes.tsv",
  feature.column = 1
)
# Load metadata
metadata <- read.csv("metadata.csv", row.names = 1)
# Create Seurat object
obj <- CreateSeuratObject(counts = counts, meta.data = metadata)
# Load latent space. A few custom modifications here based on formatting expectations from Seurat.
latent_space <- read.csv("scvi_latent_space.csv", row.names = 1)
colnames(latent_space) <- paste0("Dim", seq_len(ncol(latent_space)))
latent_space <- as.matrix(latent_space)
obj[["scvi"]] <- CreateDimReducObject(
  embeddings = latent_space,
  key = "SCVI_",
  assay = DefaultAssay(obj)
)
# Load neighbors graph
neighbors_graph <- readMM("scvi_connectivities.mtx")
obj@graphs$scvi_nn <- neighbors_graph
# Load umap coordinates
umap_coords <- read.csv("umap_coordinates.csv", row.names = 1)
obj[["umap"]] <- CreateDimReducObject(
  embeddings = as.matrix(umap_coords),
  key = "UMAP_",
  assay = DefaultAssay(obj)
)
# Optional: normalize, scale and run PCA.
# These should be very similar processes as in scanpy, although not identical.
# That said, we've transferred the latent space, neighbors graph and the umap, 
# so your final image outputs will be the same. 
# Keep in mind that all of these features can be transferred from the scanpy 
# object if needed. In general, the log normalized values will be equivalent 
# across platforms. FindVariableFeatures can be made equivalent if 
# selection.method = "vst" and nfeatures = 2000 to sc.pp.highly_variable_genes 
# if flavor = 'seurat_v3' and n_top genes = 2000. Features will be stored in 
# the VariableFeatures() slot for Seurat and adata.var['highly_variable'] 
# in scanpy. All that said, scanpy workflows typically skip the scaledata step 
# so don't expect the final product to be identical. Also, all transformed 
# data must be manually named in scanpy (our default is 
# adata.layers['lognorm']), whereas seurat automatically creates a data or 
# scale.data layer. If you want to do the default seurat workflow here, 
# just include the line below:
obj <- NormalizeData(obj) %>% 
  FindVariableFeatures(selection.method = "vst", nfeatures=2000) %>% 
  ScaleData() %>% 
  RunPCA()

# And voila! You have your scvi integrated seurat object. 
# You can try it yourself with the scanvi_model.h5ad file. 
# This can also be very useful for doublet detection, as scrublet is a heck 
# of a lot more efficient and customizable than doubletfinder. 
# The field is looking much closer now at doublet detection than in years past 
# now that reviewers can access the data. Conversely, other integration 
# methods (i.e. harmony, CCA, RPCA, fastMNN), SCTransform and spatial 
# deconvolution with RCTD, currently the gold standard, is much more suited 
# to R. I will say after using both a lot this year, python is exponentially 
# faster with larger datasets. As a quick demo: here are the final umaps in 
# seurat and scanpy.

# plot umaps
DimPlot(obj, group.by = 'celltype') %>% ggsave("celltype_umap.png", . ,)
DimPlot(obj, group.by = 'leiden0_2') %>% ggsave("umap_leiden0_2.png", . ,)

# clean up environment
rm("counts", "latent_space", "metadata", "neighbors_graph", "umap_coords")

### save Seurat object #########################################################

SaveSeuratRds(obj, file="tumor_subset.Rds")

