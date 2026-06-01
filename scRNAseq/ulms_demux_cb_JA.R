# ULMS revision: demultiplexing the first ULMS batch, which used antibody hashing
# https://satijalab.org/seurat/articles/hashing_vignette
# Based on code in tutorial and code developed by Delitto and MJ
.libPaths("/home/jpagolia/James_R_433")
library(ggplot2)
.libPaths("/scg/apps/software/r/4.3.3/lib")
library(Seurat)
library(dplyr)
options(Seurat.object.assay.version = 'v5')

# make sure to ptrepack the cellbender output file to Seurat format first
# https://cellbender.readthedocs.io/en/latest/tutorial/
ulms_all <- Read10X_h5("ref/Batch01_cb_feature_bc_matrix_filtered_seurat.h5")
ulms_all

#create Seurat object
ulms <- CreateSeuratObject(counts = ulms_all$`Gene Expression`)
ulms # there should be a gene expression matrix and an antibody capture matrix

# Normalize RNA data with log normalization
ulms <- NormalizeData(ulms)
# Find and scale variable features
ulms <- FindVariableFeatures(ulms, selection.method = "mean.var.plot")
ulms <- ScaleData(ulms, features = VariableFeatures(ulms))

# Add HTO data as a new assay independent from RNA
ulms[["HTO"]] <- CreateAssayObject(counts = ulms_all$`Antibody Capture`)
# Normalize HTO data, here we use centered log-ratio (CLR) transformation
ulms <- NormalizeData(ulms, assay = "HTO", normalization.method = "CLR")
DefaultAssay(ulms) <- "RNA"

# Demultiplex the cells based on HTO enrichment
ulms <- HTODemux(ulms, assay = "HTO", positive.quantile = 0.99)

#Visualization

# Global classification results
table(ulms$HTO_classification.global)

# Group cells based on the max HTO signal
Idents(ulms) <- "HTO_maxID"

# Visualize enrichment for selected HTOs with ridge plots
ridgeplot <- RidgePlot(ulms, assay = "HTO", features = rownames(ulms[["HTO"]]), y.max=0.1)
ggsave("ref/ridgeplot.png", width=20, height=20)

# Visualize pairs of HTO signals to confirm mutual exclusivity in singlets
FeatureScatter(ulms, feature1 = "hto_anti-humanHashTag7--LNH-94-2M2-TSB", feature2 = "hto_anti-humanHashTag9--LNH-94-2M2-TSB")
ggsave("ref/featurescatter1.jpg")
FeatureScatter(ulms, feature1 = "hto_anti-humanHashTag9--LNH-94-2M2-TSB", feature2 = "hto_anti-humanHashTag10--LNH-94-2M2-TSB")
ggsave("ref/featurescatter2.jpg")
FeatureScatter(ulms, feature1 = "hto_anti-humanHashTag7--LNH-94-2M2-TSB", feature2 = "hto_anti-humanHashTag10--LNH-94-2M2-TSB")
ggsave("ref/featurescatter3.jpg")

# Compare number of UMIs for singlets, doublets and negative cells
Idents(ulms) <- "HTO_classification.global"
VlnPlot(ulms, features = "nCount_RNA", pt.size = 0.1, log = TRUE)
ggsave("ref/vlnplot.jpg")

HTOHeatmap(ulms, assay = "HTO")
ggsave("ref/heatmap.png")

table(ulms$HTO_maxID)
ulms$sample <- NA
ulms$sample[ulms$HTO_maxID == "anti-humanHashTag7--LNH-94-2M2-TSB"] <- as.character('Sample02')
ulms$sample[ulms$HTO_maxID == "anti-humanHashTag9--LNH-94-2M2-TSB"] <- as.character('Sample03')
ulms$sample[ulms$HTO_maxID == "anti-humanHashTag10--LNH-94-2M2-TSB"] <- as.character('Sample04')
metadata <- ulms@meta.data
metadata$sample <- as.character(metadata$sample)
write.csv(metadata, 'ref/metadata_hto.csv', row.names = TRUE, quote = TRUE)

sessionInfo()