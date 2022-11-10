#!/usr/bin/env python
# coding: utf-8

# The aim for this notebook is the create a central log (meta resource) of all the information and resources I've come across in the discussions, other notebooks, in my own discovery. I tend to find it best for me to keep track of all the many nooks one can store information on Kaggle in one place. 
# 
# The following notebook has tried to aggregate important papers / resources that will help in understanding how this one might go about completing this challenge. I've also provided where these resources have been taken at the bottom. 

# # Insightful notebooks
# 
# ## EDA 
# * MSCI EDA which makes sense - https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense/notebook
# * Complette EDA of MmSCel Integration Data - https://www.kaggle.com/code/leohash/complete-eda-of-mmscel-integration-data
# 
# ## Solutions 
# * MSCI CITEseq Quickstart - https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart
# 

# ## Experiments
# ### CITE-seq
# Key to defining cell type/states indentity through modules of genes uniquely expressed
# 
# ![citeseq.com](https://citeseq.files.wordpress.com/2018/02/figure1.png?w=700)
#  * Protein (CITE-seq): Enables leveraging legacy markers used over the last decades by immunologists to define cell spectrums 
#  
#  
#  ### ATAC-seq 
#  Key to defining immune cell states transitional states best defined by up- and down- regulation of critical transcription factors (usually poorly captured transcriptionally) 
#  
#  ![ATAC-seq transposition reaction ](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41596-022-00692-9/MediaObjects/41596_2022_692_Fig1_HTML.png)
# 
# **Steps:** 
# 1. Nuclei are isolated from cells
#     * maintaining the chromatin structure and any associated DNA-binding proteins, including nucleosomes and TFs, intact
# 2. This chromatin is then exposed to the Tn5 transposase, 
#     * which acts as a homodimer to simultaneously fragment the chromatin and insert sequences containing PCR handles that enable downstream amplification with i5/P5 and i7/P7 ATAC-seq adapters.
#     * Only fragments that receive an i5/P5 adapter at one end and an i7/P7 adapter at the other end will be properly amplified and sequenced.
# 3. Sequence analysis of the library fragments, genomic regions enriched for many Tn5 transposition events are designated as peaks of chromatin accessibility, or ATAC-seq peaks
# 
# * This is the test kit that was usedin these experiments:  Multinome - ATAC + Gene Expression: https://www.10xgenomics.com/products/single-cell-multiome-atac-plus-gene-expression

# # Coding Tips
# 
# ## Reading in partial h5 files
# When you only want to read a large file partially using `pandas` you can do it as follows by using the `start` and `stop` parameters in `pd.read_hdf(..)`
# 
# ```python
# targets = pd.read_hdf('../input/targets.h5', start=0, stop=6000)
# ```

# # Data 
# **Multiome_train_inputs** 
# * Columns - gene id (228942 genes)
# * Rows - Cells (105942 cells) 
# * Values - ATAC signal falling in the range of 0 - 16.9336 
# 
# **Multiome_train_targets** 

# # Dictionary 
# * ATAC: assay for transposase-accessible chromatin 
# * bi-CCA: bi-order canonical correlation analysis 
# * BCC:  Basal cell carcinoma 
# * CITE-seq: Cellular indexing of transcriptomes and epitopes by sequencing
# * CLOUD-HSPCs: ‘continuum of low-primed undifferentiated haematopoietic stem and progenitor cells’ 
# * scRNA-seq: single-cell RNA sequencing  
# * scATAC-seq: single-cell assay for transposase-accessible chromatin using sequencing aka **chromatin accessibility**
# * PBMC:  human peripheral blood mononuclear cells
# * HSC: haematopoietic stem cells [wiki](https://en.wikipedia.org/wiki/Hematopoietic_stem_cell)

# # Data Challenges 
# Single cell analysis has a unique set of challenges. Following are papers and posts that I've found to be helpful in identifying those challenges.
# * [Computational challenges of cell cycle analysis using single cell
# transcriptomics](https://arxiv.org/pdf/2208.05229.pdf)

# # Papers
# * [Integrated analysis of multimodal single-cell data](https://www.sciencedirect.com/science/article/pii/S0092867421005833)
#     * The simultaneous measurement of multiple modalities represents an exciting frontier for single-cell genomics and necessitates computational methods that can define cellular states based on multimodal data. Here, we introduce **“weighted-nearest neighbor” analysis**, an unsupervised framework to learn the relative utility of each data type in each cell, enabling an integrative analysis of multiple modalities. We apply our procedure to a CITE-seq dataset of 211,000 human peripheral blood mononuclear cells (PBMCs) with panels extending to 228 antibodies to construct a multimodal reference atlas of the circulating immune system. **Multimodal analysis substantially improves our ability to resolve cell states, allowing us to identify and validate previously unreported lymphoid subpopulations.** Moreover, we demonstrate how to leverage this reference to rapidly map new datasets and to interpret immune responses to vaccination and coronavirus disease 2019 (COVID-19). Our approach represents a broadly applicable strategy to analyze single-cell multimodal datasets and to look beyond the transcriptome toward a unified and multimodal definition of cellular identity.
# * [New horizons in the stormy sea of multimodal single-cell data integration](https://www.sciencedirect.com/science/article/abs/pii/S1097276521010741)
#     * We review steps and challenges toward this goal. Single-cell transcriptomics is now a mature technology, and methods to measure proteins, lipids, small-molecule metabolites, and other molecular phenotypes at the single-cell level are rapidly developing. Integrating these single-cell readouts so that each cell has measurements of multiple types of data, e.g., transcriptomes, proteomes, and metabolomes, is expected to allow identification of highly specific cellular subpopulations and to provide the basis for inferring causal biological mechanisms.
# * [Computation principles and challenges in single-cell data integration](https://www.nature.com/articles/s41587-021-00895-7)
#     * The development of single-cell multimodal assays provides a powerful tool for investigating multiple dimensions of cellular heterogeneity, enabling new insights into development, tissue homeostasis and disease. **A key challenge in the analysis of single-cell multimodal data is to devise appropriate strategies for tying together data across different modalities.** The term ‘data integration’ has been used to describe this task, encompassing a broad collection of approaches ranging from batch correction of individual omics datasets to association of chromatin accessibility and genetic variation with transcription. Although existing integration strategies exploit similar mathematical ideas, they typically have distinct goals and rely on different principles and assumptions. Consequently, new definitions and concepts are needed to contextualize existing methods and to enable development of new methods.
# * [Diagonal integration of multimodal single-cell data: potential pitfalls and paths forward](https://www.nature.com/articles/s41467-022-31104-x)
#     * Diagonal integration of multimodal single-cell data emerges as a trending topic. However, empowering diagonal methods for novel biological discoveries requires bridging huge gaps. Here, we comment on **potential risks and future directions of diagonal integration for multimodal single-cell data**
# * [Bi-order multimodal integration of single-cell data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02679-x)
#     * Integration of single-cell multiomics profiles generated by different single-cell technologies from the same biological sample is still challenging. Previous approaches based on shared features have only provided approximate solutions. Here, we present **a novel mathematical solution named bi-order canonical correlation analysis (bi-CCA), which extends the widely used CCA approach to iteratively align the rows and the columns between data matrices.** Bi-CCA is generally applicable to combinations of any two single-cell modalities. Validations using co-assayed ground truth data and application to a CAR-NK study and a fetal muscle atlas demonstrate its capability in generating accurate multimodal co-embeddings and discovering cellular identity.
# * [Multimodal single-cell approaches shed light on T cell heterogeneity](https://www.sciencedirect.com/science/article/pii/S0952791519300469)
#     * Single-cell methods have revolutionized the study of T cell biology by enabling the identification and characterization of individual cells. This has led to a deeper understanding of T cell heterogeneity by generating functionally relevant measurements — like gene expression, surface markers, chromatin accessibility, T cell receptor sequences — in individual cells. While these methods are independently valuable, they can be augmented when applied jointly, either on separate cells from the same sample or on the same cells. **Multimodal approaches are already being deployed to characterize T cells in diverse disease contexts and demonstrate the value of having multiple insights into a cell’s function.** But, these data sets pose new statistical challenges for integration and joint analysis.
# * [Cobolt: integrative analysis of multimodal single-cell sequencing data](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02556-z)
#     * A growing number of single-cell sequencing platforms enable joint profiling of multiple omics from the same cells. We present **Cobolt, a novel method that not only allows for analyzing the data from joint-modality platforms, but provides a coherent framework for the integration of multiple datasets measured on different modalities.** We demonstrate its performance on multi-modality data of gene expression and chromatin accessibility and illustrate the integration abilities of Cobolt by jointly analyzing this multi-modality data with single-cell RNA-seq and ATAC-seq datasets.
# * [Human haematopoietic stem cell lineage commitment is a continuous process](https://www.nature.com/articles/ncb3493)
#     * Blood formation is believed to occur through stepwise progression of haematopoietic stem cells (HSCs) following a tree-like hierarchy of oligo-, bi- and unipotent progenitors. However, this model is based on the analysis of predefined flow-sorted cell populations. Here we integrated flow cytometric, transcriptomic and functional data at single-cell resolution to quantitatively map early differentiation of human HSCs towards lineage commitment. During homeostasis, individual HSCs gradually acquire lineage biases along multiple directions without passing through discrete hierarchically organized progenitor populations. Instead, unilineage-restricted cells emerge directly from a ‘continuum of low-primed undifferentiated haematopoietic stem and progenitor cells’ (CLOUD-HSPCs). **Distinct gene expression modules operate in a combinatorial manner to control stemness, early lineage priming and the subsequent progression into all major branches of haematopoiesis.** These data reveal a continuous landscape of human steady-state haematopoiesis downstream of HSCs and provide a basis for the understanding of haematopoietic malignancies.
# * [Normalizing and denoising protein expression data from droplet-based single cell profiling](https://www.nature.com/articles/s41467-022-29356-8)
#     * Multimodal single-cell profiling methods that measure protein expression with oligo-conjugated antibodies hold promise for comprehensive dissection of cellular heterogeneity, yet the resulting protein counts have substantial technical noise that can mask biological variations. Here we integrate experiments and computational analyses to reveal two major noise sources and develop a method called “dsb” (denoised and scaled by background) to normalize and denoise droplet-based protein expression data. **We discover that protein-specific noise originates from unbound antibodies encapsulated during droplet generation; this noise can thus be accurately estimated and corrected by utilizing protein levels in empty droplets**. We also find that isotype control antibodies and the background protein population average in each cell exhibit significant correlations across single cells, we thus use their shared variance to correct for cell-to-cell technical noise in each cell. We validate these findings by analyzing the performance of dsb in eight independent datasets spanning multiple technologies, including CITE-seq, ASAP-seq, and TEA-seq. Compared to existing normalization methods, our approach improves downstream analyses by better unmasking biologically meaningful cell populations. Our method is available as an open-source R package that interfaces easily with existing single cell software platforms such as Seurat, Bioconductor, and Scanpy.
# * [BABEL enables cross-modality translation between multiomic profiles at single-cell resolution](https://pubmed.ncbi.nlm.nih.gov/33827925/)
#     * Simultaneous profiling of multiomic modalities within a single cell is a grand challenge for single-cell biology. While there have been impressive technical innovations demonstrating feasibility-for example, generating paired measurements of single-cell transcriptome (single-cell RNA sequencing (scRNA-seq) and chromatin accessibility (single-cell assay for transposase-accessible chromatin using sequencing (scATAC-seq))-widespread application of joint profiling is challenging due to its experimental complexity, noise, and cost. Here, we introduce BABEL, a deep learning method that translates between the transcriptome and chromatin profiles of a single cell. **Leveraging an interoperable neural network model, BABEL can predict single-cell expression directly from a cell's scATAC-seq and vice versa after training on relevant data. This makes it possible to computationally synthesize paired multiomic measurements when only one modality is experimentally available. Across several paired single-cell ATAC and gene expression datasets in human and mouse, we validate that BABEL accurately translates between these modalities for individual cells.** BABEL also generalizes well to cell types within new biological contexts not seen during training. Starting from scATAC-seq of patient-derived basal cell carcinoma (BCC), BABEL generated single-cell expression that enabled fine-grained classification of complex cell states, despite having never seen BCC data. These predictions are comparable to analyses of experimental BCC scRNA-seq data for diverse cell types related to BABEL's training data. We further show that BABEL can incorporate additional single-cell data modalities, such as protein epitope profiling, thus enabling translation across chromatin, RNA, and protein. BABEL offers a powerful approach for data exploration and hypothesis generation.
# * [Current best practices in single-cell RNA-seq analysis: a tutorial](https://www.embopress.org/doi/full/10.15252/msb.20188746)
#     * Single-cell RNA-seq has enabled gene expression to be studied at an unprecedented resolution. The promise of this technology is attracting a growing user base for single-cell analysis methods. As more analysis tools are becoming available, it is becoming increasingly difficult to navigate this landscape and produce an up-to-date workflow to analyse one's data. Here, **we detail the steps of a typical single-cell RNA-seq analysis, including pre-processing (quality control, normalization, data correction, feature selection, and dimensionality reduction) and cell- and gene-level downstream analysis.** We formulate current best-practice recommendations for these steps based on independent comparison studies. We have integrated these best-practice recommendations into a workflow, which we apply to a public dataset to further illustrate how these steps work in practice. Our documented case study can be found at https://www.github.com/theislab/single-cell-tutorial. This review will serve as a workflow tutorial for new entrants into the field, and help established users update their analysis pipelines.
# 
# ### Methods 
# * [Chromatin accessibility profiling by ATAC-seq](https://www.nature.com/articles/s41596-022-00692-9)
#     * The assay for transposase-accessible chromatin using sequencing (ATAC-seq) provides a simple and scalable way to detect the unique chromatin landscape associated with a cell type and how it may be altered by perturbation or disease. ATAC-seq requires a relatively small number of input cells and does not require a priori knowledge of the epigenetic marks or transcription factors governing the dynamics of the system. Here we describe an updated and optimized protocol for ATAC-seq, called Omni-ATAC, that is applicable across a broad range of cell and tissue types. The ATAC-seq workflow has five main steps: sample preparation, transposition, library preparation, sequencing and data analysis. This protocol details the steps to generate and sequence ATAC-seq libraries, with recommendations for sample preparation and downstream bioinformatic analysis. ATAC-seq libraries for roughly 12 samples can be generated in 10 h by someone familiar with basic molecular biology, and downstream sequencing analysis can be implemented using benchmarked pipelines by someone with basic bioinformatics skills and with access to a high-performance computing environment.
# 
# ## Preprint
# * [Computational challenges in cell cycle analysis using single cell transcriptomics](https://arxiv.org/abs/2208.05229) 
# * [Multimodal single-cell chromatin analysis with Signac](https://www.biorxiv.org/content/10.1101/2020.11.09.373613v1.abstract) 
# * [MultiVI: deep generative model for the integration of multimodal-data](https://www.biorxiv.org/content/10.1101/2021.08.20.457057v1)

# # Experimental Details
# * Cell Lines Used: https://allcells.com/research-grade-tissue-products/mobilized-leukopak/
# * Multignome - ATAC + Gene Expression: https://www.10xgenomics.com/products/single-cell-multiome-atac-plus-gene-expression
#     * Chromatin accessibility to predict gene expression
# * CITESeq - Single Cell Gene Expression: https://support.10xgenomics.com/permalink/getting-started-single-cell-gene-expression-with-feature-barcoding-technology
#     * Cell Surface Reagent - https://www.biolegend.com/en-gb/products/totalseq-b-human-universal-cocktail-v1dot0-20960

# # Additional Information 
# * [EBI Ensemble Id Information](https://www.ebi.ac.uk/training/online/courses/ensembl-browsing-genomes/navigating-ensembl/investigating-a-gene/#:~:text=Ensembl%20gene%20IDs%20begin%20with,of%20species%20other%20than%20human)
# * [Eleven Grand Challenges in Single-Cell Data Science](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-1926-6)

# # Kaggle Notebooks (Background)
# * [scRNA-seq 🧬: Differential Expression with scVI](https://www.kaggle.com/code/hiramcho/scrna-seq-differential-expression-with-scvi/notebook)
# * [scRNA-seq 🧬: Scanpy & SCMER for Feature Selection](https://www.kaggle.com/code/hiramcho/scrna-seq-scanpy-scmer-for-feature-selection/notebook)
# * [scRNA-seq 🧬: scGAE with Spektral and RAPIDS](https://www.kaggle.com/code/hiramcho/scrna-seq-scgae-with-spektral-and-rapids/notebook)
# * [scATAC-seq 🧬: Feature Importance with TabNet](https://www.kaggle.com/code/hiramcho/scatac-seq-feature-importance-with-tabnet/notebook)
# * [scATAC-seq 🧬: EpiScanpy & PeakVI](https://www.kaggle.com/code/hiramcho/scatac-seq-episcanpy-peakvi)
# 
# 
# # Kaggle 
# * [MSCI CITEseq Quickstart](https://www.kaggle.com/code/ambrosm/msci-citeseq-quickstart)
# * [Data Loading - Getting Started](https://www.kaggle.com/code/peterholderrieth/getting-started-data-loading)
# * [MmSCel🧬Inst: EDA 🔍 & Stat. 🏴‍☠️ predictions](https://www.kaggle.com/code/jirkaborovec/mmscel-inst-eda-stat-predictions)
# * [MultiSCI- 📊 EDA](https://www.kaggle.com/code/vicsonsam/multisci-eda)
# * [🧫 Cell Analysis - quick h5 EDA](https://www.kaggle.com/code/queyrusi/cell-analysis-quick-h5-eda)
# * [Multimodal Single-Cell Integration](https://www.kaggle.com/code/erivanoliveirajr/multimodal-single-cell-integration)
# * [MSCI - CITEseq - TF/Keras NN Custom loss](https://www.kaggle.com/code/lucasmorin/msci-citeseq-tf-keras-nn-custom-loss)
# * [Complete EDA of MmSCel Integration Data](https://www.kaggle.com/code/leohash/complete-eda-of-mmscel-integration-data)
# * [【Tune LGBM Only - Final】CITE Task](https://www.kaggle.com/code/vuonglam/tune-lgbm-only-final-cite-task)
# * [MSCI EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense/data)
# * [MSCI Multiome Quickstart w/ Sparse Matrices](https://www.kaggle.com/code/fabiencrom/msci-multiome-quickstart-w-sparse-matrices) - 0.847
# * [【LGBM Baseline】MSCI CITEseq](https://www.kaggle.com/code/swimmy/lgbm-baseline-msci-citeseq) - 0.824
# * [MSCI Multiome Quickstart](https://www.kaggle.com/code/ambrosm/msci-multiome-quickstart)
# * [Simple Submission - Average by gene_id](https://www.kaggle.com/code/shuntarotanaka/simple-submission-average-by-gene-id) 0.741
# * [Reduce Memory Usage by 95% with Sparse Matrices](https://www.kaggle.com/code/sbunzini/reduce-memory-usage-by-95-with-sparse-matrices)
# 

# # External Notebooks / Packages 
# * [KNN Solution](https://github.com/adavoudi/msci_knn)

# # Learning Resources
# * [MIA: Multimodal Single-cell data, open benchmarks, and a NeurIPS 2021](https://www.biolegend.com/en-gb/products/totalseq-b-human-universal-cocktail-v1dot0-20960) - *video* 
# * [Open Problems in Single Cells Analysis](https://openproblems.bio/neurips_docs/data/about_multimodal/)
#     * Open problems in scAnalysis - 

# # Potentially Useful Packages (Python)
# * [muon](https://muon.readthedocs.io/en/latest/api/generated/muon.atac.pp.tfidf.html?highlight=tfidf) - muon is a Python framework for multimodal omics analysis. While there are many features that muon brings to the table, there are three key areas that its functionality is focused on.
# * [scanpy](https://scanpy.readthedocs.io/en/stable/index.html) - Scanpy is a scalable toolkit for analyzing single-cell gene expression data built jointly with anndata. It includes preprocessing, visualization, clustering, trajectory inference and differential expression testing. The Python-based implementation efficiently deals with datasets of more than one million cells.
# * [anndata](https://anndata.readthedocs.io/en/latest/#) - nndata is a Python package for handling annotated data matrices in memory and on disk, positioned between pandas and xarray. anndata offers a broad range of computationally efficient features including, among others, sparse data support, lazy operations, and a PyTorch interface.
# * [Xarray](https://docs.xarray.dev/en/v0.9.2/dask.html) - xarray (formerly xray) is an open source project and Python package that aims to bring the labeled data power of pandas to the physical sciences, by providing N-dimensional variants of the core pandas data structures. **This will help split up the large dataset**
# * [ivis](https://bering-ivis.readthedocs.io/en/latest/index.html) - ivis is a machine learning library for reducing dimensionality of very large datasets using Siamese Neural Networks. ivis preserves global data structures in a low-dimensional space, adds new data points to existing embeddings using a parametric mapping function, and scales linearly to millions of observations. The algorithm is described in detail in Structure-preserving visualisation of high dimensional single-cell datasets.
# * [epiScanpy](https://episcanpy.readthedocs.io/en/latest/) - EpiScanpy is a toolkit to analyse single-cell open chromatin (scATAC-seq) and single-cell DNA methylation (for example scBS-seq) data. EpiScanpy is the epigenomic extension of the very popular scRNA-seq analysis tool Scanpy (Genome Biology, 2018). For more information, read scanpy documentation.

# # Intro to Filetypes
# 

# # Last Year Competitions
# * (Novel team solution of ADT2GEX task in predict modality - [Presentation](https://drive.google.com/file/d/1aQss-KyfYlzdrBQcH5joiXMlTwpG5gdf/view) 
# * [Code / Methodology](https://github.com/openproblems-bio/neurips2021_multimodal_topmethods)

# # Appendix 
# 
# ## Cell Types 
# * MasP = Mast Cell Progenitor
# * MkP = Megakaryocyte Progenitor
# * NeuP = Neutrophil Progenitor
# * MoP = Monocyte Progenitor
# * EryP = Erythrocyte Progenitor
# * HSC = Hematoploetic Stem Cell
# * BP = B-Cell Progenitor

# # Thanks 
# At this point I just have aggregated the information from various notebooks and discussions as a way to keep track of all of the various notebooks 
# * Thomas Shelby - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344686
# * Daniel Burkhardt - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/344607
# * Kaggle Data Details the team at Cellarity - https://www.kaggle.com/competitions/open-problems-multimodal/data
# * Peter Holderrieth - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/345958
# * Marília Prat - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/346686
# * Alireza - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/346894
# * Jiwei Liu - https://www.kaggle.com/competitions/open-problems-multimodal/discussion/348792
# * AMBROSM - https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense/notebook
# * Lennard Henze - https://www.kaggle.com/code/leohash/complete-eda-of-mmscel-integration-data

# In[ ]:




