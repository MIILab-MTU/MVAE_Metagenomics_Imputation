# Multi-View Variational Autoencoder for Missing Value Imputation in Untargeted Metabolomics 

url: https://pubmed.ncbi.nlm.nih.gov/37873011/

## Abstract

*Background*: Missing data is a common challenge in mass spectrometry-based metabolomics, which can lead to biased and incomplete analyses. The integration of whole-genome sequencing (WGS) data with metabolomics data has emerged as a promising approach to enhance the accuracy of data imputation in metabolomics studies.
*Method*: In this study, we propose a novel method that leverages the information from WGS data and reference metabolites to impute unknown metabolites. Our approach utilizes a multi-view variational autoencoder to jointly model the burden score, polygenetic risk score (PGS), and linkage disequilibrium (LD) pruned single nucleotide polymorphisms (SNPs) for feature extraction and missing metabolomics data imputation. By learning the latent representations of both omics data, our method can effectively impute missing metabolomics values based on genomic information.
*Results*: We evaluate the performance of our method on empirical metabolomics datasets with missing values and demonstrate its superiority compared to conventional imputation techniques. Using 35 template metabolites derived burden scores, PGS and LD-pruned SNPs, the proposed methods achieved 𝑅𝑅2-scores > 0.01 for 71.55% of metabolites.
*Conclusion*: The integration of WGS data in metabolomics imputation not only improves data completeness but also enhances downstream analyses, paving the way for more comprehensive and accurate investigations of metabolic pathways and disease associations. Our findings offer valuable insights into the potential benefits of utilizing WGS data for metabolomics data imputation and underscore the importance of leveraging multi-modal data integration in precision medicine research.

## Environment
tensorflow==2.5
numpy==1.20
matplotlib==3.3
scikit-learn=1.1
pandas=1.4

## Use your own data

please see utils.utils.py, `load_data_within`.

## Training

python main.py 
