# Glucose Level Prediction via Random Forest in Genomic Data

This repository contains the statistical analysis pipeline developed for evaluating genetic markers (SNPs) and sociodemographic variables. Maintained by a research group from the Department of Statistics at the Federal University of São Carlos (UFSCar), this project focuses on robust predictive modeling for genomic data.

## Reference / Related Publication
The methodologies implemented in this pipeline are based on foundational concepts discussed in the following work:

* **arXiv:** [1109.0152](https://arxiv.org/abs/1109.0152)
## Data Structure
The project integrates three distinct data sources, harmonized through unique individual identifiers.

### 1. LD-Imputed Genetic Database
Located in the `Banco Genetico Imputado LD` directory, this folder contains `.raw` files (PLINK output) for each chromosome.

*   **Naming Convention:** `chr{num}_imputado_LD.raw`
*   **Sample Size:** 720 individuals.
*   **Metadata:** The first six columns contain pedigree and phenotype identifiers. The `IID` column serves as the primary key for sorting and data integration.
*   **Genetic Predictors:** Numerically encoded SNPs (e.g., `rs62224618_T`).

### 2. Marker Maps
Files detailing the exact locations of the SNPs.

*   **Naming Convention:** `chr{num}map_imputado_LD.map`
*   **Content:** Chromosome code, SNP Name, Position in Morgans, and Base pair coordinate.

### 3. Phenotypes and Covariates
The file `banco_fenotipos_conformal.csv` contains the target outcomes and clinical variables.

*   **Key Variable:** `samplefilename`, which must be strictly paired and sorted alongside the `IID` variable from the genetic database.
*   **Categorical Variables:** Includes data regarding smoking habits, race/color, marital status, occupation, and alcohol consumption.

## Statistical Methodology
The data processing and analysis follow strict statistical rigor to handle the high dimensionality inherent to genomic data (p >> n):

*   **Preprocessing:** Detailed in the `info_pre_processamento` folder, covering imputation criteria and quality control protocols.
*   **Encoding:** Categorical variables are processed using One-Hot Encoding to prevent arbitrary hierarchies in nominal data.
*   **Model:** Implementation of the Random Forest Regressor and Classifier algorithm.
*   **Validation:** Error estimation is conducted via Permutation Score, and methods that you can see in the paper.

## Repository Notes & Data Access
Due to GitHub's storage constraints and the sensitive nature of genomic data, the raw `.raw` and `.csv` files are not hosted in this repository. The scripts are configured to read the local directory structure exactly as described above.

## Contact
For inquiries regarding the methodology or to request access to the raw data for academic reproduction, please contact the research team via the email provided in this GitHub profile.
