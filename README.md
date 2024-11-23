# Bayesian Analysis in Flood Frequency Hydrology

This repository contains the source code, data, analysis, and documentation for the project **"Bayesian Analysis in Flood Frequency Hydrology: Verification with RMC-BestFit"** for the coursework ISYE 6420: Bayesian Statistics. 
## Repository Structure

- **`src/`**  
  Contains Python source code for implementing the Bayesian computational framework and statistical methods:
  - `.py` files include classes and utility functions.
  - `.ipynb` files are used for analysis and visualizations of results.

- **`data/`**  
  Includes the datasets and output files used in the case studies:
  - *Harricana River dataset*: Stationary flood frequency analysis.
  - *O.C. Fisher Dam dataset*: Both stationary and non-stationary analyses.

- **`report/`**  
  Contains the LaTeX files for the project report detailing methodology, results, and discussion.

- **`bestfit/`**  
  Verifies the Bayesian framework results against the **BestFit** software developed by the USACE Risk Management Center.

## Project Overview

### Objective
- Employ a Bayesian computational framework for Flood Frequency Analysis (FFA).
- Validate the results using **RMC-BestFit**, focusing on parameter estimation, posterior distributions, and flood frequency curves.

### Highlights
- **Case Study 1**: Harricana River, Canada  
  Stationary analysis comparing Bayesian and MLE methods.
- **Case Study 2**: O.C. Fisher Dam, USA  
  Includes non-stationary analysis incorporating temporal dynamics in flood frequency parameters.

### Methodology
- Implemented **Log-Pearson Type III (LP3)** distribution for modeling annual flood peak data.
- Utilized an adaptive **Differential Evolution Markov Chain with Snooker Updater (DE-MCzS)** algorithm for Bayesian inference.
- Performed sensitivity analyses on parameters and time indices for non-stationary models.

## Instructions

### Prerequisites
- Python 3.11+
- Jupyter Notebook

### Running the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/Wang-Yanni/Bayesian_FFA.git
   cd Bayesian_FFA