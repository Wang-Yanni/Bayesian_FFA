# Bayesian Analysis in Flood Frequency Hydrology

This repository contains the source code, data, analysis, and documentation for the project **"Bayesian Analysis in Flood Frequency Hydrology: Verification with RMC-BestFit"**, undertaken as part of the coursework ISYE6420: Bayesian Statistics. The project explores the application of Bayesian methods in Flood Frequency Analysis (FFA). The methodology and results are rigorously validated against **BestFit**, a state-of-the-art hydrological software developed by the U.S. Army Corps of Engineers (USACE) Risk Management Center.
## Repository Structure

- **`src/`**  
  Contains Python source code and Jupyter notebooks:
  - **`.py` files**: Implement the Bayesian computational framework, including the Differential Evolution Markov Chain with Snooker Updater (DE-MCzS) algorithm for robust posterior sampling.
  - **Jupyter notebooks**: Facilitate detailed exploratory data analysis, model validation, and visualization.

- **`data/`**  
  Includes input datasets and results for the case studies:
  - **Harricana River dataset** (Canada): Used for stationary flood frequency analysis.
  - **O.C. Fisher Dam dataset** (USA): Used for both stationary and non-stationary flood frequency analyses.

- **`report/`**  
  LaTeX files and outputs for the comprehensive project report, documenting methodology, results, and comparison with **BestFit**.

- **`bestfit/`**  
  Outputs from the **BestFit** software used for validation of Bayesian model results.

## Project Description

### Overview

This study investigates the application of Bayesian analysis in flood frequency hydrology, employing an adaptive Differential Evolution Markov Chain with Snooker Updater (DE-MCzS) algorithm for parameter estimation. The results were verified against BestFit, a state-of-the-art hydrological software developed by the U.S. Army Corps of Engineers Risk Management Center (USACE-RMC). Its latest release (version 2.0) in October 2024, introduces the non-stationary feature that enables hydrologists to model temporal changes in flood frequency parameters.

### Highlights

1. **Bayesian Framework**:
   - Adopts the **Log-Pearson Type III (LP3)** distribution to model annual flood peaks, capturing the skewed nature of hydrological data.
   - Utilizes the **Differential Evolution Markov Chain with Snooker Updater (DE-MCzS)** algorithm for efficient posterior sampling, enabling robust exploration of parameter spaces.
   - Incorporates temporal changes in parameters in **non-stationary** flood flood frequency analysis.

2. **Case Study 1: Harricana River, Canada**:
   - Evaluates the Bayesian framework against the classical Maximum Likelihood Estimation (MLE) method in a stationary flood frequency context.
   - Validates Bayesian outputs, including posterior modes and predictive distributions, against results from **BestFit**.

3. **Case Study 2: O.C. Fisher Dam, USA**:
   - Extends the analysis to **non-stationary** models, exploring temporal trends in flood parameters using linear and exponential approaches.
   - Examines how time index selection impacts flood frequency curves, providing insights into climate-driven variability.


### Methodology

1. **Statistical Modeling**:
   - Applies the **LP3 distribution** to annual peak flow data, effectively handling skewness in hydrological datasets.
   - Employs **Z-score transformations** to standardize probabilities, ensuring consistency across analyses.

2. **Bayesian Estimation**:
   - Defines priors for LP3 parameters (mean, standard deviation, and skew) to incorporate expert knowledge.
   - Combines priors with observed data likelihoods to compute posteriors, capturing parameter uncertainties.
   - Leverages the **DE-MCzS algorithm** for Markov Chain Monte Carlo (MCMC) sampling, ensuring efficient and accurate posterior approximations.

3. **Model Validation**:
   - Compares Bayesian outputs to **BestFit** results, focusing on posterior modes, predictive distributions, and credible intervals.
   - Assesses model performance and quantifies uncertainty through credible intervals and posterior predictions.

## Instructions

### Prerequisites
- Python 3.11+
- Jupyter Notebook
- Required Python libraries (install via `requirements.txt` if provided).

### Running the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/Wang-Yanni/Bayesian_FFA.git
   cd Bayesian_FFA