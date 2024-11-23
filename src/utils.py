import numpy as np
import pandas as pd
from scipy.stats import norm, gamma

def lp3_pdf(x, mu, sigma, gamma_val):
    # Calculate xi, beta, and alpha based on input parameters
    xi = mu - (2 * sigma) / gamma_val
    beta = 0.5 * sigma * gamma_val
    alpha = 4 / (gamma_val ** 2)

    # Shifted data for Gamma PDF calculation (log10 scale)
    if beta > 0:
        shifted_x = np.log10(x) - xi
        pdf_values = gamma.pdf(shifted_x, a=alpha, scale=beta)
    else:
        shifted_x = xi - np.log10(x)
        pdf_values = gamma.pdf(shifted_x, a=alpha, scale=-beta)

    # Adjust for the Jacobian
    jacobian = 1 / (x * np.log(10))
    pdf_values *= jacobian  # Apply the Jacobian to transform back to original scale

    return pdf_values

# def lp3_pdf(x, mu, sigma, gamma_val):
#     # Calculate xi, beta, and alpha based on input parameters
#     xi = mu - (2 * sigma) / gamma_val
#     beta = 0.5 * sigma * gamma_val
#     alpha = 4 / (gamma_val ** 2)

#     # Shifted data for Gamma PDF calculation (log10 scale)
#     if beta > 0:
#         shifted_x = np.log10(x) - xi
#     else:
#         shifted_x = xi - np.log10(x)

#     # Compute PDF using the Gamma distribution with alpha and beta
#     pdf_values = gamma.pdf(shifted_x, a=alpha, scale=np.abs(beta))

#     # Adjust for the Jacobian
#     jacobian = 1 / (x * np.log(10))
#     pdf_values *= jacobian  # Apply the Jacobian to transform back to original scale

#     return pdf_values

def lp3_cdf(x, mu, sigma, gamma_val):
    # Calculate xi, beta, and alpha based on input parameters
    xi = mu - (2 * sigma) / gamma_val
    beta = 0.5 * sigma * gamma_val
    alpha = 4 / (gamma_val ** 2)

    # Handle shifted_x and compute CDF based on the sign of beta
    if beta > 0:
        shifted_x = np.log10(x) - xi
        cdf_values = gamma.cdf(shifted_x, a=alpha, scale=beta)
    else:
        shifted_x = xi - np.log10(x)
        cdf_values = 1 - gamma.cdf(shifted_x, a=alpha, scale=-beta)

    return cdf_values

def lp3_ppf(p, mu, sigma, gamma_val):
    # Calculate xi, beta, and alpha based on input parameters
    xi = mu - (2 * sigma) / gamma_val
    beta = 0.5 * sigma * gamma_val
    alpha = 4 / (gamma_val ** 2)

    # Ensure probabilities are within valid range
    # p = np.asarray(p)
    # if np.any((p <= 0) | (p >= 1)):
    if p <= 0 or p >= 1:
        raise ValueError("Probabilities must be between 0 and 1 (exclusive).")

    # Compute the quantiles using the gamma distribution
    if beta > 0:
        q = gamma.ppf(p, a=alpha, scale=beta) + xi
    else:
        q = xi - gamma.ppf(1 - p, a=alpha, scale=-beta)

    # Transform back from log-space to the original scale
    quantiles = 10 ** q

    return quantiles

def lp3_mu_ppf(p, sigma, gamma_val, model_type='linear', t=0, **kwargs):
    # Calculate the time-dependent mean (mu) based on the trend type
    if model_type == 'linear':
        beta_0 = kwargs.get('beta_0', 0)
        beta_1 = kwargs.get('beta_1', 0)
        mu_t =  beta_0 + beta_1 * t
    elif model_type == 'exponential':
        beta_0 = kwargs.get('beta_0', 0.01)
        beta_1 = kwargs.get('beta_1', 0.01)
        mu_t = beta_0 * np.exp(-beta_1 * t)
    elif model_type == 'quadratic':
        beta_0 = kwargs.get('beta_0', 0)
        beta_1 = kwargs.get('beta_1', 0)
        beta_2 = kwargs.get('beta_2', 0)
        mu_t = beta_0 + beta_1 * t + beta_2 * t ** 2
    else:
        raise ValueError("Invalid trend_type. Choose from 'linear', 'exponential', 'quadratic'.")

    # Calculate LP3 distribution parameters
    xi_t = mu_t - (2 * sigma) / gamma_val
    beta_param = 0.5 * sigma * gamma_val
    alpha_param = 4 / (gamma_val ** 2)
    
    # Ensure probabilities are within valid range
    p = np.asarray(p)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Probabilities must be between 0 and 1 (exclusive).")

    # Compute the quantiles using the gamma distribution
    if beta_param > 0:
        q = gamma.ppf(p, a=alpha_param, scale=beta_param) + xi_t
    else:
        q = xi_t - gamma.ppf(1 - p, a=alpha_param, scale=-beta_param)
    
    # Transform back from log-space to the original scale
    quantiles = 10 ** q

    return quantiles

if __name__ == "__main__":
    # print(lp3_cdf(0.999999, 2.248632372, 0.098884524, 0.310975175))
    # print(lp3_cdf(115.9043534, 2.270976657, 0.101227947, -0.450195629))
    annual_maxima_csv = r"C:\ISYE6420\Homework\Project\data\OC_Dam.csv"
    df = pd.read_csv(annual_maxima_csv)
    df["zstd"] = -norm.ppf(df["Plotting_Position"])
    # convert flow to array
    data = df["Flow"].to_xarray()
    time_index = np.arange(len(data))
    beta0_lp3_linear, beta1_lp3_linear, sigma_lp3_linear, gamma_lp3_linear =  1.871538, -0.010892,  0.622922, -0.042078
    flood_quantiles_lp3 = lp3_mu_ppf(
        p=0.01, sigma = sigma_lp3_linear, gamma_val = gamma_lp3_linear,
        model_type = 'linear', t= time_index, beta_0 = beta0_lp3_linear, beta_1 = beta1_lp3_linear)
    print(len(flood_quantiles_lp3))