import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import pearson3

class LP3Fitting:
    def __init__(self, data):
        # Log-transform the data
        self.data = data
        self.log_data = np.log10(data)

        # Display basic statistics on log-transformed data
        self.mean_log = np.mean(self.log_data)
        self.variance_log = np.var(self.log_data, ddof=0)
        self.skew_log = stats.skew(self.log_data, bias=False)

        # Initial parameter guesses based on method of moments
        self.mu_initial = self.mean_log
        self.sigma_initial = np.sqrt(self.variance_log)
        self.gamma_initial = self.skew_log
        self.initial_params = [self.mu_initial, self.sigma_initial, self.gamma_initial]

    @staticmethod
    def moments_to_lp3(mu, sigma, gamma):
        xi = mu - (2 * sigma) / gamma
        beta = 0.5 * sigma * gamma
        alpha = 4 / (gamma ** 2)

        return xi, beta, alpha
    
    def neg_log_likelihood(self, params):
        mu, sigma, gamma = params

        # Handle the gamma == 0 case (Log-Normal distribution)
        if np.isclose(gamma, 0.0):
            # Log-Normal distribution log-likelihood
            log_likelihood = stats.norm.logpdf(self.log_data, loc=mu, scale=sigma)
            total_log_likelihood = np.sum(log_likelihood)
            return -total_log_likelihood
        
        # Compute derived parameters
        xi, beta, alpha = self.moments_to_lp3(mu, sigma, gamma)

        # Shift the data based on the sign of beta
        if beta > 0:
            shifted_x = self.log_data - xi
        else:
            shifted_x = xi - self.log_data

        # Compute the log-pdf for the Pearson Type III (Gamma) distribution on shifted data
        log_pdf = stats.gamma.logpdf(shifted_x, a=alpha, scale=np.abs(beta))

        # Compute the Jacobian for the change of variable (from original scale to log scale)
        # d(log10(x))/dx = 1 / (x * ln(10))
        jacobian = -1.0 / (self.data * np.log(10))

        # Total log-likelihood
        total_log_likelihood = np.sum(log_pdf + np.log(np.abs(jacobian)))

        return -total_log_likelihood

    def fit(self, method="Nelder-Mead", disp=False, tol=1e-12, max_iter=10000):
        result = minimize(
            self.neg_log_likelihood,
            self.initial_params,
            method=method,             # Optimization algorithm
            options={
                'disp': False,         # Set to True to see convergence messages
                'xatol': tol,          # Absolute error in xopt between iterations
                'fatol': tol,          # Absolute error in func(xopt) between iterations
                'maxiter': max_iter,   # Maximum number of iterations
                'maxfev': max_iter     # Maximum number of function evaluations
                }
        )

        return result
    
    def get_best_params(self, result, type = "MOM"):
        best_params = result.x
        mu, sigma, gamma = best_params
        
        if type == "MOM":
            return mu, sigma, gamma
        
        elif type == "MLE":
            xi, beta, alpha = self.moments_to_lp3(mu, sigma, gamma)
            return xi, beta, alpha
        
        else:   
            raise ValueError("Unknown type. Use 'MLE' or 'MOM'.")

if __name__ == "__main__":
    # data = np.array([
    #     122, 240, 125, 244, 230, 166, 214, 192, 99.1, 173, 195, 202,
    #     229, 172, 230, 156, 173, 158, 212, 172, 262, 263, 153, 154,
    #     146, 142, 164, 183, 317, 182, 161, 161, 164, 205, 201, 183,
    #     135, 204, 171, 331, 194, 250, 225, 164, 184, 174, 183, 205,
    #     98.8, 161, 237, 149, 167, 177, 238, 179, 239, 262, 185, 187,
    #     132, 117, 180, 235, 192, 173, 216, 337, 174
    # ])

    # lp3 = LP3Fitting(data)

    # result = lp3.fit(method="Nelder-Mead", disp=False, tol=1e-12, max_iter=10000)
    # mu, sigma, gamma = lp3.get_best_params(result, type = "MOM")
    # xi, beta, alpha = lp3.get_best_params(result, type = "MLE")

    # print("\nBest MLE Results (Log-Pearson Type III Distribution):")
    # print(f"Mu (μ): {mu:.5f}")
    # print(f"Sigma (σ): {sigma:.5f}")
    # print(f"Gamma (γ): {gamma:.5f}\n")

    # print("Converted MLE Parameters to Log-Pearson Type III:")
    # print(f"Location (xi): {xi:.5f}")
    # print(f"Scale (beta): {beta:.5f}")
    # print(f"Shape (alpha): {alpha:.5f}\n")

    mu = 2.26878
    sigma = 2.26878
    gamma = -0.02925
    # loc = 9.53025
    # scale = -0.00155
    # shape = 4674.10696

    exceedance_prob = 0.99 # Exceedance probability

    # Calculate the quantile
    log_quantile_99 = pearson3.ppf(exceedance_prob, gamma, loc=mu, scale=sigma)
    print(log_quantile_99)
    print(np.exp(log_quantile_99))

    # gageData = np.array([
    #     6290, 2700, 13100, 16900, 14600, 9600, 7740, 8490, 8130, 12000, 17200, 15000,
    #     12400, 6960, 6500, 5840, 10400, 18800, 21400, 22600, 14200, 11000, 12800, 15700,
    #     4740, 6950, 11800, 12100, 20600, 14600, 14600, 8900, 10600, 14200, 14100, 14100,
    #     12500, 7530, 13400, 17600, 13400, 19200, 16900, 15500, 14500, 21900, 10400, 7460
    # ])

    # normal = NormalFitting(gageData)
    # result = normal.fit(method="Nelder-Mead", disp=False, tol=1e-12, max_iter=10000)
    # mu, sigma = normal.get_best_params(result, type="MOM")

    # print("Best Results (Normal Distribution):")
    # print(f"Mu (μ): {mu:.0f}")
    # print(f"Sigma (σ): {sigma:.2f}\n")