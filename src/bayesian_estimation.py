import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, uniform, gamma
from typing import List, Callable, Tuple
import arviz as az

class LogLikelihoods:
    def __init__(self, data: np.ndarray):
        self.data = data

    def lp3(self, theta: np.ndarray):
        log_data = np.log10(self.data)
        mu, sigma, gamma = theta

        xi = mu - (2 * sigma) / gamma
        beta = 0.5 * sigma * gamma
        alpha = 4 / (gamma ** 2)
        
        # Handle the gamma == 0 case (Log-Normal distribution)
        if np.isclose(gamma, 0.0):
            # Log-Normal distribution log-likelihood
            log_likelihood = stats.norm.logpdf(log_data, loc=mu, scale=sigma)
            total_log_likelihood = np.sum(log_likelihood)
            return total_log_likelihood
        
        # Shift the data based on the sign of beta
        if beta > 0:
            shifted_x = log_data - xi
        else:
            shifted_x = xi - log_data

        # Compute the log-pdf for the Pearson Type III (Gamma) distribution on shifted data
        log_pdf = stats.gamma.logpdf(shifted_x, a=alpha, scale=np.abs(beta))

        # Compute the Jacobian for the change of variable (from original scale to log scale)
        # d(log10(x))/dx = 1 / (x * ln(10))
        jacobian = -1.0 / (self.data * np.log(10))

        # Total log-likelihood
        total_log_likelihood = np.sum(log_pdf + np.log(np.abs(jacobian)))
        return total_log_likelihood
    
    def lp3_mu(self, theta: np.ndarray, time_index: np.ndarray, model_type: str = "linear"):
        log_data = np.log10(self.data)

        # Determine model type and compute mu_t
        if model_type == "linear":
            beta_0, beta_1, sigma, gamma = theta
            mu_t = beta_0 + beta_1 * time_index
        elif model_type == "exponential":
            beta_0, beta_1, sigma, gamma = theta
            mu_t = beta_0 * np.exp(beta_1 * time_index)
        elif model_type == "quadratic":
            beta_0, beta_1, beta_2, sigma, gamma = theta
            mu_t = beta_0 + beta_1 * time_index + beta_2 * time_index**2
        else:
            raise ValueError("Invalid model_type. Choose from 'linear', 'exponential', or 'quadratic'.")
        
        # Pre-compute parameters for LP3
        xi_t = mu_t - (2 * sigma) / gamma
        beta_param = 0.5 * sigma * gamma
        alpha_param = 4 / (gamma ** 2)
        jacobian = -np.log(self.data * np.log(10))  # Precompute Jacobian

        # Check for log-normal case when gamma ~ 0
        if np.isclose(gamma, 0.0):
            log_likelihood = stats.norm.logpdf(log_data, loc=mu_t, scale=sigma)
            return np.sum(log_likelihood)

        # Compute shifted variable z_i
        if beta_param > 0:
            z_i = log_data - xi_t
        else:
            z_i = xi_t - log_data
        valid = z_i > 0

        # Compute log-pdf for valid z_i values
        log_pdf = np.zeros_like(log_data)
        # log_pdf = np.full_like(log_data, -1e6)  # Penalize invalid z_i
        log_pdf[valid] = stats.gamma.logpdf(z_i[valid], a=alpha_param, scale=np.abs(beta_param))

        # Total log-likelihood
        # if not np.any(valid):
        #     return -np.inf  # All z_i are invalid, reject
        total_log_likelihood = np.sum(log_pdf[valid] + jacobian[valid])
        # return total_log_likelihood        
        return total_log_likelihood if np.all(valid) else -np.inf
    
class BayesianEstimation:
    def __init__(self, data: np.ndarray, log_likelihood_func: Callable[[np.ndarray], float], prior: List[Callable[[], float]], prior_limits: List[Tuple[float, float]], seed: int = 123456):
        self.data = data
        self.prior = prior
        self.prior_limits = prior_limits
        self.log_likelihood_func = log_likelihood_func
        self.prng = np.random.default_rng(seed)

    def initialize_population(self, num_chains: int, num_para: int):      
        population = np.zeros((num_chains, num_para))

        # Generate the population matrix
        for i in range(num_para):
            # population[:, i] = [self.prior[i]() for _ in range(num_chains)]
            population[:, i] = self.prng.uniform(self.prior_limits[i][0], self.prior_limits[i][1], num_chains)

        return population
    
    def propose_new_state(self, current_state: np.ndarray, population: np.ndarray, G_val: float, noise: float) -> np.ndarray:
        # Sample two distinct indices
        num_chains, num_para = population.shape
        r1, r2 = self.prng.choice(num_chains, size=2, replace=False)

        # Calculate the proposal
        # e = np.power(norm.ppf(self.prng.choice(num_para)), num_para)
        # e = norm.ppf(self.prng.random(num_para)) * noise
        e = self.prng.normal(0, noise, size=current_state.shape)
        # now: e = np.power(norm.ppf(self.prng.random()), num_para)
        proposal = current_state + G_val * (population[r1] - population[r2]) + e

        # Check feasibility
        for i in range(len(proposal)):
            if proposal[i] < self.prior_limits[i][0] or proposal[i] > self.prior_limits[i][1]:
                return current_state  # Infeasible, reject
            
        return proposal
    
    def snooker_update(self, chain_idx: int, chains: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        num_chains, num_param = chains.shape
        # Get G from Uniform(1.2, 2.2)
        G = self.prng.uniform(1.2, 2.2)
        # Select another chain c != chain_idx
        c = chain_idx
        while c == chain_idx:
            c = self.prng.integers(0, num_chains)
        # Select two other random chains c1 and c2 != c and != chain_idx
        c1 = c
        while c1 == c or c1 == chain_idx:
            c1 = self.prng.integers(0, num_chains)
        c2 = c1
        while c2 == c or c2 == c1 or c2 == chain_idx:
            c2 = self.prng.integers(0, num_chains)
        # xi is the current state of chain_idx
        xi = chains[chain_idx]
        # z is the state of chain c
        z = chains[c]
        # Compute line xi - z
        line = xi - z
        line_norm_sq = np.dot(line, line)
        if line_norm_sq == 0:
            # Avoid division by zero
            return xi, z  # Reject
        # Project zr1 and zr2 onto the line
        zr1 = chains[c1]
        zr2 = chains[c2]
        proj_factor1 = np.dot(zr1 - z, line) / line_norm_sq
        zp1 = z + proj_factor1 * line
        proj_factor2 = np.dot(zr2 - z, line) / line_norm_sq
        zp2 = z + proj_factor2 * line
        # Calculate the proposal vector
        proposal = xi + G * (zp1 - zp2)
        # Check feasibility
        for i in range(num_param):
            if proposal[i] < self.prior_limits[i][0] or proposal[i] > self.prior_limits[i][1]:
                return xi, z  # Infeasible, reject
        return proposal, z
    
    def demcz_sampler(self, num_chains: int = 10, iterations: int = 11000, burn_in: int = 1000,
                      jump: float = None, jump_threshold: float = 0.2, noise: float = 1e-3,
                      snooker_threshold: float = 0.1, thinning_interval: int = 1):
        
        num_param = len(self.prior)
        if jump is None:
            jump = 2.38 / np.sqrt(2 * num_param)
        population = self.initialize_population(num_chains, num_param)

        # Check the range of input parameters
        if jump_threshold < 0 or jump_threshold > 1:
            raise ValueError("The jump_threshold parameter must be in the range [0, 1].")
        if noise <= 0:
            raise ValueError("The noise parameter must be positive.")
        if jump <= 0 or jump >= 2:
            raise ValueError("The jump parameter must be between 0 and 2.")
        if num_chains < 3:
            raise ValueError("There must be at least 3 chains.")
        if snooker_threshold < 0 or snooker_threshold > 0.5:
            raise ValueError("The snooker_threshold must be between 0 and 0.5.")
        if thinning_interval < 1 or not isinstance(thinning_interval, int):
            raise ValueError("The thinning_interval must be an integer greater than or equal to 1.")
        
        # Initialize the chains and log-likelihoods
        chains = population.copy()
        log_likelihoods = np.array([self.log_likelihood_func(chain) for chain in chains])
        accept_counts = np.zeros(num_chains, dtype=int)
        sample_counts = np.zeros(num_chains, dtype=int)

        # Generate a uniform distribution for the population
        log_uniforms = np.log(self.prng.uniform(size=(iterations, num_chains)))

        # MCMC sampling loop
        samples = []
        for i in range(iterations):
            for chain_idx in range(num_chains):
                # Update sample count
                sample_counts[chain_idx] += 1
                current_state = chains[chain_idx]
                current_log_likelihood = log_likelihoods[chain_idx]

                # Decide whether to do snooker update
                if (self.prng.uniform() <= snooker_threshold and
                    sample_counts[chain_idx] > 5 * thinning_interval):

                    # Do snooker update
                    proposal, z = self.snooker_update(chain_idx, chains)
                    if np.array_equal(proposal, current_state):
                        continue  # Proposal was rejected inside snooker_update

                    proposal_log_likelihood = self.log_likelihood_func(proposal)
                    if not np.isfinite(proposal_log_likelihood):
                        continue  # Reject proposal and continue
                    
                    # Compute distances
                    distance_xi_z = np.linalg.norm(current_state - z)
                    distance_proposal_z = np.linalg.norm(proposal - z)
                    if distance_xi_z == 0 or distance_proposal_z == 0:
                        continue  # Avoid division by zero
                    
                    # Compute log ratio
                    log_ratio = (proposal_log_likelihood +
                                 (num_param - 1) * np.log(distance_proposal_z) -
                                 current_log_likelihood -
                                 (num_param - 1) * np.log(distance_xi_z))
                    
                    if log_uniforms[i, chain_idx] < log_ratio:
                        # Accept the proposal
                        chains[chain_idx] = proposal
                        log_likelihoods[chain_idx] = proposal_log_likelihood
                        accept_counts[chain_idx] += 1

                else:
                    # Regular update
                    G_val = 1.0 if self.prng.uniform() < jump_threshold else jump
                    proposal = self.propose_new_state(current_state, population, G_val, noise)
                    proposal_log_likelihood = self.log_likelihood_func(proposal)
                    if not np.isfinite(proposal_log_likelihood):
                        continue  # Reject proposal and continue

                    # Compute acceptance probability
                    log_ratio = proposal_log_likelihood - current_log_likelihood
                    
                    if log_uniforms[i, chain_idx] < log_ratio:
                        # Accept the proposal
                        chains[chain_idx] = proposal
                        log_likelihoods[chain_idx] = proposal_log_likelihood
                        accept_counts[chain_idx] += 1

            # Update population
            population = chains.copy()

            # Store the sample for the iteration, if thinning_interval
            if (i % thinning_interval) == 0:
                samples.append(chains.copy())

        # Calculate the acceptance rate
        acceptance_rates = accept_counts / sample_counts

        # Discard the burn-in samples
        num_burnin_samples = burn_in // thinning_interval
        final_samples = np.array(samples[num_burnin_samples:])
        self.samples = final_samples

        return final_samples, acceptance_rates

    def calculate_log_likelihoods(self, variable) -> dict:
        # Flatten the samples across chains for each parameter
        samples = np.array(self.samples)
        flattened_samples = samples.reshape(-1, samples.shape[-1])
        
        log_likelihoods = []
        for sample in flattened_samples:
            log_likelihood = np.float64(self.log_likelihood_func(sample))
            log_likelihoods.append(log_likelihood)
        
        # Create a DataFrame
        df = pd.DataFrame(flattened_samples, columns=variable)
        df['LogLikelihood'] = log_likelihoods
        return df
    
    def compute_r_hat(self):
        if not hasattr(self, 'samples'):
            raise ValueError("Chains are not available. Run the sampler first.")
        
        samples = np.array(self.samples)  # Shape: (iterations, num_chains, num_params)
        num_iterations, num_chains, num_params = samples.shape

        r_hat_values = {}
        for param_idx in range(num_params):
            # Extract the chains for the parameter
            chains = samples[:, :, param_idx]  # Shape: (iterations, num_chains)

            # Calculate the mean per chain
            chain_means = np.mean(chains, axis=0)  # Shape: (num_chains,)
            
            # Overall mean across all chains and iterations
            overall_mean = np.mean(chain_means)
            
            # Between-chain variance B
            B = num_iterations * np.sum((chain_means - overall_mean)**2) / (num_chains - 1)
            
            # Within-chain variance W
            W = np.mean(np.var(chains, axis=0, ddof=1))  # ddof=1 for unbiased estimate
            
            # Estimate of target variance V_hat
            V_hat = (num_iterations - 1) / num_iterations * W + B / num_iterations
            
            # Gelman-Rubin statistic (R-hat)
            R_hat = np.sqrt(V_hat / W)
            r_hat_values[f"param_{param_idx + 1}"] = R_hat

        return r_hat_values
    
    def compute_r_hat_and_ess_arviz(self):
        # Reshape samples to ArviZ format (chains, iterations, parameters)
        samples_arviz = np.array(self.samples).swapaxes(0, 1)  # Shape: (chains, iterations, parameters)

        # Create dictionary for ArviZ
        posterior_dict = {
            "param_{}".format(i + 1): samples_arviz[:, :, i]
            for i in range(samples_arviz.shape[2])
        }

        # Convert to InferenceData
        inference_data = az.from_dict(posterior=posterior_dict)
        
        # Compute R-hat
        r_hat = az.rhat(inference_data)

        # Compute ESS
        ess = az.ess(inference_data)

        # Combine results
        results = {
            "r_hat": r_hat.to_dict(),  # Convert xarray DataArray to dictionary
            "ess": ess.to_dict()  # Convert xarray DataArray to dictionary
        }
        r_hat_values = {param: data['data'] for param, data in results['r_hat']['data_vars'].items()}
        ess_values = {param: data['data'] for param, data in results['ess']['data_vars'].items()}

        # Combine into a DataFrame
        formatted_df = pd.DataFrame({
            "R-hat": r_hat_values,
            "ESS": ess_values
        })
        return formatted_df
        
if __name__ == "__main__":
    annual_maxima_csv = r"C:\ISYE6420\Homework\Project\data\Harricana_River_at_Amos.csv"

    df = pd.read_csv(annual_maxima_csv)
    df["zstd"] = -norm.ppf(df["Plotting_Position"])

    # convert flow to array
    data = df["Flow"].to_xarray()

    time_index = np.arange(len(data))

    # Compute the prior limits
    beta_0_min, beta_0_max = 0, 4  # Prior for beta_0
    beta_1_min, beta_1_max = -1, 1  # Prior for beta_1 (slope of mu_t)
    sigma_min, sigma_max = 0, 2
    gamma_min, gamma_max = -2, 2

    # Define the prior distributions
    priors = [
        lambda: uniform.rvs(loc=beta_0_min, scale=beta_0_max - beta_0_min),
        lambda: uniform.rvs(loc=beta_1_min, scale=beta_1_max - beta_1_min),
        lambda: uniform.rvs(loc=sigma_min, scale=sigma_max - sigma_min),
        lambda: uniform.rvs(loc=gamma_min, scale=gamma_max - gamma_min)
    ]

    prior_limits = [
        (beta_0_min, beta_0_max),
        (beta_1_min, beta_1_max),
        (sigma_min, sigma_max),
        (gamma_min, gamma_max)
    ]

    log_likelihood_func_linear = lambda theta: LogLikelihoods(data).lp3_mu(
        theta, 
        time_index, 
        model_type = "linear"
        )

    bayesian_estimation_lp3 = BayesianEstimation(
        data=data, 
        log_likelihood_func=log_likelihood_func_linear, 
        prior=priors, 
        prior_limits=prior_limits, 
        seed = 253
        )

    # Run the DEMCz sampler
    samples, acceptance_rates = bayesian_estimation_lp3.demcz_sampler(
        num_chains=5, 
        iterations=44000, 
        burn_in=4000, 
        jump=0.84145, 
        jump_threshold=0.1,
        noise=1e-3,
        snooker_threshold=0.1,
        thinning_interval=20
        )
    np.save(r'C:\ISYE6420\Homework\Project\data\HRA\HRA_bayesian_linear_mu_lp3_samples.npy', samples)

    # Display the acceptance rates
    print("Acceptance rates:\n")
    print(acceptance_rates)

    summaries_lp3 = bayesian_estimation_lp3.calculate_log_likelihoods(
        variable=["beta_0", "beta_1", "sigma", "gamma"]
        )

    summaries_lp3.to_csv(r'C:\ISYE6420\Homework\Project\data\HRA\HRA_bayesian_linear_mu_lp3_summaries.csv', index=False)

    posterior_mode = summaries_lp3.loc[summaries_lp3['LogLikelihood'].idxmax()]
        
    print("\nPosterior mode:\n")
    print(posterior_mode)