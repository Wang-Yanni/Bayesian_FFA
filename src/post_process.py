import numpy as np
import pandas as pd
from scipy.stats import norm
from utils import lp3_cdf, lp3_ppf

class PostProcess:
    def __init__(self, df, quantiles):
        self.df = df
        self.quantiles = quantiles
        self.df_ppf = None
        self.df_intervals = None
    
    def calculate_ppf_df(self):
        df_ppf = self.df.copy()
        for p in self.quantiles:
            df_ppf[f'ppf_{p}'] = df_ppf.apply(lambda row: lp3_ppf(p, row['mu'], row['sigma'], row['gamma']), axis=1)
        
        self.df_ppf = df_ppf
        return df_ppf

    def calculate_predictive_intervals_df(self):
        if self.df_ppf is None:
            df_ppf = self.calculate_ppf_df()
        else:
            df_ppf = self.df_ppf

        ppf_columns = df_ppf.filter(like='ppf_')
        log_ppf_min = np.log10(ppf_columns.min().min())
        log_ppf_max = np.log10(ppf_columns.max().max())
        
        # Split into 20 intervals
        n_intervals = 20
        log_intervals = np.linspace(log_ppf_min, log_ppf_max, n_intervals)
        intervals = [10 ** x for x in log_intervals]

        # Calculate the PDF for each interval
        df_pred = self.df.copy()
        for i in range(n_intervals):
            x = intervals[i]
            df_pred[f'cdf_{i}'] = df_pred.apply(lambda row: lp3_cdf(x, row['mu'], row['sigma'], row['gamma']), axis=1)

        # Calculate the average PDF and Z-Scores
        pdf_columns = df_pred.filter(like='cdf_')
        averages = [df_pred[col].mean() for col in pdf_columns]
        x_values_z_scores = [-norm.ppf(1-x) for x in averages]

        # Create a DataFrame
        df_intervals = pd.DataFrame(
            {
                'log_intervals': log_intervals,
                'intervals': intervals,
                'PDF_average': averages,
                'Z_scores': x_values_z_scores
            }
        )

        self.df_intervals = df_intervals
        return df_intervals

    def calculate_posterior_predictive_and_CI_df(self):
        # Load the PPF and Predictive Intervals DataFrames
        if self.df_ppf is None:
            df_ppf = self.calculate_ppf_df()
        else:
            df_ppf = self.df_ppf
        
        if self.df_intervals is None:
            df_intervals = self.calculate_predictive_intervals_df()
        else:
            df_intervals = self.df_intervals

        # Filter the PPF columns
        df_post_pred = df_ppf.copy()
        ppf_columns = df_post_pred.filter(like='ppf_')
        Z_scores = [-norm.ppf(1-p) for p in self.quantiles]

        # Calculate the Posterior Predictive and Percentiles
        posterior_predictive = np.interp(Z_scores, df_intervals['Z_scores'], df_intervals['intervals'])
        percentile_5 =  [df_post_pred[col].quantile(0.05) for col in ppf_columns]
        percentile_95 =  [df_post_pred[col].quantile(0.95) for col in ppf_columns]

        df_summary = pd.DataFrame(
            {   
                'quantile': self.quantiles,
                'posterior_predictive': posterior_predictive,
                'percentile_5': percentile_5,
                'percentile_95': percentile_95
            }
        )

        return df_summary


if __name__ == '__main__':
    csv_file = r"C:\ISYE6420\Homework\Project\data\HRA\HRA_bayesian_lp3_summaries.csv"
    quantiles = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 
                0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998,
                0.9999, 0.99995, 0.99998, 0.99999, 0.999995, 0.999998, 
                0.999999] 
    
    pp = PostProcess(csv_file, quantiles)
    df_ppf = pp.calculate_posterior_predictive_and_CI_df()
    print(df_ppf)
    
