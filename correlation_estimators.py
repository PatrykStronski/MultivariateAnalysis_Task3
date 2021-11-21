from scipy.stats import pearsonr
import pandas as pd

def calculate_pearson(data_sample: pd.DataFrame, predictor: str, target: str) -> float:
    p = pearsonr(data_sample[target], data_sample[predictor])
    return p.r