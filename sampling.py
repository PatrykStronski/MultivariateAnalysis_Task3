import pandas as pd
import numpy as np

def pps(df: pd.DataFrame, property: str, size_f: float) -> pd.DataFrame:
    fire_size_total = df[property].sum()
    sample_size = int(df.size * size_f)

    df['cumulative_sum'] = df[property].cumsum()
    interval_width = int(fire_size_total/sample_size)

    num = interval_width #can be a random number also as in the example

    sampled_series = np.arange(num, fire_size_total, interval_width)
    cum_array = np.asarray(df['cumulative_sum'])
    idx = np.searchsorted(cum_array,sampled_series) #the heart of code
    result = cum_array[idx-1] 
    ndf = df[df.cumulative_sum.isin(result)]
    del ndf['cumulative_sum'] #so that new file doesn't have cum_sum column
    return ndf