import pandas as pd
import numpy as np
import math
from typing import Callable

def pps(df: pd.DataFrame, property: str, size_f: float) -> pd.DataFrame:
    fire_size_total = df[property].sum()
    sample_size = int(df.size * size_f)

    df['cumulative_sum'] = df[property].cumsum()
    interval_width = int(fire_size_total/sample_size)

    num = interval_width #can be a random number also as in the example

    sampled_series = np.arange(num, fire_size_total, interval_width)
    cum_array = np.asarray(df['cumulative_sum'])
    idx = np.searchsorted(cum_array, sampled_series) #the heart of code
    result = cum_array[idx-1] 
    ndf = df[df.cumulative_sum.isin(result)]
    del ndf['cumulative_sum'] #so that new file doesn't have cum_sum column
    return ndf

def accept_reject(df: pd.DataFrame, property: str, dis: Callable, dis_params: tuple):
    df_size = df.shape[0]

    min_val = df[property].min()
    max_val = df[property].max()
    x = np.linspace(min_val, max_val, num=50)

    ind = 0

    while ind < len(x) -1:
        min_s = x[ind]
        max_s = x[ind + 1]

        avg = (min_s + max_s) / 2
        sm = df.loc[(df[property] >= min_s) & (df[property] < max_s)]
        prob_current = sm.shape[0]/df_size
        prob_needed = 1.0
        if len(dis_params) == 2:
            prob_needed = dis.pdf(avg, dis_params[0], dis_params[1])
        else:
            prob_needed = dis.pdf(avg, dis_params[0], dis_params[1], dis_params[2])

        if prob_current > prob_needed:
            to_drop = sm.sample(n=math.floor((prob_current - prob_needed) * df_size))
            df = df.drop(index=to_drop.index)
    
        ind += 1
    return df