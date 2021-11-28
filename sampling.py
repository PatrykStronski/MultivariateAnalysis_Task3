import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import math
import random
from typing import Callable

def pps(df: pd.DataFrame, property: str, size_f: float) -> pd.DataFrame:
    fire_size_total = df[property].sum()
    sample_size = int(df.size * size_f)

    df['cumulative_sum'] = df[property].cumsum()
    interval_width = int(fire_size_total/sample_size)

    num = interval_width 

    sampled_series = np.arange(num, fire_size_total, interval_width)
    cum_array = np.asarray(df['cumulative_sum'])
    idx = np.searchsorted(cum_array, sampled_series)
    result = cum_array[idx-1] 
    ndf = df[df.cumulative_sum.isin(result)]
    del ndf['cumulative_sum'] 
    return ndf 

def accept_reject(df: pd.DataFrame, property: str, dis: Callable, M: float):
    dis_params = dis.fit(df[property])
    df_sample = pd.DataFrame(columns=df.columns)
    df_size = df.shape[0]

    min_val = df[property].min()
    max_val = df[property].max()
    x = np.linspace(min_val, max_val, num=5000)

    ind = 0

    while ind < len(x) -1:
        min_s = x[ind]
        max_s = x[ind + 1]

        avg = (min_s + max_s) / 2
        sm = df.loc[(df[property] >= min_s) & (df[property] < max_s)]
        prob_current = M * sm.shape[0]/df_size
        prob_needed = dis.pdf(avg, dis_params[0], dis_params[1])

        r_m = random.random()

        if r_m <= (prob_needed/prob_current):
            df_sample = df_sample.append(sm)
    
        ind += 1
    return (df_sample, dis_params)
