import pandas as pd
from sampling import pps
from correlation_estimators import calculate_pearson

predictors = ['Temp_pre_7', 'Hum_pre_7', 'Wind_pre_7', 'Prec_pre_7', 'remoteness']
targets = ['fire_size', 'fire_mag']

df = pd.read_csv('./data/FW_Veg_Rem_Combined.csv')

for pred in predictors:
    for tar in targets:
        pr = calculate_pearson(df, pred, tar)
        print(f'The Pearsons coefficient for {tar} and {pred} is {pr}')