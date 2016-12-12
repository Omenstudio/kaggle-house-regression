import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

from data_prepocessor import DataPreprocessor


def main():
    dp = DataPreprocessor()
    dp.read_all()
    model = GradientBoostingRegressor()
    model.fit(dp.train_inputs, dp.train_outputs)
    res = model.predict(dp.target_inputs)
    ans = pd.DataFrame(dp.target_df['Id'])
    ans['SalePrice'] = res
    ans.to_csv('output.csv', index=False)
    print('Finished')
    print(ans[:10])

main()
