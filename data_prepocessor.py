from copy import copy

import numpy as np
import pandas as pd


class DataPreprocessor:

    def __init__(self):
        self.__transform_dicts = dict()
        self.target_inputs = None
        self.train_inputs = None
        self.train_outputs = None
        self.target_set = None
        self.target_df = None

    def read_all(self):
        df_train = pd.read_csv('train.csv', header=0)
        df_target = pd.read_csv('test.csv', header=0)
        df_target['SalePrice'] = None
        self.target_df = df_target
        train_len = df_train.shape[0]
        target_len = df_target.shape[0]
        df = df_train
        df = df.append(copy(df_target))
        # Заменяем пропущенные значения
        #
        df.loc[df.LotFrontage.isnull(), "LotFrontage"] = df.loc[df.LotFrontage.notnull(), "LotFrontage"].median()
        df.loc[df.Alley.isnull(), "Alley"] = 'NoAC'
        df.loc[df.MasVnrType.isnull(), "MasVnrType"] = 'NoVen'
        df.loc[df.MasVnrArea.isnull(), "MasVnrArea"] = 0
        df.loc[df.BsmtQual.isnull(), "BsmtQual"] = 'NoBsmt'
        df.loc[df.BsmtCond.isnull(), "BsmtCond"] = 'NoBsmt'
        df.loc[df.BsmtExposure.isnull(), "BsmtExposure"] = 'NoBsmt'
        df.loc[df.BsmtFinType1.isnull(), "BsmtFinType1"] = 'NoBsmt'
        df.loc[df.BsmtFinType2.isnull(), "BsmtFinType2"] = 'NoBsmt'
        df.loc[df.Electrical.isnull(), "Electrical"] = 'Mix'
        df.loc[df.FireplaceQu.isnull(), "FireplaceQu"] = 'noFrp'
        df.loc[df.GarageType.isnull(), "GarageType"] = 'noGrg'
        df.loc[df.GarageYrBlt.isnull(), "GarageYrBlt"] = df.loc[df.GarageYrBlt.isnull(), "YearBuilt"]
        df.loc[df.GarageFinish.isnull(), "GarageFinish"] = 'noGrg'
        df.loc[df.GarageQual.isnull(), "GarageQual"] = 'noGrg'
        df.loc[df.GarageCond.isnull(), "GarageCond"] = 'noGrg'
        df.loc[df.GarageCars.isnull(), "GarageCars"] = 0
        df.loc[df.GarageArea.isnull(), "GarageArea"] = 0
        df.loc[df.PoolQC.isnull(), "PoolQC"] = 'noPol'
        df.loc[df.Fence.isnull(), "Fence"] = 'noFnc'
        df.loc[df.MiscFeature.isnull(), "MiscFeature"] = 'noMsc'
        # additional to test.csv
        df.loc[df.MSZoning.isnull(), "MSZoning"] = 'C (all)'
        df.loc[df.Utilities.isnull(), "Utilities"] = 'AllPub'
        df.loc[df.Exterior1st.isnull(), "Exterior1st"] = 'VinylSd'
        df.loc[df.Exterior2nd.isnull(), "Exterior2nd"] = 'VinylSd'
        df.loc[df.KitchenQual.isnull(), "KitchenQual"] = 'TA'
        df.loc[df.Functional.isnull(), "Functional"] = 'Typ'
        df.loc[df.SaleType.isnull(), "SaleType"] = 'WD'
        df.loc[df.BsmtFinSF1.isnull(), "BsmtFinSF1"] = 0.
        df.loc[df.BsmtFinSF2.isnull(), "BsmtFinSF2"] = 0.
        df.loc[df.BsmtUnfSF.isnull(), "BsmtUnfSF"] = 0
        df.loc[df.TotalBsmtSF.isnull(), "TotalBsmtSF"] = 0
        df.loc[df.BsmtFullBath.isnull(), "BsmtFullBath"] = 0
        df.loc[df.BsmtHalfBath.isnull(), "BsmtHalfBath"] = 0
        # Обрабатываем категорийные атрибуты
        categorical_features = ['MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig',
                                'LandSlope', 'Neighborhood', 'LotShape', 'Condition1', 'Condition2', 'BldgType',
                                'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                                'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                                'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                                'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                                'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
                                'SaleCondition']
        for categorical_feature in categorical_features:
            self.__transform_categorical_feature(df, categorical_feature)
        # Заменяем категорийные атрибуты бинарными векторами
        self.train_outputs = df.values[:train_len, -1]
        df = df.drop('SalePrice', 1)
        df = df.drop('Id', 1)
        categorical_features += ['MSSubClass', 'OverallQual', 'OverallCond']
        df_with_dummies = pd.get_dummies(data=df, columns=categorical_features)
        self.train_inputs = df_with_dummies.values[:train_len]
        self.target_inputs = df_with_dummies.values[train_len:]
        if len(self.train_inputs) != train_len or len(self.target_inputs) != target_len:
            raise Exception('Wrong division to train and target set')

    def __transform_categorical_feature(self, dataset, feature_name):
        if feature_name not in self.__transform_dicts:
            feature_unique_list = list(enumerate(np.unique(dataset[feature_name])))
            feature_dict = {name: i for i, name in feature_unique_list}
            self.__transform_dicts[feature_name] = feature_dict
        dataset[feature_name] = dataset[feature_name].map(lambda x: self.__transform_dicts[feature_name][x]).astype(int)
