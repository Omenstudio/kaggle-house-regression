import numpy as np
from sklearn import linear_model, svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from data_prepocessor import DataPreprocessor


def test_GBR():
    dp = DataPreprocessor()
    dp.read_all()
    # Test params
    criterions = ['friedman_mse', 'mse']
    for criterion in criterions:
        model = GradientBoostingRegressor(criterion=criterion)
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('criterions =', criterion, res.mean(), res.std())
    print('--------------')
    lrs = [1.0, 0.9, 0.5, 0.1, 0.05, 0.01, 0.001]
    for lr in lrs:
        model = GradientBoostingRegressor(learning_rate=lr, criterion='mse')
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('lr =', lr, res.mean(), res.std())
    print('--------------')
    max_depths = [1, 2, 3, 4, 5, 8, 12]
    for max_depth in max_depths:
        model = GradientBoostingRegressor(max_depth=max_depth, criterion='mse')
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('max_depth =', max_depth, res.mean(), res.std())
    print('--------------')
    alphas = [.99, .5, .1, .05, .01, .001]
    for alpha in alphas:
        model = GradientBoostingRegressor(alpha=alpha, criterion='mse')
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('alpha =', alpha, res.mean(), res.std())
    print('--------------')


def test_KernelRidge():
    dp = DataPreprocessor()
    dp.read_all()
    # Test params
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for kernel in kernels:
        model = KernelRidge(kernel=kernel)
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('criterions =', kernel, res.mean(), res.std())
    print('--------------')
    alphas = [20., 10., 5., 2., 1., .9, .1, .01]
    for alpha in alphas:
        model = KernelRidge(alpha=alpha)
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('alpha =', alpha, res.mean(), res.std())
    print('--------------')
    gammas = [1., .8, .6, .4, .2, .05, .01]
    for gamma in gammas:
        model = KernelRidge(gamma=gamma, kernel='sigmoid')
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('gamma =', gamma, res.mean(), res.std())
    print('--------------')


def test_Ridge():
    dp = DataPreprocessor()
    dp.read_all()
    # Test params
    normalizes = [True, False]
    for normalize in normalizes:
        model = linear_model.Ridge(normalize=normalize)
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('normalize =', normalize, res.mean(), res.std())
    print('--------------')
    fit_intercepts = [True, False]
    for fit_intercept in fit_intercepts:
        model = linear_model.Ridge(normalize=True, fit_intercept=fit_intercept)
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('fit_intercept =', fit_intercept, res.mean(), res.std())
    solvers = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
    for solver in solvers:
        model = linear_model.Ridge(normalize=True, fit_intercept=True, solver=solver)
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('solver =', solver, res.mean(), res.std())
    print('--------------')
    tols = np.linspace(.05, 1., 20)
    for tol in tols:
        model = linear_model.Ridge(normalize=True, fit_intercept=True, tol=tol)
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('tol =', tol, res.mean(), res.std())
    print('--------------')
    alphas = np.linspace(.01, 2., 20)
    for alpha in alphas:
        model = linear_model.Ridge(normalize=True, alpha=alpha)
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        print('alpha =', alpha, res.mean(), res.std())
    print('--------------')


def test_all():
    dp = DataPreprocessor()
    dp.read_all()

    models = [
        GradientBoostingRegressor(),
        MLPRegressor(),
        DecisionTreeRegressor(),
        GaussianProcessRegressor(),
        KNeighborsRegressor(),
        svm.SVR(),
        KernelRidge(),
        linear_model.HuberRegressor(),
        linear_model.BayesianRidge(),
        linear_model.LassoLars(alpha=.1),
        linear_model.Lars(n_nonzero_coefs=25),
        linear_model.ElasticNet(tol=1),
        linear_model.Lasso(alpha=0.1, tol=0.1),
        linear_model.Lasso(alpha=0.3, tol=1),
        LinearRegression(),
        linear_model.Ridge(),
    ]
    results = []
    print(0)
    for model in models:
        res = cross_val_score(model, dp.train_inputs, dp.train_outputs, cv=KFold(n_splits=20))
        results.append([res.mean(), res.std(), model])
        print(1)
    for r in sorted(results):
        print(r[2], '\nAccuracy (mean std): {:.4f} {:.4f}'.format(r[0], r[1]), '\n--------------')


test_all()
