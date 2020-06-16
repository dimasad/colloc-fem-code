#!/usr/bin/env python3

"""Monte Carlo blackbox collocation-based FEM system id."""


import argparse
import importlib
import os
import pathlib

import numpy as np
import scipy.io
import sympy
import sym2num.model

import fem
import symfem


# Reload modules for testing
for m in (fem, symfem):
    importlib.reload(m)


class Model(symfem.MaximumLikelihoodDTModel, symfem.BalancedDTModel):
    generated_name = 'GeneratedBalancedMaximumLikelihoodModel'


class MLProblem(fem.MaximumLikelihoodDTProblem, fem.BalancedDTProblem):
    pass


def load_data(datafile):
    data = scipy.io.loadmat(datafile)
    
    # Retrieve data
    u = data['u']
    y = data['y']
    
    N = len(y)
    ue = u[N//2:]
    ye = y[N//2:]
    uv = u[:N//2]
    yv = y[:N//2]
    return uv, yv, ue, ye


def load_estimates(datafile):
    estfile = datafile.parent / ('estim_' + datafile.name)
    return scipy.io.loadmat(estfile)


def predict(mdl, y, u, x0=None):
    A = mdl['A']
    B = mdl['B']
    C = mdl['C']
    D = mdl['D']
    Lun = mdl['Lun']
    
    nx = len(A)
    N = len(y)
    
    if x0 is None:
        x0 = np.zeros(nx)
    x = np.tile(x0, (N, 1))
    eun = np.empty_like(y)
    
    for k in range(N):
        eun[k] = y[k] - C @ x[k] - D @ u[k]
        if k+1 < N:
            x[k+1] = A @ x[k] + B @ u[k] + Lun @ eun[k]
    
    Rp = 1/N * eun.T @ eun
    sRp = np.linalg.cholesky(Rp)
    

def estimate(datafile):
    pass


def get_model(config):
    nx = config['nx']
    nu = config['nu']
    ny = config['ny']
    
    modname = f'{Model.generated_name}_nx{nx}_nu{nu}_ny{ny}'
    try:
        mod = importlib.import_module(modname)        
        cls = getattr(mod, Model.generated_name)
        return cls()
    except ImportError:
        symmodel = Model(nx=nx, nu=nu, ny=ny)
        with open(f'{modname}.py', mode='w') as f:
            print(symmodel.print_code(), file=f)
        return get_model(config)


def cmdline_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'edir', nargs='?', default='data/mc_experim', 
        help='experiment directory',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = cmdline_args()
    edir = pathlib.Path(args.edir)
    config = scipy.io.loadmat(edir / 'config.mat', squeeze_me=True)
    
    model = get_model(config)
    datafiles = sorted(edir.glob('exp*.mat'))
    
    for datafile in datafiles:
        uv, yv, ue, ye = load_data(datafile)
        in_prob = fem.InnovationDTProblem(model, ye, ue)
        ml_prob = MLProblem(model, ye, ue)
        
        raise SystemExit
