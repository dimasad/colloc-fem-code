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
    return u, y


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
    data_files = sorted(edir.glob('exp*.mat'))
    
