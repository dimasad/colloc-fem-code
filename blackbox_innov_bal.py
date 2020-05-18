"""Tests with the ATTAS aircraft short-period mode estimation."""


import importlib
import os

import numpy as np
import scipy.io
import sympy
import sym2num.model

import fem
import symfem


# Reload modules for testing
for m in (fem, symfem):
    importlib.reload(m)


def save_generated_model(symmodel):
    clsname = type(symmodel).__name__
    nx = symmodel.nx
    nu = symmodel.nu
    ny = symmodel.ny
    with open(f'{clsname}_nx{nx}_nu{nu}_ny{ny}.py', mode='w') as f:
        code = symmodel.print_code()
        print(code, file=f)


def get_model(nx, nu, ny):
    clsname = 'InnovationBalDTModel'
    modname = f'{clsname}_nx{nx}_nu{nu}_ny{ny}'
    mod = importlib.import_module(modname)
    
    genclsname = f'Generated{clsname}'
    cls = getattr(mod, genclsname)
    return cls()


def load_data():
    # Retrieve data
    u = np.loadtxt('/tmp/u.txt')
    y = np.loadtxt('/tmp/y.txt')
    return u, y


if __name__ == '__main__':
    nx = 3
    nu = 2
    ny = 2
    
    # Load experiment data
    u, y = load_data()
    symmodel = symfem.InnovationBalDTModel(nx=nx, nu=nu, ny=ny)
    model = symmodel.compile_class()()
    problem = fem.InnovationBalDTProblem(model, y, u)
    N = len(y)
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = np.loadtxt('/tmp/a.txt')
    var0['B'][:] = np.loadtxt('/tmp/b.txt')
    var0['C'][:] = np.loadtxt('/tmp/c.txt')
    var0['D'][:] = np.loadtxt('/tmp/d.txt')
    var0['L'][:] = np.loadtxt('/tmp/k.txt')
    var0['x'][:] = np.loadtxt('/tmp/xpred.txt')
    var0['W_diag'][:] = np.loadtxt('/tmp/gram.txt')
    var0['isRp_tril'][:] = np.loadtxt('/tmp/isRp.txt')[np.tril_indices(ny)]
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['isRp_tril'][symfem.tril_diag(ny)] = 0
    var_L['W_diag'][:] = 0
    
    # Define bounds for constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    var_constr_L = problem.unpack_constraints(constr_L)
    var_constr_U = problem.unpack_constraints(constr_U)
    
    # Define problem scaling
    obj_scale = -1.0
    constr_scale = np.ones(problem.ncons)
    var_constr_scale = problem.unpack_constraints(constr_scale)
    var_constr_scale['innovation'][:] = 1e3
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['isRp_tril'][:] = 1e-2
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 10.0)
        nlp.add_num_option('tol', 1e-5)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)    
    xopt = opt['x']
    A = opt['A']
    B = opt['B']
    C = opt['C']
    D = opt['D']
    L = opt['L']
    W = np.diag(opt['W_diag'])
    ybias = opt['ybias']
    isRp = symfem.tril_mat(ny, opt['isRp_tril'])
    yopt = xopt @ C.T + u @ D.T + ybias
    eopt = opt['e']
