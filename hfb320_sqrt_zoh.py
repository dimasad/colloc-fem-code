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


def load_data():
    # Retrieve data
    # Load experiment data
    dirname = os.path.dirname(__file__)
    data = np.loadtxt(os.path.join(dirname, 'data', 'hfb320_1_10.asc'))
    Ts = 0.1
    n = len(data)
    t = np.arange(n) * Ts
    y = data[:, 4:11]
    u = data[:, [1,3]]
    
    # Shift and rescale
    yscale = np.r_[0.15, 70, 15, 30, 10, 5, 0.8]
    y = (y - [106, 0.11, 0.1, 0, 0, 0.95, -9.5]) * yscale
    u = (u - [-0.007, 11600]) * [100, 0.01]
    
    return t, u, y[:, :]


def save_generated_model(symmodel):
    clsname = type(symmodel).__name__
    nx = symmodel.nx
    nu = symmodel.nu
    ny = symmodel.ny
    with open(f'{clsname}_nx{nx}_nu{nu}_ny{ny}.py', mode='w') as f:
        code = symmodel.print_code()
        print(code, file=f)


def get_model(nx, nu, ny):
    clsname = 'NaturalSqrtZOHModel'
    modname = f'{clsname}_nx{nx}_nu{nu}_ny{ny}'
    mod = importlib.import_module(modname)
    
    genclsname = f'Generated{clsname}'
    cls = getattr(mod, genclsname)
    return cls()


if __name__ == '__main__':
    nx = 4
    nu = 2
    ny = 7
    
    # Load experiment data
    t, u, y = load_data()
    symmodel = symfem.NaturalSqrtZOHModel(nx=nx, nu=nu, ny=ny)
    model = symmodel.compile_class()()
    #model = get_model(nx, nu, ny)
    model.dt = t[1] - t[0]
    problem = fem.NaturalSqrtZOHProblem(model, y, u)
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = np.eye(nx)
    var0['C'][:] = np.eye(ny, nx)
    var0['D'][:] = np.zeros((ny, nu))
    var0['L'][:] = np.eye(nx, ny)
    var0['KsRp'][:] = np.eye(nx, ny) * 1e-2
    var0['x'][:] = y[:, :nx]
    var0['isRp_tril'][symfem.tril_diag(ny)] = 1e2
    var0['sRp_tril'][symfem.tril_diag(ny)] = 1e-2
    var0['sQ_tril'][symfem.tril_diag(nx)] = 1e-2
    var0['sR_tril'][symfem.tril_diag(ny)] = 1e-2
    var0['sPp_tril'][symfem.tril_diag(nx)] = 1e-2
    var0['sPc_tril'][symfem.tril_diag(nx)] = 1e-2
    var0['pred_orth'][:] = np.eye(2*nx, nx)
    var0['corr_orth'][:] = np.eye(nx + ny, nx + ny)
    var0['Qc'][:] = np.eye(nx) * 1e-2
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['Ac'][2, :3] = 0
    var_U['Ac'][2, :3] = 0
    var_L['C'][:nx] = np.eye(nx)
    var_U['C'][:nx] = np.eye(nx)
    var_L['D'][:nx] = np.zeros((nx, nu))
    var_U['D'][:nx] = np.zeros((nx, nu))
    var_L['isRp_tril'][symfem.tril_diag(ny)] = 0
    var_L['sRp_tril'][symfem.tril_diag(ny)] = 1e-7
    var_L['sPp_tril'][symfem.tril_diag(nx)] = 0
    var_L['sPc_tril'][symfem.tril_diag(nx)] = 0
    var_L['sQ_tril'][symfem.tril_diag(nx)] = 0
    var_L['sR_tril'][symfem.tril_diag(ny)] = 1e-4
    var_L['sR_tril'][0] = 0.02
    var_L['sR_tril'][~symfem.tril_diag(ny)] = 0
    var_U['sR_tril'][~symfem.tril_diag(ny)] = 0
    var_L['Qc'][np.arange(nx), np.arange(nx)] = 0
    
    # Define bounds for constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    # Define problem scaling
    obj_scale = -1.0
    constr_scale = np.ones(problem.ncons)
    var_constr_scale = problem.unpack_constraints(constr_scale)
    var_constr_scale['pred_cov'][:] = 100
    var_constr_scale['corr_cov'][:] = 100
    var_constr_scale['kalman_gain'][:] = 100
    var_constr_scale['discretize_Q'][:] = 1e4
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['isRp_tril'][:] = 1e-2
    var_scale['sRp_tril'][:] = 1e2
    var_scale['sPp_tril'][:] = 1e2
    var_scale['sPc_tril'][:] = 1e2
    var_scale['sQ_tril'][:] = 1e2
    var_scale['sR_tril'][:] = 1e2
    var_scale['KsRp'][:] = 1e2
    var_scale['Qc'][:] = 1e2
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-9)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)    
    xopt = opt['x']
    Ac = opt['Ac']
    Bc = opt['Bc']
    Qc = opt['Qc']
    A = opt['A']
    B = opt['B']
    C = opt['C']
    D = opt['D']
    L = opt['L']
    KsRp = opt['KsRp']
    ybias = opt['ybias']
    pred_orth = opt['pred_orth']
    corr_orth = opt['corr_orth']
    sR = symfem.tril_mat(ny, opt['sR_tril'])
    sRp = symfem.tril_mat(ny, opt['sRp_tril'])
    isRp = symfem.tril_mat(ny, opt['isRp_tril'])
    sPp = symfem.tril_mat(nx, opt['sPp_tril'])
    sPc = symfem.tril_mat(nx, opt['sPc_tril'])
    sQ = symfem.tril_mat(nx, opt['sQ_tril'])
    
    yopt = model.output(xopt, u, C, D, ybias)
    e = y - yopt
    
    Pc = sPc @ sPc.T
    Pp = sPp @ sPp.T
    Rp = sRp @ sRp.T
    Q = sQ @ sQ.T
    R = sR @ sR.T

