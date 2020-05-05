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
    d2r = np.pi / 180
    module_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(module_dir, 'data', 'fAttasElv1.mat')
    data = scipy.io.loadmat(data_file_path)['fAttasElv1'][30:-30]
    t = data[:, 0] - data[0, 0]
    u = data[:, [21]] * d2r
    y = data[:, [7, 12]] * d2r
    return t, u, y


if __name__ == '__main__':
    # Load experiment data
    t, u, y = load_data()
    symmodel = symfem.NaturalDTModel(nx=2, nu=1, ny=2)
    model = symmodel.compile_class()()
    problem = fem.NaturalDTProblem(model, y, u)
    
    Aoem = np.array([[0.92092623, -0.18707263], [0.04193602, 0.97679607]])
    Boem = np.array([[-0.27215983], [0.01984017]])
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = Aoem
    var0['B'][:] = Boem
    var0['C'][:] = np.eye(2)
    var0['D'][:] = np.zeros((2,1))
    var0['x'][:] = y - y[0]
    var0['ybias'][:] = y[0]
    var0['iS'][symfem.tril_diag(2)] = 1e3
    var0['S'][symfem.tril_diag(2)] = 1e-3
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['S'][symfem.tril_diag(2)] = 1e-7
    var_L['C'][:] = np.eye(2)
    var_U['C'][:] = np.eye(2)
    var_L['D'][:] = np.zeros((2,1))
    var_U['D'][:] = np.zeros((2,1))
    var_L['Pp_sqrt'][symfem.tril_diag(2)] = 0
    var_L['Pc_sqrt'][symfem.tril_diag(2)] = 0
    var_L['Q_sqrt'][symfem.tril_diag(2)] = 0
    var_L['R_sqrt'][symfem.tril_diag(2)] = 0
    
    # Define bounds for constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    # Define problem scaling
    dec_scale = np.ones(problem.ndec)
    constr_scale = np.ones(problem.ncons)
    obj_scale = -1.0

    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)    
    xopt = opt['x']
    A = opt['A']
    B = opt['B']
    C = opt['C']
    D = opt['D']
    Pp = opt['Pp']
    Pc = opt['Pc']
    Q = opt['Q']
    R = opt['R']
    K = opt['K']
    ybias = opt['ybias']
    S = symfem.tril_mat(model.ny, opt['S'])
    iS = symfem.tril_mat(model.ny, opt['iS'])
    Pp_sqrt = symfem.tril_mat(model.nx, opt['Pp_sqrt'])
    Pc_sqrt = symfem.tril_mat(model.nx, opt['Pc_sqrt'])
    Q_sqrt = symfem.tril_mat(model.nx, opt['Q_sqrt'])
    R_sqrt = symfem.tril_mat(model.nx, opt['R_sqrt'])
    yopt = model.output(xopt, u, C, D, ybias)
    e = y - yopt

