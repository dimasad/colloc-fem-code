"""Symbolic filter-error method estimation models."""


import numpy as np

import sympy
from ceacoest.modelling import symoptim


class InnovationDTModel(symoptim.Model):
    def __init__(self, nx, nu, ny):
        super().__init__()

        self.nx = nx
        """Number of states."""

        self.nu = nu
        """Number of inputs."""
        
        self.ny = ny
        """Number of outputs."""
        
        # Define decision variables
        v = self.variables
        v['x'] = [f'x{i}' for i in range(nx)]
        v['xn'] = [f'xn{i}' for i in range(nx)]
        v['xp'] = [f'xp{i}' for i in range(nx)]
        v['ybias'] = [f'ybias{i}' for i in range(ny)]
        v['A'] = [[f'A{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['B'] = [[f'B{i}_{j}' for j in range(nu)] for i in range(nx)]
        v['C'] = [[f'C{i}_{j}' for j in range(nx)] for i in range(ny)]
        v['D'] = [[f'D{i}_{j}' for j in range(nu)] for i in range(ny)]        
        v['K'] = [[f'K{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['iS'] = [f'iS{i}_{j}' for i,j in tril_ind(ny)]
        self.decision.update({k for k in v if k != 'self'})

        # Define auxiliary variables
        v['u'] = [f'u{i}' for i in range(nu)]
        v['y'] = [f'y{i}' for i in range(ny)]
        v['up'] = [f'up{i}' for i in range(nu)]
        v['yp'] = [f'yp{i}' for i in range(ny)]
        
        # Register optimization functions
        self.add_constraint('defects')
        self.add_objective('L')

        # Mark extra functions to generate code
        self.generate_functions.update({'output', 'propagate'})
        
    def defects(self, xn, xp, up, yp, A, B, C, D, K, ybias):
        """Model dynamics defects."""
        return xn - self.propagate(xp, up, yp, A, B, C, D, K, ybias)
    
    def L(self, y, x, u, C, D, iS, ybias):
        """Measurement log-likelihood."""
        iS = tril_mat(self.ny, iS)
        
        ymodel = self.output(x, u, C, D, ybias)
        e = y - ymodel
        
        log_det_iS = sum(sympy.log(d) for d in iS.diagonal())
        L = -0.5 * e.T @ iS.T @ iS @ e + log_det_iS
        return L
    
    def output(self, x, u, C, D, ybias):
        return C @ x + D @ u + ybias
    
    def propagate(self, x, u, y, A, B, C, D, K, ybias):
        e = y - self.output(x, u, C, D, ybias)
        return A @ x + B @ u + K @ e
    
    @property
    def generate_assignments(self):
        gen = {'nx': self.nx, 'nu': self.nu, 'ny': self.ny, 
               'niS': len(self.variables['iS']),
               'iS_diag': tril_diag(self.ny).tolist(),
               **getattr(super(), 'generate_assignments', {})}
        return gen


class NaturalDTModel(InnovationDTModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['Pp'] = [[f'Pp{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['Pc'] = [[f'Pp{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['Q'] = [[f'Q{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['R'] = [[f'R{i}_{j}' for j in range(ny)] for i in range(ny)]
        v['S'] = [f'S{i}_{j}' for i,j in tril_ind(ny)]
        self.decision.update({'Pp', 'Pc', 'Q', 'R', 'S'})

        # Register additional constraints
        self.add_constraint('Pp_symmetry')
        self.add_constraint('S_inverse')
        self.add_constraint('output_cov')
        self.add_constraint('x_pred_cov')
        self.add_constraint('x_corr_cov')
        self.add_constraint('kalman_gain')
    
    def Pp_symmetry(self, Pp):
        return symmetry_constraint(Pp)
    
    def Pc_symmetry(self, Pc):
        return symmetry_constraint(Pc)
    
    def S_inverse(self, iS, S):
        iS = tril_mat(self.ny, iS)
        S = tril_mat(self.ny, S)
        I = np.eye(self.ny)
        resid = S @ iS - I
        return [resid[i] for i in tril_ind(self.ny)]
    
    def output_cov(self, C, Pp, R, S):
        S = tril_mat(self.ny, S)
        return C @ Pp @ C.T + R - S @ S.T
    
    def x_pred_cov(self, A, Pp, Pc, Q):
        return A @ Pc @ A.T + Q - Pc
    
    def x_corr_cov(self, C, K, Pp, Pc):
        return Pp - K @ C @ Pp - Pc
    
    def kalman_gain(self, C, K,  Pp, S):
        S = tril_mat(self.ny, S)        
        return Pp @ C - K @ S @ S.T


def tril_ind(n):
    yield from ((i,j) for (i,j) in np.ndindex(n, n) if i<=j)


def tril_diag(n):
    return np.array([i==j for (i,j) in tril_ind(n)])


def tril_mat(n, elem):
    mat = np.zeros((n, n), dtype=elem.dtype)
    for ind, val in zip(tril_ind(n), elem):
        mat[ind] = val
    return mat


def symmetry_constraint(M):
    n = len(M)
    return [M[i,j] - M[j,i] for (i,j) in np.ndindex(n, n) if i < j]
