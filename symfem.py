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
        
        v = self.variables
        v['x'] = [f'x{i}' for i in range(nx)]
        v['u'] = [f'u{i}' for i in range(nu)]
        v['y'] = [f'y{i}' for i in range(ny)]
        v['xn'] = [f'xn{i}' for i in range(nx)]
        v['xp'] = [f'xp{i}' for i in range(nx)]
        v['up'] = [f'up{i}' for i in range(nu)]
        v['yp'] = [f'yp{i}' for i in range(ny)]
        v['ybias'] = [f'ybias{i}' for i in range(ny)]
        v['A'] = [[f'A{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['B'] = [[f'B{i}_{j}' for j in range(nu)] for i in range(nx)]
        v['C'] = [[f'C{i}_{j}' for j in range(nx)] for i in range(ny)]
        v['D'] = [[f'D{i}_{j}' for j in range(nu)] for i in range(ny)]        
        v['K'] = [[f'K{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['S'] = [f'S{i}_{j}' for i,j in np.ndindex(nx, nx) if i<=j]
        v['iS'] = [f'iS{i}_{j}' for i,j in np.ndindex(nx, nx) if i<=j]
        
        self.decision.update({k for k in v if k != 'self'})

        # Register optimization functions
        self.add_constraint('defects')
        self.add_objective('L')
    
    def iS_mat(self, iS):
        """Build the iS matrix from its nonzero elements."""
        iS_mat = np.zeros((self.nx, self.nx), dtype=object)
        tril_ind = ((i,j) for (i,j) in np.ndindex(self.nx, self.nx) if i<=j)
        for ind, val in zip(tril_ind, iS):
            iS_mat[ind] = val
        return iS_mat
    
    def defects(self, xn, xp, up, yp, A, B, C, D, K, ybias):
        """Model dynamics defects."""
        ymodel = C @ xp + D @ up + ybias
        e = yp - ymodel
        return A @ xp + B @ up + K @ e - xn
    
    def L(self, y, x, u, C, D, iS, ybias):
        """Measurement log-likelihood."""
        iS_mat = self.iS_mat(iS)
        
        ymodel = C @ x + D @ u + ybias
        e = y - ymodel
        
        log_det_iS = sum(sympy.log(d) for d in iS_mat.diagonal())
        L = -0.5 * e.T @ iS_mat.T @ iS_mat @ e + log_det_iS
        return L
    
    @property
    def generate_assignments(self):
        gen = {'nx': self.nx, 'nu': self.nu, 'ny': self.ny,
               **getattr(super(), 'generate_assignments', {})}
        return gen

