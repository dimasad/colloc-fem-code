"""Symbolic filter-error method estimation models."""


import numpy as np

import scipy.special
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
        v['yp'] = [f'yp{i}' for i in range(ny)]
        v['en'] = [f'en{i}' for i in range(ny)]
        v['xnext'] = [f'xnext{i}' for i in range(nx)]
        v['xprev'] = [f'xprev{i}' for i in range(nx)]
        v['enprev'] = [f'enprev{i}' for i in range(ny)]
        v['ybias'] = [f'ybias{i}' for i in range(ny)]
        v['A'] = [[f'A{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['B'] = [[f'B{i}_{j}' for j in range(nu)] for i in range(nx)]
        v['C'] = [[f'C{i}_{j}' for j in range(nx)] for i in range(ny)]
        v['D'] = [[f'D{i}_{j}' for j in range(nu)] for i in range(ny)]
        v['Ln'] = [[f'Ln{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['isRp_tril'] = [f'isRp{i}_{j}' for i,j in tril_ind(ny)]
        self.decision.update({k for k in v if k != 'self'})
        
        # Define auxiliary variables
        v['u'] = [f'u{i}' for i in range(nu)]
        v['y'] = [f'y{i}' for i in range(ny)]
        v['uprev'] = [f'uprev{i}' for i in range(nu)]
        
        # Register optimization functions
        self.add_constraint('dynamics')
        self.add_constraint('output')
        self.add_constraint('innovation')
        self.add_objective('loglikelihood')
    
    def dynamics(self, xnext, xprev, uprev, enprev, A, B, Ln):
        """Model dynamics defects."""
        xpred = A @ xprev + B @ uprev + Ln @ enprev
        return xnext - xpred

    def output(self, yp, x, u, C, D, ybias):
        """Predicted output constraint."""
        return C @ x + D @ u + ybias - yp
    
    def innovation(self, y, yp, en, isRp_tril):
        """Model normalized innovation constraint."""
        isRp = tril_mat(isRp_tril)
        return isRp @ (y - yp) - en
        
    def loglikelihood(self, en, isRp_tril):
        """Log-likelihood function."""
        isRp = tril_mat(isRp_tril)
        log_det_isRp = sum(sympy.log(d) for d in isRp.diagonal())
        return -0.5 * (en ** 2).sum() + log_det_isRp
    
    @property
    def generate_assignments(self):
        gen = {'nx': self.nx, 'nu': self.nu, 'ny': self.ny, 
               'nty': len(self.variables['sRp_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


class StableDTModel(InnovationDTModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['Apred'] = [[f'Apred{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['sM_tril'] = [f'sM{i}_{j}' for i,j in tril_ind(nx)]
        so = [[f'stab_orth{i}_{j}' for j in range(2*nx)] for i in range(nx)]
        v['stab_orth'] = so
        self.decision.update({'Apred', 'sM_tril', 'stab_orth'})

        # Register additional constraints
        self.add_constraint('predictor_stability')
        self.add_constraint('predictor_A')
        self.add_constraint('stab_orthogonality')
    
    def predictor_stability(self, sM_tril, Apred, stab_orth):
        sM = tril_mat(sM_tril)
        I = np.eye(self.nx)
        bmat = np.block([Apred @ sM, I])
        return sM @ stab_orth - bmat
    
    def stab_orthogonality(self, stab_orth):
        resid = 0.5 * (stab_orth @ stab_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]
    
    def predictor_A(self, A, Ln, isRp_tril, C, Apred):
        isRp = tril_mat(isRp_tril)
        return A - Ln @ isRp @ C - Apred


class BalancedDTModel(InnovationDTModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['sW_diag'] = [f'sW{i}' for i in range(nx)]
        co = [[f'ctrl_orth{i}_{j}' for j in range(nx+nu)] for i in range(nx)]
        oo = [[f'obs_orth{i}_{j}' for j in range(nx+ny)] for i in range(nx)]
        v['ctrl_orth'] = co
        v['obs_orth'] = oo
        self.decision.update({'sW_diag', 'ctrl_orth', 'obs_orth'})

        # Register additional constraints
        self.add_constraint('ctrl_gram')
        self.add_constraint('obs_gram')
        self.add_constraint('ctrl_orthogonality')
        self.add_constraint('obs_orthogonality')
    
    def ctrl_gram(self, sW_diag, A, B, ctrl_orth):
        bmat = np.block([A * sW_diag, B])
        return sW_diag[:, None] * ctrl_orth - bmat
    
    def obs_gram(self, sW_diag, A, C, obs_orth):
        bmat = np.block([A.T * sW_diag, C.T])
        return sW_diag[:, None] * obs_orth - bmat
    
    def ctrl_orthogonality(self, ctrl_orth):
        resid = 0.5 * (ctrl_orth @ ctrl_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]
    
    def obs_orthogonality(self, obs_orth):
        resid = 0.5 * (obs_orth @ obs_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]


class MaximumLikelihoodDTModel(InnovationDTModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['Kn'] = [[f'Kn{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['sRp_tril'] = [f'sRp{i}_{j}' for i,j in tril_ind(ny)]
        v['sQ_tril'] = [f'sQ{i}_{j}' for i,j in tril_ind(nx)]
        v['sR_tril'] = [f'sR{i}_{j}' for i,j in tril_ind(ny)]
        v['sPp_tril'] = [f'sPp{i}_{j}' for i,j in tril_ind(nx)]
        v['sPc_tril'] = [f'sPc{i}_{j}' for i,j in tril_ind(nx)]
        po = [[f'pred_orth{i}_{j}' for j in range(2*nx)] for i in range(nx)]
        co = [[f'corr_orth{i}_{j}' for j in range(nx+ny)] for i in range(nx+ny)]
        v['pred_orth'] = po
        v['corr_orth'] = co
        self.decision.update({'sPp_tril', 'sPc_tril', 'sQ_tril'})
        self.decision.update({'sRp_tril', 'sR_tril', 'Kn'})
        self.decision.update({'pred_orth', 'corr_orth'})
        
        # Register additional constraints
        self.add_constraint('pred_orthogonality')
        self.add_constraint('corr_orthogonality')
        self.add_constraint('pred_cov')
        self.add_constraint('corr_cov')
        self.add_constraint('kalman_gain')
        self.add_constraint('sRp_inv')
    
    def pred_orthogonality(self, pred_orth):
        resid = 0.5 * (pred_orth @ pred_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]
    
    def corr_orthogonality(self, corr_orth):
        resid = 0.5 * (corr_orth @ corr_orth.T - np.eye(self.nx + self.ny))
        return [resid[i] for i in tril_ind(self.nx + self.ny)]
    
    def pred_cov(self, A, sPp_tril, sPc_tril, sQ_tril, pred_orth):
        sPp = tril_mat(sPp_tril)
        sPc = tril_mat(sPc_tril)
        sQ = tril_mat(sQ_tril)
        bmat = np.block([[A @ sPc, sQ]])
        return sPp @ pred_orth - bmat
    
    def corr_cov(self, C, sR_tril, sRp_tril, sPp_tril, sPc_tril, Kn, 
                 corr_orth):
        sPp = tril_mat(sPp_tril)
        sPc = tril_mat(sPc_tril)
        sRp = tril_mat(sRp_tril)
        sR = tril_mat(sR_tril)
        
        zeros = np.zeros((self.nx, self.ny))
        M1 = np.block([[sRp,  zeros.T], 
                       [Kn, sPc]])
        M2 = np.block([[sR,    C @ sPp], 
                       [zeros, sPp]])
        return M1 @ corr_orth - M2
    
    def kalman_gain(self, Ln, Kn, A):
        return Ln - A @ Kn
        
    def sRp_inv(self, sRp_tril, isRp_tril):
        """Model normalized innovation constraint."""
        sRp = tril_mat(sRp_tril)
        isRp = tril_mat(isRp_tril)
        I = np.eye(self.ny)
        resid = sRp @ isRp - I
        return [resid[i] for i in tril_ind(self.ny)]
    
    @property
    def generate_assignments(self):
        gen = {'ntx': len(self.variables['sPp_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


class ZOHDynamicsModel(InnovationDTModel):
    
    expm_order = 3
    
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['dt'] = 'dt'
        v['Ac'] = [[f'Ac{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['Bc'] = [[f'Bc{i}_{j}' for j in range(nu)] for i in range(nx)]
        self.decision.update({'Ac', 'Bc'})
        
        # Register additional constraints
        self.add_constraint('discretize_AB')
    
    def discretize_AB(self, A, B, Ac, Bc, dt):
        A_resid = A - expm_taylor(Ac*dt, self.expm_order)
        
        term = dt * Bc
        B_sum = term
        for k in range(2, self.expm_order + 1):
            term = dt / k * Ac @ term
            B_sum = B_sum + term
        B_resid = B - B_sum 
        return np.c_[A_resid, B_resid]


class DiscretizedNoiseModel(MaximumLikelihoodDTModel):
    
    noise_disc_order = 1
    
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['sQc_tril'] = [f'sQc{i}_{j}' for i,j in tril_ind(nx)]
        if 'Ac' not in v:
            v['Ac'] = [[f'Ac{i}_{j}' for j in range(nx)] for i in range(nx)]
        if 'dt' not in v:
            v['dt'] = 'dt'
        self.decision.update({'Ac', 'sQc_tril'})
        
        # Register additional constraints
        self.add_constraint('discretize_Q')
    
    def discretize_Q(self, Ac, sQ_tril, sQc_tril, dt):
        fact = scipy.special.factorial
        
        sQ = tril_mat(sQ_tril)
        sQc = tril_mat(sQc_tril)
        Q = sQ @ sQ.T
        Qc = sQc @ sQc.T
        
        Q_sum = np.zeros_like(sQ)
        for j in range(self.noise_disc_order + 1):
            for k in range(self.noise_disc_order + 1):
                if j < k:
                    Acj = np.linalg.matrix_power(Ac, j)
                    Ack = np.linalg.matrix_power(Ac, k)
                    scal = dt **(j+k+1) / (j+k+1) / fact(j) / fact(k)
                    term = scal * Acj @ Qc @ Ack.T
                    Q_sum += term + term.T
                elif j == k:
                    Acj = np.linalg.matrix_power(Ac, j)
                    scal = dt **(j+k+1) / (j+k+1) / fact(j) ** 2
                    term = scal * Acj @ Qc @ Acj.T
                    Q_sum += term
                    
        resid = Q_sum - Q
        return [resid[i] for i in tril_ind(self.nx)]


def tril_ind(n):
    yield from ((i,j) for (i,j) in np.ndindex(n, n) if i>=j)


def tril_diag(n):
    return np.array([i==j for (i,j) in tril_ind(n)])


def tril_mat(elem):
    ntril = len(elem)
    n = int(round(0.5*(np.sqrt(8*ntril + 1) - 1)))
    mat = np.zeros((n, n), dtype=elem.dtype)
    for ind, val in zip(tril_ind(n), elem):
        mat[ind] = val
    return mat


def symmetry_constraint(M):
    n = len(M)
    return [M[i,j] - M[j,i] for (i,j) in np.ndindex(n, n) if i < j]


def psd_constraint(mat, sqrt_tril_elem):
    sqrt = tril_mat(len(mat), sqrt_tril_elem)
    return mat - sqrt @ sqrt.T


def expm_taylor(a, order):
    """Taylor approximation for matrix exponential."""
    a = np.asarray(a)
    n = len(a)
    assert a.shape == (n, n)
    
    term = np.eye(n)
    expm = term
    for i in range(1, order + 1):
        term = 1 / i * term @ a
        expm = expm + term
    return expm


class NormInnovationModel(symoptim.Model):
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
        v['yp'] = [f'yp{i}' for i in range(ny)]
        v['en'] = [f'en{i}' for i in range(ny)]
        v['xnext'] = [f'xnext{i}' for i in range(nx)]
        v['xprev'] = [f'xprev{i}' for i in range(nx)]
        v['enprev'] = [f'enprev{i}' for i in range(ny)]
        v['ybias'] = [f'ybias{i}' for i in range(ny)]
        v['A'] = [[f'A{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['B'] = [[f'B{i}_{j}' for j in range(nu)] for i in range(nx)]
        v['C'] = [[f'C{i}_{j}' for j in range(nx)] for i in range(ny)]
        v['D'] = [[f'D{i}_{j}' for j in range(nu)] for i in range(ny)]
        v['Ln'] = [[f'Ln{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['sRp_tril'] = [f'sRp{i}_{j}' for i,j in tril_ind(ny)]
        v['isRp_tril'] = [f'isRp{i}_{j}' for i,j in tril_ind(ny)]
        self.decision.update({k for k in v if k != 'self'})
        
        # Define auxiliary variables
        v['u'] = [f'u{i}' for i in range(nu)]
        v['y'] = [f'y{i}' for i in range(ny)]
        v['uprev'] = [f'uprev{i}' for i in range(nu)]
        
        # Register optimization functions
        self.add_constraint('dynamics')
        self.add_constraint('output')
        self.add_constraint('innovation')
        self.add_constraint('sRp_inv')
        self.add_objective('loglikelihood')
    
    def dynamics(self, xnext, xprev, uprev, enprev, A, B, Ln):
        """Model dynamics defects."""
        xpred = A @ xprev + B @ uprev + Ln @ enprev
        return xnext - xpred

    def output(self, yp, x, u, C, D, ybias):
        """Predicted output constraint."""
        return C @ x + D @ u + ybias - yp
    
    def innovation(self, y, yp, en, isRp_tril):
        """Model normalized innovation constraint."""
        isRp = tril_mat(isRp_tril)
        return isRp @ (y - yp) - en
        
    def sRp_inv(self, sRp_tril, isRp_tril):
        """Model normalized innovation constraint."""
        sRp = tril_mat(sRp_tril)
        isRp = tril_mat(isRp_tril)
        I = np.eye(self.ny)
        resid = sRp @ isRp - I
        return [resid[i] for i in tril_ind(self.ny)]
    
    def loglikelihood(self, en, isRp_tril):
        """Log-likelihood function."""
        isRp = tril_mat(isRp_tril)
        log_det_isRp = sum(sympy.log(d) for d in isRp.diagonal())
        return -0.5 * (en ** 2).sum() + log_det_isRp
    
    @property
    def generate_assignments(self):
        gen = {'nx': self.nx, 'nu': self.nu, 'ny': self.ny, 
               'nty': len(self.variables['sRp_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen
