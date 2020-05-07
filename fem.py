"""Filter-error method estimation module."""


import numpy as np

from ceacoest import optim


class InnovationDTProblem(optim.Problem):

    def __init__(self, model, y, u):
        super().__init__()

        self.model = model
        """Underlying model."""
        
        self.y = np.asarray(y)
        """Measurements."""
        
        self.yp = self.y[:-1]
        """Previous measurements."""        
        
        self.u = np.asarray(u)
        """Inputs."""

        self.up = self.u[:-1]
        """Previous inputs."""        
        
        N = len(y)
        self.N = N
        """Number of measurement instants."""
        
        assert y.ndim == 2
        assert y.shape[1] == model.ny
        assert u.shape == (N, model.nu)
        assert N > 1
        
        # Register decision variables
        self.add_decision('ybias', model.ny)
        self.add_decision('isRp_tril', model.n_tril_y)
        self.add_decision('A', (model.nx, model.nx))
        self.add_decision('B', (model.nx, model.nu))
        self.add_decision('C', (model.ny, model.nx))
        self.add_decision('D', (model.ny, model.nu))
        self.add_decision('K', (model.nx, model.ny))
        x = self.add_decision('x', (N, model.nx))
        
        # Define and register dependent variables
        xp = optim.Decision((N-1, model.nx), x.offset)
        xn = optim.Decision((N-1, model.nx), x.offset + model.nx)
        self.add_dependent_variable('xp', xp)
        self.add_dependent_variable('xn', xn)
    
        # Register problem functions
        self.add_objective(model.L, N)
        self.add_constraint(model.defects, (N - 1, model.nx))
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'y': self.y, 'yp': self.yp, 'u': self.u, 'up': self.up,
                **super().variables(dvec)}



class NaturalDTProblem(InnovationDTProblem):

    def __init__(self, model, y, u):
        super().__init__(model, y, u)
        
        # Register decision variables
        self.add_decision('Pp', (model.nx, model.nx))
        self.add_decision('Pc', (model.nx, model.nx))
        self.add_decision('Q', (model.nx, model.nx))
        self.add_decision('R', (model.ny, model.ny))
        self.add_decision('sRp_tril', model.n_tril_y)
        self.add_decision('sPp_tril', model.n_tril_x)
        self.add_decision('sPc_tril', model.n_tril_x)
        self.add_decision('sQ_tril', model.n_tril_x)
        self.add_decision('sR_tril', model.n_tril_y)
        
        # Register constraint functions
        self.add_constraint(model.Rp_inverse, model.n_tril_y)
        self.add_constraint(model.output_cov, model.n_tril_y)
        self.add_constraint(model.x_pred_cov, model.n_tril_x)
        self.add_constraint(model.x_corr_cov, model.n_tril_x)
        self.add_constraint(model.kalman_gain, (model.nx, model.ny))
        self.add_constraint(model.Pp_psd, (model.nx, model.nx))
        self.add_constraint(model.Pc_psd, (model.nx, model.nx))
        self.add_constraint(model.Q_psd, (model.nx, model.nx))
        self.add_constraint(model.R_psd, (model.ny, model.ny))


class NaturalSqrtDTProblem(InnovationDTProblem):

    def __init__(self, model, y, u):
        super().__init__(model, y, u)
        
        nxy = model.nx + model.ny
        n_tril_xy = nxy * (nxy + 1) // 2
        
        # Register decision variables
        self.add_decision('sRp_tril', model.n_tril_y)
        self.add_decision('sPp_tril', model.n_tril_x)
        self.add_decision('sPc_tril', model.n_tril_x)
        self.add_decision('sQ_tril', model.n_tril_x)
        self.add_decision('sR_tril', model.n_tril_y)
        self.add_decision('pred_orth', (2*model.nx, model.nx))
        self.add_decision('corr_orth', (nxy, nxy))
        
        # Register constraint functions
        self.add_constraint(model.pred_orthogonality, model.n_tril_x)
        self.add_constraint(model.corr_orthogonality, n_tril_xy)
        self.add_constraint(model.pred_cov, (2*model.nx, model.nx))
        self.add_constraint(model.corr_cov, (nxy, nxy))
        self.add_constraint(model.Rp_inverse, model.n_tril_y)


class NaturalSqrtZOHProblem(NaturalSqrtDTProblem):
    def __init__(self, model, y, u):
        super().__init__(model, y, u)
        
        nx = model.nx
        nu = model.nu
        
        # Register decision variables
        self.add_decision('Ac', (nx, nx))
        self.add_decision('Bc', (nx, nu))
        self.add_decision('Qc', (nx, nx))

        # Register constraint functions
        self.add_constraint(model.discretize_AB, (nx, nx + nu))
        self.add_constraint(model.discretize_Q, (nx, nx))
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'dt': self.model.dt, **super().variables(dvec)}
