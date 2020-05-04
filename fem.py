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
        self.add_decision('iS', model.niS)
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

