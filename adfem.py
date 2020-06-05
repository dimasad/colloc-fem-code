"""Automatic differentiation filter-error method estimation models."""


import functools
import inspect
import itertools

import jax
from jax import numpy as jnp
from jax import scipy as jscipy

import numpy as onp


### Enable 64-bit in jax ###
jax.config.update("jax_enable_x64", True)


class ADFunction:
    """Helper for optimization function automatic differentiation."""
    
    def __init__(self, fun):
        self.fun = fun
        """Underlying function."""
        
        self.hess = None
        """Hessian elements."""
    
    @property
    def sig(self):
        return inspect.signature(self.fun)
    
    @property
    def args(self):
        return self.sig.parameters.keys()

    def argnum(self, argname):
        return list(self.args).index(argname)

    def deriv(self, wrt):
        if isinstance(wrt, str):
            wrt_tuple = wrt,
            return self.deriv(wrt_tuple)

        deriv = self.fun
        for argname in wrt:
            argnum = self.argnum(argname)
            deriv = jax.jacobian(deriv, argnum)
        return deriv

class ADConstraint(ADFunction):
    """Helper for constraint function automatic differentiation."""
    
    def __init__(self, fun, core_shape=None):
        super().__init__(fun)
        
        self.core_shape = core_shape
        """The shape of the output of the elementary function, if vectorized."""
    

class ADObjective(ADFunction):
    """Helper for constraint function automatic differentiation."""


def constraint(core_shape_or_fun=None):
    if callable(core_shape_or_fun):
        fun = core_shape_or_fun
        return constraint()(fun)
    else:
        core_shape = core_shape_or_fun
        def decorator(fun):
            return ADConstraint(fun, core_shape)
        return decorator


def objective(fun):
    return ADObjective(fun)


def hessian(*args):
    def decorator(obj):
        obj.hessian = args
        return obj
    return decorator


class ADModel:
    def __init_subclass__(cls):
        base_dec = getattr(super(), 'decision', set())
        cls_dec = getattr(cls, 'decision', set())  
        cls.decision = set.union(base_dec, cls_dec)
        """The decision variables of this model."""
        
        base_vec = getattr(super(), 'vectorized', {})
        cls_vec = getattr(cls, 'vectorized', {})  
        cls.vectorized = {**base_vec, **cls_vec}
        """The core shape of vectorized variables."""
        
        cls_items = cls.__dict__.items()
        adfun = {k:v for k,v in cls_items if isinstance(v, ADFunction)}
        for name, spec in adfun.items():
            cls.setup_ad_function(name, spec)
            
    @classmethod
    def setup_ad_function(cls, name, spec):
        wrt = cls.decision.intersection(spec.args)
        
        for argname in wrt:
            derivname = cls.first_derivative_name(name, argname)
            deriv = spec.deriv(argname)
            setattr(cls, derivname, deriv)
    
        hess = spec.hess
        if hess is None:
            hess = itertools.combinations_with_replacement(wrt, 2)
        for pair in hess:
            derivname = cls.second_derivative_name(name, pair)
            deriv = spec.deriv(pair)
            setattr(cls, derivname, deriv)
        
        setattr(cls, name, spec.fun)
        
        # Vectorize
        # Create ceacoest.optim wrapper
    
    @staticmethod
    def first_derivative_name(fname, wrtname):
        """Generator of default name of first derivatives."""
        assert isinstance(fname, str)
        assert isinstance(wrtname, str)
        return f'd{fname}_d{wrtname}'
    
    @staticmethod
    def second_derivative_name(fname, wrt):
        """Generator of default name of second derivatives."""
        if not isinstance(wrt, tuple) or len(wrt) != 2:
            raise ValueError("wrt must be a two-element tuple")
        
        if wrt[0] == wrt[1]:
            return f'd2{fname}_d{wrt[0]}2'
        else:
            return f'd2{fname}_d{wrt[0]}_d{wrt[1]}'


class InnovationDTModel(ADModel):
    
    decision = {
        'x', 'e', 'xnext', 'xprev', 'eprev', 'ybias', 
        'A', 'B', 'C', 'D', 'L', 'isRp_tril'
    }
    """Decision variables of the optimization problem."""
    
    vectorized = dict(
        x='nx', e='ny', xnext='nx', xprev='nx', eprev='ny',
        u='nu', y='ny', uprev='nu'
    )
    """Decision variables of the optimization problem."""
    
    def __init__(self, nx, nu, ny):
        self.nx = nx
        """Number of states."""
        
        self.nu = nu
        """Number of inputs."""
        
        self.ny = ny
        """Number of outputs."""
    
    
    @hessian(('xprev', 'A'), ('eprev', 'L'))
    @constraint('nx')
    def defects(self, xnext, xprev, uprev, eprev, A, B, L):
        """Model dynamics defects."""
        xpred = A @ xprev + B @ uprev + L @ eprev
        return xnext - xpred
    
    @hessian(('x', 'C'), ('x', 'isRp_tril'), ('C', 'isRp_tril'), 
             ('D', 'isRp_tril'), ('ybias', 'isRp_tril'))
    @constraint('ny')
    def innovation(self, y, e, x, u, C, D, ybias, isRp_tril):
        """Model dynamics defects."""
        ymodel = C @ x + D @ u + ybias
        isRp = tril_mat(self.ny, isRp_tril)
        return isRp @ (y - ymodel) - e
    
    @hessian(('e', 'e'), ('isRp_tril', 'isRp_tril'))
    @objective
    def L(self, e, isRp_tril):
        """Measurement log-likelihood."""
        isRp = tril_mat(self.ny, isRp_tril)
        log_det_isRp = jnp.log(isRp.diagonal()).sum()
        return -0.5 * (e ** 2).sum() + log_det_isRp


def tril_mat(n, tril_elem):
    """Build a matrix from its lower-triangular elements."""
    tril_ind = onp.tril_indices(n)
    M = jnp.zeros((n, n))
    return M.at[tril_ind].set(tril_elem)

