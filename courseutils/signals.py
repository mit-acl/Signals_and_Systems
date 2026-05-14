import numpy as np

def step(t):
    """Unit step function u(t)"""
    return (t >= 0).astype(float)

def rect(t, width=1):
    """Rectangle function centered at 0"""
    return ((-width/2 <= t) & (t <= width/2)).astype(float)

def tri(t, width=1):
    """Triangle function centered at 0"""
    y = np.zeros_like(t)
    mask1 = (-width <= t) & (t < 0)
    mask2 = (0 <= t) & (t <= width)
    y[mask1] = 1 + t[mask1]/width
    y[mask2] = 1 - t[mask2]/width
    return y