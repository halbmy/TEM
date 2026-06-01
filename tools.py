import numpy as np

mu0 = 4e-7 * np.pi


def rhoa(t, dbzdt, m=1.0):
    return mu0 / np.pi * (mu0 * m / 20.0)**(2/3) * np.abs(dbzdt)**(-2/3) * t**(-5/3)

# dr = dr/du * du = -2/3 r / u *du => |dr/r| = 2/3 |dr/r|

def skinDepthTEM(t, rho):
    """Return diffusion (skin) depth of TEM."""
    return np.sqrt(2 * t * rho / mu0)

