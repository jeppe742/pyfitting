import numpy as np

def test_linfit():
    from pyfitting import linfit
    f = lambda x: 5*x + 27
    x = np.arange(0,100,1.0/1000)

    fit =  linfit(x, f(x))
    np.testing.assert_almost_equal(fit['p'][0],5.0)
    np.testing.assert_almost_equal(fit['p'][1], 27.0)

def test_nonlinfit():
    from pyfitting import nonlinfit
    from autodiff import sin

    f = lambda x, p: p[0]*sin(x)
    x = np.arange(0,100,1.0/1000)
    fit = nonlinfit(f, x, f(x,[10]),[10])

    np.testing.assert_almost_equal(fit['p'][0], 10)