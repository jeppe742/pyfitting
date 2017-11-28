import numpy as np

def test_linfit():
    from pyfitting import linfit
    f = lambda x: 5*x + 27
    x = np.arange(0,100,1.0/1000)

    fit =  linfit(x, f(x))
    assert fit['p'][0] == 5.0000000000000036
    assert fit['p'][1] == 26.999999999999734

def test_nonlinfit():
    from pyfitting import nonlinfit
    from autodiff import sin

    f = lambda x, p: p[0]*sin(x)
    x = np.arange(0,100,1.0/1000)
    fit = nonlinfit(f, x, f(x,[10]),[10])

    assert fit['p'][0] == 10