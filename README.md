# Pyfitting - Linear and non-linear fitting in python

## Installation
![WINDOWS](https://ci.appveyor.com/api/projects/status/1soo9ttlod4ruy7r/branch/master?svg=true)  
To install pyfitting, simply run  
`pip install pyfitting`

## Interactive mode
The gui can be used in the following way
```python
from pyfitting import cftool

#Create some test data
f = lambda x, p: return p[0]*(x**2) + p[1]*x+p[2]*sin(x)
x = np.arange(0,100,1/1000)
y = f(x,[0.1,0.5,100]) + np.random.normal(0,1,len(x))
yerr = np.ones((len(x),1))*100

cftool(x,y,yerr=yerr)
```
If the error on y is not know, it can be left out

## Noninteractive mode
The module also exposes to functions. `linfit` for linear regression, and `nonlinfit` for non-linear regression. 
### Linear regression
`linfit` can be used simply like the following

```python
from pyfitting import linfit

#Generate some test data
f = lambda x: return 5*x+27
x = np.arange(0,100,1/1000)
y = f(x) + np.random.normal(0,1,len(x))
yerr = np.ones((len(x),1))*100

fit = linfit(x, y, yerr=yerr)
```
`linfit` returns a `dict` with three values. The parameters `p`, the reduced chi square `chi2_red`, and a function `f` which can be used to evaluate the fit 

### Non-linear regression
Non-linear regression works almost the same as linear. The only difference is, that `nonlinfit` two additional parameters. The function `f`, and starting values for the parameters `p`.  
The function can either be passed as a string, where the parameters are named `p[0]`,`p[1]` etx.
```python
from pyfitting import nonlinfit

#Generate some test data
f = lambda x: return 5*x+27
x = np.arange(0,100,1/1000)
y = f(x) + np.random.normal(0,1,len(x))
yerr = np.ones((len(x),1))*100

#starting value for the parameters
p_start = [1,1]

fit = nonlinfit('p[0]*x + p[1]', x, y, p_start, yerr=yerr)
```

It is also possible to pass a declared function.  
Please note, that automatic differentiation is used, so any non-linear functions should be imported from `autodiff`

```python
from pyfitting import nonlinfit
from autodiff import sin

#Generate some test data
f = lambda x, p : return p[0]*x + p[1]*sin(x)
x = np.arange(0,100,1/1000)
y = f(x,[5,27]) + np.random.normal(0,1,len(x))
yerr = np.ones((len(x),1))*100

#starting values for the parameter
p_start = [1, 1]
fit = nonlinfit(f, x, y, yerr=yerr)
```

## Additional parameters
`nonlinfit` has two additional parameters
- `tol` : describes how much each iteration should change before we assume convergence  
default value is 1e-11
- `maxiter` : which is the maximum times of iterations, before we stop the optimization.  
default value is 50
