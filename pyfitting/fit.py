import numpy as np
from autodiff import jacobian, sin
def linfit(x, y, yerr=None):
    """
    Solves the linear least squares. That is solves the linear system

    A*p=y where A = [x 1]

    Using the QR-decomposition A=QR, the optimal b can be estimated as
    p = R^(-1)*Q^T*y
    """


    N = len(x)
    #If no errors specified, just set them all to 1
    if yerr is None:
        yerr = np.ones((N,1), dtype=np.float64)
    yerr = yerr.reshape(N,1)


    x = x.reshape(N,1)
    #W = np.matrix(np.diag(1/(yerr.flatten())))

    A = np.matrix(np.hstack((x,np.ones((N,1)))))
    #We want to calculate W*A, but W is just a diagonal matrix. If we just elementwise multiply the diagonal and A, this will be much faster
    #Q, R = np.linalg.qr(W*A)
    Q, R = np.linalg.qr(np.multiply(yerr,A))

    #Create these as matricies to make the multiplication prettier
    Q, R, y = np.matrix(Q), np.matrix(R), y.reshape(N,1)

    #Again, W*y is horribly inefficient. Just multiply yerr and y elementwise. Since they both are ndarrays, we can just * them to do this
    #p = R**(-1)*Q.T*W*y
    p = (R**(-1)*Q.T)*(yerr*y)


    chi2 = float(sum(np.power((y-x*p[0]+p[1]),2)/(yerr**2)))
    dof = N-1
    chi2_red = chi2/dof

    return {'p':p, 'chi2_red':chi2_red, 'f': lambda x: f(x, p)}

def nonlinfit(f, x, y, p, yerr=None, tol=1e-11, maxiter=50):
    """
    Finds the best parameters using least squares

    If J is the jacobian, W the weight, and Q*R=WJ the qr-decomposition, then the update vector p' is
    p' = R^-1 * Q^T* (y-f(x,p)).
    
    This is then used to update the estimate for the parameter p_i+1 = p_i + p'
    """

    if isinstance(f, str):
        f = eval('lambda x,p: '+ f)

    N = len(x)
    num_param = len(p)

    #Convert all input to numpy arrays of specific type and shape
    y = np.asarray(y, dtype=np.float64).reshape(N, 1)
    x = np.asarray(x, dtype=np.float64).reshape(N, 1)
    p = np.asarray(p, dtype=np.float64).reshape(num_param,1)
    

    #If no errors specified, just set them all to 1
    if yerr is None:
        yerr = np.ones((N,1), dtype=np.float64)
    yerr = yerr.reshape(N,1)


    chi2 = sum(np.power((y-f(x,p)),2)/(yerr**2))
    dof = N-num_param
    chi2_red = chi2/dof

    for _ in range(maxiter):
        delta_y = y-f(x,p)
        J = jacobian(f, x, p)
        
        #We want to calculate W*J, but W is just a diagonal matrix. If we just elementwise multiply the diagonal and J, this will be much faster
        #Q, R = np.linalg.qr(W*J)
        Q, R = np.linalg.qr(np.multiply(yerr,J))

        #Create these as matricies to make multiplication prettier
        Q, R, J = np.matrix(Q), np.matrix(R), np.matrix(J)
        
        #Again, W*delta_y is horribly inefficient. Just multiply yerr and delta_y elementwise. Since they both are ndarrays, we can just * them to do this
        #p = p + R**(-1)*Q.T*W*delta_y
        p = p + (R**(-1)*Q.T) * (yerr*delta_y)

        #p need to be in the right format
        p=np.asarray(p)
        
        #Check for convergence
        chi2_tmp = float(sum(np.power((y-f(x,p)),2)/(yerr**2)))
        chi2_red_tmp = chi2_tmp/(N-num_param)
        diff = abs(chi2_red-chi2_red_tmp)
        print("reduced chi2 = %0.4E   diff = %0.4E"%(chi2_red_tmp, diff))
        if diff<tol:
            print("Converged!")
            return {'p':p, 'chi2_red':chi2_red_tmp, 'f': lambda x: f(x, p)}
        
        #Update chi2 value, and print progress
        chi2_red = chi2_red_tmp


    print("Hit max number of iterations without converging!!")
    return {'p':p, 'chi2_red':chi2_red_tmp, 'f': lambda x: f(x, p)}

if __name__ == "__main__":
    def f(x,p):
        return p[0]*(x**2) + p[1]*x+p[2]*sin(x)

    x = np.arange(0,100,1/10)
    y = f(x,[3,5,1000]) + np.random.normal(0,1,len(x))

    pp=nonlinfit(f,x,y,[2,7,1000])

