import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from numpy import arange
from autodiff import *
import numpy as np
from fit import nonlinfit, linfit

try:
    import tkinter as Tk
except:
    import Tkinter as Tk

def cftool(x, y, yerr=None, num_points=1000, tol=1e-11, maxiter=50):
    """
    Opens the gui for fitting

    :param x: numpy array of data
    :param y: numpy array of measurements
    :param yerr: numpy array of the error on each measurement. Defaults to unit weights, if nothing is specified
    :param num_points: number of points to use when plotting the fitted function.
    """
    x_func = np.arange(min(x),max(x),(max(x)-min(x))/num_points)

    root = Tk.Tk()
    root.wm_title("CFTool")
    root.geometry('900x800')

    frame1 = Tk.Frame(root)
    frame1.pack()
    label_func = Tk.Label(frame1, text='Function', width=6)
    func_input = Tk.Entry(frame1)

    label_func.pack(side=Tk.LEFT, padx = 5, pady = 5)
    func_input.pack(side=Tk.LEFT, pady = 20)
    func_input.insert(0,'p[0]*x+p[1]')

    param_label = Tk.Label(frame1, text='Starting parameters')
    param_input = Tk.Entry(frame1)
    
    param_label.pack(side = Tk.LEFT, padx = 5, pady= 5)
    param_input.pack(side = Tk.LEFT)
    param_input.insert(0,'1,1')

    chi2_label = Tk.Label(frame1, text='chi2_red')
    chi2_text = Tk.Entry(frame1)
    chi2_label.pack(side = Tk.LEFT, padx = 5, pady= 5)
    chi2_text.pack(side = Tk.LEFT)
    

    def _fitlindata():
        """
        Internal function to c
        """
        fit = linfit(x,y,yerr, tol, maxiter) 
        #Update UI with the value of chi2
        chi2_text.delete("0",Tk.END)
        chi2_text.insert(Tk.INSERT,"%0.3f"%fit['chi2_red'])
        
        p = fit['p']
        p = np.asarray(p)
        a.cla()
        b.cla()
        c.cla()

        a.errorbar(x, y, yerr=yerr, fmt='.r', label='data')
        a.plot(x_func,p[0]*x_func+p[1], label='%0.4f*x+%0.4f'%tuple(p))
        a.set_title('Fit')
        a.set_xlabel('x')     
        a.legend()

        residuals =  (p[0]*x+p[1])-y
        b.plot(x, residuals ,'.')
        mean = np.mean(residuals)
        std = np.std(residuals)

        b.plot([0,max(x)],[mean, mean],'C1')
        b.plot([0,max(x)],[mean+std, mean+std],'r--')
        b.plot([0,max(x)],[mean-std, mean - std],'r--')
        b.set_title('Residuals')
        b.set_xlabel('x')


        hist = c.hist((p[0]*x+p[1])-y)
        c.plot([mean, mean],[0,max(hist[0])],'C1')
        c.plot([mean + std, mean + std],[0,max(hist[0])],'r--')
        c.plot([mean - std, mean - std],[0,max(hist[0])],'r--')
        c.set_title('Residual histogram')
        canvas.show()

    def fitdata():
        func = eval('lambda x,p: '+ func_input.get())
        param = [int(p) for p in param_input.get().replace('[','').replace(']','').split(',')]

        func_label = func_input.get()
        for i in range(len(param)):
            func_label = func_label.replace('p['+str(i)+']','%0.4f')

        fit = nonlinfit(func, x, y, param, yerr=yerr, tol=tol, maxiter=maxiter)

        #Update UI with the value of chi2
        chi2_text.delete("0",Tk.END)
        chi2_text.insert(Tk.INSERT,"%0.3f"%fit['chi2_red'])
        p = fit['p']

        
        a.cla()
        b.cla()
        c.cla()

        a.errorbar(x, y, yerr=yerr, fmt='.r', label='data')
        a.plot(x_func,func(x_func,p), label=func_label%tuple(p))
        a.set_title('Fit')
        a.set_xlabel('x')     
        a.legend()

        residuals =  func(x,p)-y
        mean = np.mean(residuals)
        std = np.std(residuals)
        b.plot(x,residuals,'.')
        b.plot([0,max(x)],[mean, mean],'C1')
        b.plot([0,max(x)],[std, std],'r--')
        b.plot([0,max(x)],[-std, - std],'r--')
        b.set_title('Residuals')
        b.set_xlabel('x')

        hist = c.hist(residuals)
        c.plot([mean, mean],[0,max(hist[0])],'C1')
        c.plot([std, std],[0,max(hist[0])],'r--')
        c.plot([-std, - std],[0,max(hist[0])],'r--')
        c.set_title('Residual histogram')
        canvas.show()

    but = Tk.Button(master=root, text="Fit data" ,width=10, command=fitdata)
    but.pack()

    
    
    grid = gridspec.GridSpec(2,2)

    fig = Figure(figsize=(5, 4), dpi=100)
    a = fig.add_subplot(grid[0,:])
    b = fig.add_subplot(grid[1,0])
    c = fig.add_subplot(grid[1,1])

    fig.set_tight_layout(True)

    a.set_title('Fit')
    a.set_xlabel('x')
    b.set_title('Residuals')
    b.set_xlabel('x')

    a.plot(x, y, '.r')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    toolbar = NavigationToolbar2TkAgg(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    _fitlindata()
    Tk.mainloop()
    
if __name__ == '__main__':
    def f(x,p):
        return p[0]*(x**2) + p[1]*x+p[2]*sin(x)
        # return p[0]*x+p[1]
    x = np.arange(0,100,1/1000)
    y = f(x,[0.1,0.5,100]) + np.random.normal(0,1,len(x))
    yerr = np.ones((len(x),1))*100
    cftool(x,y, yerr=yerr)