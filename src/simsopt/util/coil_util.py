import numpy as np
from math import pi

def sin_coeff(f, n, accuracy = 50, L=pi):
    '''
    Calculates the coefficients for the sine of order n in the Fourier expansion of function f.
    
    Args:
        f: function to fourier expand
        n: order of the coefficient
    ''' 
    a, b = 0, 2*L
    dx = (b - a) / accuracy
    integration = 0
    x=np.linspace(a, b, accuracy)
    
    integration=np.sum(f(x) * np.sin((n * pi * x) / L))
    integration *= dx
    
    return (1 / L) * integration

def cos_coeff(f, n, accuracy = 50, L=pi):
    '''
    Calculates the coefficients for the cosine of order n in the Fourier expansion of function f.
    
    Args:
        f: function to Fourier expand
        n: order of the coefficient
    ''' 
    a, b = 0, 2*L
    dx = (b - a) / accuracy
    integration = 0
    x=np.linspace(a, b, accuracy)

    integration=np.sum(f(x) * np.cos((n * pi * x) / L))
    integration *= dx

    if n==0: integration=integration/2
    return (1 / L) * integration

def importCurves(file):
    '''
    Reads a coils file and returns an array of shape (Ncoils,Npoints,3) in cartesian coordinates
    
    args:
        file: file to read.
    '''
    with open(file, 'r') as f:
        allCoilsValues = f.read().splitlines()[3:] 
    
    coilN=0
    coilPos=[[]]
    for nVals in range(len(allCoilsValues)):
        vals=allCoilsValues[nVals].split()
        try:
            floatVals = [float(nVals) for nVals in vals][0:3]
            coilPos[coilN].append(floatVals)
        except:
            try:
                floatVals = [float(nVals) for nVals in vals[0:3]][0:3]
                coilPos[coilN].append(floatVals)
                coilN=coilN+1
                coilPos.append([])
            except:
                break
    
    return coilPos[:-1]

def importCoils_and_current(file):
    with open(file, 'r') as f:
        allCoilsValues = f.read().splitlines()[3:] 
    
    coilN=0
    coilPos=[[]]
    current=[]
    flag = True
    for nVals in range(len(allCoilsValues)):
        vals=allCoilsValues[nVals].split()
        try:
            floatVals = [float(nVals) for nVals in vals][0:3]
            coilPos[coilN].append(floatVals)
            if flag:
                current.append(float(vals[3]))
                flag=False
        except:
            try:
                floatVals = [float(nVals) for nVals in vals[0:3]][0:3]
                coilPos[coilN].append(floatVals)
                coilN=coilN+1
                coilPos.append([])
                flag=True
            except:
                break
    return coilPos[:-1], current
def import_current(file):
    with open(file, 'r') as f:
        allCoilsValues = f.read().splitlines()[3:] 
        
    current=[]
    flag=True
    for nVals in range(len(allCoilsValues)-1):
        vals=allCoilsValues[nVals].split()
        if flag:
            current.append(float(vals[3]))
            flag=False
        if len(vals)>4:
            flag=True

    return current
def get_curves_fourier(curve,order,accuracy):
    '''
    Calculates the fourier coefficients of a curve given in cartesian coordinates up to a given order and with a given accuracy
    Args:
		curve: Matrix of shape (Npoints,3) that describes a curve in cartesian coordinates
		order: Maximum order coefficient to calculate
		accuracy: Increases the resolution with which the integration of the fourier coefficients is done
    '''
    from scipy import interpolate

    xArr=[i[0] for i in curve]
    yArr=[i[1] for i in curve]
    zArr=[i[2] for i in curve]

    L = [0 for i in range(len(xArr))]
    for itheta in range(1,len(xArr)): 
        dx = xArr[itheta]-xArr[itheta-1]
        dy = yArr[itheta]-yArr[itheta-1]
        dz = zArr[itheta]-zArr[itheta-1]
        dL = np.sqrt(dx*dx+dy*dy+dz*dz)
        L[itheta]=L[itheta-1]+dL

    L = np.array(L)*2*pi/L[-1] 
 
    xf  = interpolate.CubicSpline(L,xArr) #use the CubicSpline method with periodic bc instead? would require closing the line.
    yf  = interpolate.CubicSpline(L,yArr)
    zf  = interpolate.CubicSpline(L,zArr)
    
    order_interval = range(order+1)
    curvesFourierXS=[sin_coeff(xf,j,accuracy) for j in order_interval]
    curvesFourierXC=[cos_coeff(xf,j,accuracy) for j in order_interval]
    curvesFourierYS=[sin_coeff(yf,j,accuracy) for j in order_interval]
    curvesFourierYC=[cos_coeff(yf,j,accuracy) for j in order_interval]
    curvesFourierZS=[sin_coeff(zf,j,accuracy) for j in order_interval]
    curvesFourierZC=[cos_coeff(zf,j,accuracy) for j in order_interval]
 
    return np.concatenate([curvesFourierXS,curvesFourierXC,curvesFourierYS,curvesFourierYC,curvesFourierZS,curvesFourierZC])
