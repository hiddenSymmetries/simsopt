import numpy as np
import numbers

class finiteDifference:
    def __init__(self, method, epsilon, function, args):
        self.method = method
        self.epsilon = epsilon
        self.function = function
        self.args = args
    
    def evaluate(self,x):
        """
        Evaluates function at point x
        
        Args:
            x (float, list, or array): point for evaluation of function
            
        Returns:
            function_value (float, list, or array): value of function at point
            
        """
        if (self.args is not None):
            function_value = self.function(x, *self.args)
        else:
            function_value = self.function(x)
        if isinstance(function_value, (list, np.ndarray)):
            return function_value.copy()
        else:
            return function_value
       
    def evaluateEpsilon(self, x, epsilon):
        """
        Evaluates function at step epsilon from x
        
        Args:
            x (float, list, or array): point for evaluation of function
            epsilon (float, list, or array): same type and shape as x. Defines
                absolute step away from x at which to evaluate the function.
            
        Returns:
            function_value (float, list, or array): value of function at point
            
        """
        if isinstance(x, (list, np.ndarray)):
            x_epsilon = np.copy(x) + epsilon
        else:
            x_epsilon = x + epsilon
        return self.evaluate(x_epsilon) 


def finiteDifferenceDerivative(x,function, args=None, epsilon=1e-2,
                               method='forward'):
    """
    Approximates finite difference derivative with an N-point stencil with  
        step size epsilon
        
        Args:
            x (float, list, or np array): point to evaluate derivative of
                function.
            function (function handle): function which accepts x as first
                argument and args as additional arguments 
            args (tuple): additional arguments to pass to function (optional)
            epsilon (float): finite difference step size (optional)
            method (str): must be 'forward' or 'centered'. Determines
                finite difference method. (optional)
        Returns:
            dfdx (float, list, or np array): finite difference approximation
                of derivative of function at x. The first dimension of f 
                corresponds to the elements of x. 
        
    """
    assert method in ['forward','centered'], \
        '''method passed to finiteDifferenceDerivative must be 'forward' or 
            'centered'.'''
  
    if isinstance(x, list):
        x = np.array(x)

    assert(x.ndim<2)
    finiteDifferenceObject = finiteDifference(method, epsilon, function, args)
    # Call function once to get size of output
    test_function = finiteDifferenceObject.evaluate(x)
    if (x.ndim==1):
        dims = [len(x_arr)]
        dims.extend(list(np.shape(test_function)))
        dfdx = np.zeros(dims)
        for i in range(np.size(x_arr)):
            step = np.zeros(np.shape(x_arr))
            step[i] = epsilon
            if (method == 'centered'):
                function_r = finiteDifferenceObject.evaluateEpsilon(x, step)
                function_l = finiteDifferenceObject.evaluateEpsilon(x, -step)
            if (dims == ()):
                dfdx[i] = (function_r - function_l)/(2 * epsilon)
            else:
                dfdx[i,...] = (function_r - function_l)/ (2 * epsilon)
            if (method == 'forward'):
                function_r = finiteDifferenceObject.evaluateEpsilon(x,step)
                function_l = test_function
            if (dims == ()):
                dfdx[i] = (function_r - function_l) / (epsilon)
            else:
                dfdx[i,...] = (function_r - function_l) / (epsilon) 
    else:
        step = epsilon
        if (method == 'centered'):
            function_r = finiteDifferenceObject.evaluateEpsilon(x, step)
            function_l = finiteDifferenceObject.evaluateEpsilon(x, -step)
            dfdx = (function_r - function_l) / (2 * epsilon)
        if (method == 'forward'):
            function_r = finiteDifferenceObject.evaluateEpsilon(x, step)
            function_l = test_function
            dfdx = (function_r - function_l) / (epsilon)

    return dfdx
