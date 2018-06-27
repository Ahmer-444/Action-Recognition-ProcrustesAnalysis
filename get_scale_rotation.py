from scipy.linalg import norm
import numpy as np
from math import atan

def get_rotation_scale(reference_shape, shape):
    '''
    Calculates rotation and scale
    that would optimally align shape
    with reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference for scaling and 
        alignment
        
        shape(2nx1 NumPy array), a shape that is scaled
        and aligned
        
    Returns:
        scale(float), a scaling factor
        theta(float), a rotation angle in radians
    '''
    
    a = np.dot(shape, reference_shape) / norm(reference_shape)**2
    
    #separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]
    
    x = shape[::2]
    y = shape[1::2]
    
    b = np.sum(x*ref_y - ref_x*y) / norm(reference_shape)**2
    
    scale = np.sqrt(a**2+b**2)
    theta = atan(b / max(a, 10**-10)) #avoid dividing by 0
    return round(scale,1), round(theta,2)