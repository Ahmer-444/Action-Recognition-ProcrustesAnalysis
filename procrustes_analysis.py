import numpy as np
from translate_to_origin import *
from get_scale_rotation import *
from rotation_translate import *

def procrustes_analysis(reference_shape, shape):
    '''
    Scales, and rotates a shape optimally to
    be aligned with a reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference alignment
        
        shape(2nx1 NumPy array), a shape that is aligned
        
    Returns:
        aligned_shape(2nx1 NumPy array), an aligned shape
        translated to the location of reference shape
    '''
    #copy both shapes in caseoriginals are needed later
    temp_ref = np.copy(reference_shape)
    temp_sh = np.copy(shape)
 
    translate(temp_ref)
    translate(temp_sh)
    
    #get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)
    
    #scale, rotate both shapes
    temp_sh = temp_sh / scale
    aligned_shape = rotate(temp_sh, theta)
    return aligned_shape