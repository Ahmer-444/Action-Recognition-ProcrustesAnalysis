import numpy as np

def procrustes_distance(reference_shape, shape):
    
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]
    
    x = shape[::2]
    y = shape[1::2]
    
    dist = np.sum(np.sqrt((ref_x - x)**2 + (ref_y - y)**2))
    return dist
