from math import sin, cos
import numpy as np

def get_rotation_matrix(theta):
    
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def scale(shape, scale):
    
    return shape / scale

def rotate(shape, theta):
    '''
    Rotates a shape by angle theta
    Assumes a shape is centered around 
    origin
    Args:
        shape(2nx1 NumPy array) an shape to be rotated
        theta(float) angle in radians
    Returns:
        rotated_shape(2nx1 NumPy array) a rotated shape
    '''
    
    matr = get_rotation_matrix(theta)
    
    #reshape so that dot product is eascily computed
    temp_shape = shape.reshape((-1,2)).T
    
    #rotate
    rotated_shape = np.dot(matr, temp_shape)
    
    return rotated_shape.T.reshape(-1)