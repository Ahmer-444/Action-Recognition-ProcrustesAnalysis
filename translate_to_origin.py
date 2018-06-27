import numpy as np

def get_translation(shape):
  '''
  Calculates a translation for x and y
  axis that centers shape around the
  origin
  Args:
    shape(2n x 1 NumPy array) an array 
    containing x coodrinates of shape
    points as first column and y coords
    as second column
   Returns:
    translation([x,y]) a NumPy array with
    x and y translationcoordinates
  '''
  
  mean_x = np.mean(shape[::2]).astype(np.int)
  mean_y = np.mean(shape[1::2]).astype(np.int)
  
  return np.array([mean_x, mean_y])

def translate(shape):
  '''
  Translates shape to the origin
  Args:
    shape(2n x 1 NumPy array) an array 
    containing x coodrinates of shape
    points as first column and y coords
    as second column
  '''
  mean_x, mean_y = get_translation(shape)
  shape[::2] -= mean_x
  shape[1::2] -= mean_y