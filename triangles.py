import cv2
import numpy as np
from random import randint
from time import *

def show_image(img):
    '''
    Displays an image
    Args:
        img(a NumPy array of type uint 8) an image to be
        dsplayed
    '''
    
    cv2.imshow('', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
def generate_color():
    '''
    Generates a random combination
    of red, green and blue channels
    Returns:
        (r,g,b), a generated tuple
    '''
    col = []
    for i in range(3):
        col.append(randint(0, 255))
        
    return tuple(col)

def create_test_set():
    
    #create canvas on which the triangles will be visualized
    canvas = np.full([400,400], 255).astype('uint8')
    
    #convert to 3 channel RGB for fun colors!
    canvas = cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB)
    
    #initialize triangles as sets of vertex coordinates (x,y)
    triangles = []
    triangles.append(np.array([250,250, 250,150, 300,250]))
    #tr1 translated by 50 points on both axis
    triangles.append(triangles[0] - 50)
    #tr1 shrinked and consequently translated as well
    triangles.append((triangles[0] / 2).astype(np.int))
    #tr1 rotated by 90 defrees annd translated by 20 pixels
    triangles.append(np.array([250,250,150,250, 250, 200]) - 20)
    #a random triangle
    triangles.append(np.array([360,240, 370,100, 390, 240]))
    
    return canvas, triangles

def draw_shapes(canvas, shapes):
    '''
    Draws shapes on canvas
    Args:
        canvas(a NumPy matrix), a background on which
        shapes are drawn
        shapes(list), shapes to be drawn
    '''
    for sh in shapes:
        pts = sh.reshape((-1,1,2))
        color = generate_color()
        cv2.polylines(canvas, [pts], True, color, 2)
    
    show_image(canvas)

if __name__ == '__main__':
    
    canvas, triangles = create_test_set()
    draw_shapes(canvas, triangles)