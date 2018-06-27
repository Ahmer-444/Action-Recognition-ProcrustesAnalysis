import numpy as np
from triangles import *
from translate_to_origin import *
from rotation_translate import *
from procrustes_analysis import *
from procrustes_distance import *
from GPA import *

if __name__ == '__main__':
    
    canvas, triangles = create_test_set()

    #draw_shapes(canvas, triangles)

    #get translation of reference landmark
    x,y = get_translation(triangles[0])
    #create array for new shapes, append reference shape to it
    new_shapes = []
    variations = []

    new_shapes.append(triangles[0])
    
    # calculate the mean shape first
    #mean_shape, new_shapes = 
    
    #superimpose all shapes to reference shape
    for i in range(1,5):
        new_shape = procrustes_analysis(triangles[0], triangles[i])
        new_shape[::2] = new_shape[::2] + x
        new_shape[1::2] = new_shape[1::2] + y
        new_shapes.append(new_shape)
    
    new_shapes = [ map(int ,x) for x in new_shapes ]
    new_shapes = [ np.array(x) for x in new_shapes ]
    draw_shapes(canvas, new_shapes)

    for i in range(5):
        dist = procrustes_distance(triangles[0], new_shapes[i])
        variations.append(dist)

    print variations