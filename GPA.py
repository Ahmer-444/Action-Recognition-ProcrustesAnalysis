
def generalized_procrustes_analysis(shapes):
    '''
    Performs superimposition on a set of 
    shapes, calculates a mean shape
    Args:
        shapes(a list of 2nx1 Numpy arrays), shapes to
        be aligned
    Returns:
        mean(2nx1 NumPy array), a new mean shape
        aligned_shapes(a list of 2nx1 Numpy arrays), super-
        imposed shapes
    '''
    #initialize Procrustes distance
    current_distance = 0
    
    #initialize a mean shape
    mean_shape = np.array(shapes[0])

    num_shapes = len(shapes)
    
    #create array for new shapes, add 
    new_shapes = np.zeros(np.array(shapes).shape)
    
    
    while True:
        
        #add the mean shape as first element of array
        new_shapes[0] = mean_shape
        
        #superimpose all shapes to current mean
        for sh in range(1, num_shapes):
            new_sh = procrustes_analysis(mean_shape, shapes[sh])
            new_shapes[sh] = new_sh
        
        #calculate new mean
        new_mean = np.mean(new_shapes, axis = 0)
        
        new_distance = procrustes_distance(new_mean, mean_shape)
        
        #if the distance did not change, break the cycle
        if new_distance == current_distance:
            break
        
        #align the new_mean to old mean
        new_mean = procrustes_analysis(mean_shape, new_mean)
        
        #update mean and distance
        mean_shape = new_mean
        current_distance = new_distance
        
    return mean_shape, new_shapes 