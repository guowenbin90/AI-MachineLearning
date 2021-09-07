#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    import numpy as np
    errors = net_worths - predictions
    threshold = np.percentile(np.absolute(errors),90)
    #print threshold
    #print np.max(numpy.absolute(errors))
    #print np.sort(numpy.absolute(errors), axis=0)
    cleaned_data = [(age, net_worth, error) for age, net_worth, error in zip(ages, net_worths, errors) if abs(error)<=threshold]


    return cleaned_data

