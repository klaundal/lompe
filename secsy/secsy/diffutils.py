""" 
diffutils

"""

import numpy as np
from scipy.special import factorial
from fractions import Fraction


def lcm_arr(arr):
    """ Calculate least common multiplier for array of integers
    """
    result = np.lcm(arr[0], arr[1])
    for i in range(2, len(arr)-1):
        result = np.lcm(result, arr[i])

    return result


def stencil(evaluation_points, order = 1, h = 1, fraction = False):
    """ 
    Calculate stencil for finite difference calculation of derivative

    Parameters
    ----------
    evaluation_points: array_like
        evaluation points in regular grid. e.g. [-1, 0, 1] for 
        central difference or [-1, 0] for backward difference
    order: integer, optional
        order of the derivative. Default 1 (first order)
    h: scalar, optional
        Step size. Default 1
    fraction: bool, optional
        Set to True to return coefficients as integer numerators
        and a common denomenator. Be careful with this if you use
        a very large number of evaluation points...

    Returns
    -------
    coefficients: array
        array of coefficients in stencil. Unless fraction is set 
        to True - in which case a tuple will be returned with
        an array of numerators and an integer denominator. If 
        fraction is True, h is ignored - and you should multiply the 
        denominator by h**order to get the coefficients

    Note
    ----
    Algorithm from Finte Difference Coefficient Calculator
    (https://web.media.mit.edu/~crtaylor/calculator.html)
    """

    # calculate coefficients:
    evaluation_points = np.array(evaluation_points).flatten().reshape((1, -1))
    p = np.arange(evaluation_points.size).reshape((-1, 1))
    d = np.zeros(evaluation_points.size)
    d[order] = factorial(order)

    coeffs = np.linalg.inv(evaluation_points**p).dot(d)

    if fraction:
        # format nicely:
        fracs = [Fraction(c).limit_denominator() for c in coeffs]
        denominators = [c.denominator for c in fracs]
        numerators   = [c.numerator for c in fracs]
        cd = lcm_arr(denominators)
        numerators = [int(c * cd / a) for (c, a) in zip(numerators, denominators)]
        return (numerators, cd)
    else:
        return coeffs / h ** order