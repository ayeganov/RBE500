import numpy as np


def dot3(A, B, C):
    '''
    Multiplies 3 matrices.
    '''
    return np.dot(A, np.dot(B, C))


def predict(x, P, F=1, Q=0, B=1, u=0):
    '''
    Predict next position using the Kalman filter state propagation equations.

    @param x:numpy.array - state vector
    @param P:numpy.array - covariance matrix
    @param F:numpy.array - state transition matrix
    @param Q:numpy.array - process noise matrix
    @param B:numpy.array - control transition matrix
    @param u:numpy.array - control vector. This is multiplied by B to the
                           create the control input into the system

    @Returns:
        x:numpy.array - prior state vector
        P:numpy.array - prior covariance matrix
    '''
    if np.isscalar(F):
        F = np.array(F)

    x = np.dot(F, x) + np.dot(B, u)
    P = dot3(F, P, F.T) + Q

    return x, P


def update(x, P, z, R, H=None, return_all=False):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.
    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.

    update(1, 2, 1, 1, 1)  # univariate

    Parameters
    ----------
    x : numpy.array(dim_x, 1), or float
        State estimate vector
    P : numpy.array(dim_x, dim_x), or float
        Covariance matrix
    z : numpy.array(dim_z, 1), or float
        measurement for this update.
    R : numpy.array(dim_z, dim_z), or float
        Measurement noise matrix
    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.
    return_all : bool, default False
        If true, y, K, S, and log_likelihood are returned, otherwise
        only x and P are returned.

    Returns
    -------
    x : numpy.array
        Posterior state estimate vector
    P : numpy.array
        Posterior covariance matrix
    y : numpy.array or scalar
        Residua. Difference between measurement and state in measurement space
    K : numpy.array
        Kalman gain
    S : numpy.array
        System uncertainty in measurement space
    log_likelihood : float
        log likelihood of the measurement
    """


    if z is None:
        if return_all:
            return x, P, None, None, None, None
        else:
            return x, P

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    if not np.isscalar(x):
        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

    # error (residual) between measurement and prediction
    y = z - dot(H, x)

    # project system uncertainty into measurement space
    S = dot3(H, P, H.T) + R

    # map system uncertainty into kalman gain
    try:
        K = dot3(P, H.T, linalg.inv(S))
    except:
        K = dot3(P, H.T, 1/S)

    # predict new x with residual scaled by the kalman gain
    x = x + dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = dot(K, H)

    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array(1 - KH)
    P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    if return_all:
        # compute log likelihood
        log_likelihood = logpdf(z, dot(H, x), S)
        return x, P, y, K, S, log_likelihood
    else:
        return x, P
