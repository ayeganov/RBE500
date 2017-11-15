import numpy as np
from scipy import linalg


def dot3(A, B, C):
    '''
    Multiplies 3 matrices.
    '''
    return np.dot(A, np.dot(B, C))


def predict(x, P, F=1, Q=0):
    '''
    Predict next position using the Kalman filter state propagation equations.

    @param x:numpy.array - state vector
    @param P:numpy.array - covariance matrix
    @param F:numpy.array - state transition matrix
    @param Q:numpy.array - process noise matrix

    @Returns:
        x:numpy.array - prior state vector
        P:numpy.array - prior covariance matrix
    '''
    if np.isscalar(F):
        F = np.array(F)

    x = np.dot(F, x)
    P = dot3(F, P, F.T) + Q

    return x, P


def update(x, P, z, R, H):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.

    Parameters
    ----------
    @param x:numpy.array - State estimate vector
    @param P:numpy.array - Covariance matrix
    @param z:numpy.array - measurement for this update
    @param R:numpy.array - Measurement noise matrix
    @param H:numpy.array - Measurement conversion matrix

    @Returns:
        x:numpy.array - Posterior state estimate vector
        P:numpy.array - Posterior covariance matrix
    """
    if z is None:
        return x, P

    # error (residual) between measurement and prediction
    y = z - np.dot(H, x)

    # project system uncertainty into measurement space
    S = dot3(H, P, H.T) + R

    # map system uncertainty into kalman gain
    try:
        K = dot3(P, H.T, linalg.inv(S))
    except:
        K = dot3(P, H.T, 1/S)

    # predict new x with residual scaled by the kalman gain
    x = x + np.dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = np.dot(K, H)

    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array(1 - KH)
    P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)

    return x, P
