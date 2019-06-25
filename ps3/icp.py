#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def convert2xy(scan, fov=260, min_dist=0.02):
    """Converts scan data to list of XY points descarding values with distances
    less than `min_dist`.

    Parameters
    ----------
    scan : array-like
        List of scan measurments in mm
    fov : scalar, optional
        Field Of View of the sensor in degrees
    min_dist : scalar, optional
        Minimal distance of measurment in mm for filtering out values

    Returns
    -------
    points : ndarray
        List of XY points
    """
    angles = np.radians(np.linspace(-fov/2, fov/2, len(scan)))
    points = np.vstack([scan*np.cos(angles), scan*np.sin(angles)]).T
    return points[scan>min_dist]

def get_rt(x):
    '''Get rototranslation from vector of deltas `[dx, dy, d_theta]`'''
    s, c = np.sin(x[2]), np.cos(x[2])
    r = np.array([[c, s], [-s, c]])
    t = np.array(x[:2])
    return r, t

def transform(points, rt):
    '''Transform `points` using given rototranslation `rt`'''
    r, t = rt
    return r.dot(points) + t[:, np.newaxis]

def corresp_icp_step(points, ref_points):
    '''Perform ICP iteration on corresponding points. Length of both lists
    must be equal.'''
    assert(points.shape == ref_points.shape)
    # centers of masses
    cm_p = np.mean(points, axis=1)
    cm_rp = np.mean(ref_points, axis=1)
    # center correspondencies list
    centered_p = points - cm_p[:, np.newaxis]
    centered_rp = ref_points - cm_rp[:, np.newaxis]
    # find `r` and `t` which minimize the sum distance between correspondencies
    m = centered_p.dot(centered_rp.T)
    u, s, v = np.linalg.svd(m)
    r = v.dot(u.T)
    t = cm_rp - r.dot(cm_p)
    new_err = np.sum(centered_p[0]**2 + centered_p[1]**2 + centered_rp[0]**2 + centered_rp[1]**2) - 2*np.sum(s)
    return r, t, new_err

def find_corresp(points, ref_points, thresh=1):
    crsp_p = []
    crsp_rp = []
    for point in points.T:
        dist = np.linalg.norm(ref_points - point[:, np.newaxis], axis=0)
        amin = np.argmin(dist)
        if dist[amin] < thresh:
            crsp_p.append(point)
            crsp_rp.append(ref_points[:, amin])
    return np.array(crsp_p).T, np.array(crsp_rp).T

def icp_step(points, ref_points, guess):
    points = transform(points, guess)
    crsp_p, crsp_rp = find_corresp(points, ref_points)
    r, t, new_err = corresp_icp_step(crsp_p, crsp_rp)
    gr, gt = guess
    return r.dot(gr), r.dot(gt) + t, new_err

def icp(points, ref_points, guess, eps=0.001, max_iters=100):
    '''Find optimal roto-translation using simple Iterative Closest Point algorithm.

    Parameters
    ----------
    points : ndarray
        Array of XY points to align with a shape (N, 2)
    ref_points : ndarray
        Array of XY reference points a shape (N, 2)
    guess : (ndarray, ndarray)
        Initial guess (rotational and translational matrices)
    eps : float
        Termination criteria based on an error change speed.
    max_iters : int
        Maximum number of iterations to perform.
    '''
    old_err = None
    for _ in range(max_iters):
        r, t, new_err = icp_step(points.T, ref_points.T, guess)
        guess = r, t
        new_eps = abs(old_err - new_err)/old_err if old_err is not None else 1
        if new_eps < eps: break
        old_err = new_err
    return guess

if __name__ == "__main__":
    scans = np.load('scans.npy')
    xy0 = convert2xy(scans[0])
    xy1 = convert2xy(scans[10])

    guess = get_rt([0,0,0])
    res = icp(xy0, xy1, guess)
    print("Result:", res[0], res[1])
    plt.plot(xy0[:, 0], xy0[:, 1], '.', label="before")
    xy3 = xy0.dot(res[0].T) + res[1]
    plt.plot(xy3[:, 0], xy3[:, 1], '.', label="after")
    plt.plot(xy1[:, 0], xy1[:, 1], '.', label="ref")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()
