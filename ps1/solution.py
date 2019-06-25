#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def get_ellipse(a, b, n):
    '''Return `n` points of an ellipse defined by `a` and `b`'''
    phi = np.linspace(0, 2*np.pi, n)
    r = a*b/np.sqrt((b*np.cos(phi))**2 + (a*np.sin(phi))**2)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return np.vstack([x, y])

def get_noisy_ellipse(sigma=0.01, a=0.5, b=1, n=100):
    '''Return `n` points of an ellipse defined by `a` and `b` with added
    normal noise with standard deviation `sigma`'''
    xy = get_ellipse(a, b, n)
    return xy + np.random.normal(scale=sigma, size=xy.shape)

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
    return r, t

def find_corresp(points, ref_points, thresh=20):
    crsp_p = []
    crsp_rp = []
    for point in points.T:
        dist = np.linalg.norm(ref_points - point[:, np.newaxis], axis=0)
        amin = np.argmin(dist)
        if dist[amin] < thresh:
            crsp_p.append(point)
            crsp_rp.append(ref_points[:, amin])
    return np.array(crsp_p).T, np.array(crsp_rp).T

def icp_step(points, ref_points, guess, plot=False):
    points = transform(points, guess)
    crsp_p, crsp_rp = find_corresp(points, ref_points)

    if plot:
        plt.clf()
        plt.plot(ref_points[0, :], ref_points[1, :], '.', label="reference scan")
        plt.plot(points[0, :], points[1, :], '.', label="original scan")

        for (cp, crp) in zip(crsp_p.T, crsp_rp.T):
            plt.plot([cp[0], crp[0]], [cp[1], crp[1]], 'black')
        plt.axis('equal')
        plt.legend()
        plt.grid()
        input()

    r, t = corresp_icp_step(crsp_p, crsp_rp)
    gr, gt = guess
    return r.dot(gr), r.dot(gt) + t

if __name__ == "__main__":
    rp = get_noisy_ellipse()
    p = transform(get_noisy_ellipse(n=80), get_rt([-0.2, 0.3, 0.5]))

    plt.ion()

    rt = np.identity(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    while True:
        rt = icp_step(p, rp, rt, plot=True)
        print(rt)
