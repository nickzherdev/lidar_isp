import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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

def convert2map(pose, points, map_pix, map_size, prob):
    """Converts list of XY points to 2D array map in which each pixel denotes
    probability of pixel being occupied.

    Parameters
    ----------
    pose : ndarray
        XY coordinates of the robot in the map reference frame
    points : ndarray
        List of XY points measured by sensor in the map reference frame
    map_pix : int
        Size of map pixel in m
    map_size : tuple
        Size of the map in pixels
    prob : float
        Probability


    Returns
    -------
    map : ndarray
        2D array representing map with dtype numpy.float32
    """
    zero = (pose//map_pix).astype(np.int32)
    pixels = (points//map_pix).astype(np.int32)
    mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < map_size[0]) & \
           (pixels[:, 1] >= 0) & (pixels[:, 1] < map_size[1])
    pixels = pixels[mask]
    img = Image.new('L', (map_size[1], map_size[0]))
    draw = ImageDraw.Draw(img)
    zero = (zero[1], zero[0])
    for p in set([(q[1], q[0]) for q in pixels]):
        draw.line([zero, p], fill=1)
    data = -np.fromstring(img.tobytes(), np.int8).reshape(map_size)
    data[pixels[:, 0], pixels[:, 1]] = 1
    return 0.5 + prob*data.astype(np.float32)
