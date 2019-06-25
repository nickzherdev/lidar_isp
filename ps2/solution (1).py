import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from tools import *

GRID_SCALE = 0.1
GRID_SIZE = (400, 500)

poses = np.load('poses.npy')
scans = np.load('scans.npy')

delta = np.array([15, -105, 0])

assert(len(poses) == len(scans))

grid = np.zeros(GRID_SIZE, dtype=np.float32)
max_value=len(poses)
bar = ProgressBar(max_value)
for pose, scan in bar(zip(poses + delta, scans)):
    points = rotate(convert2xy(scan), pose[2]) + pose[:2]
    subgrid = convert2map(pose[:2], points, GRID_SCALE, GRID_SIZE, 0.01)
    grid += np.log(subgrid/(1-subgrid))

# convert log-odds representation to probabilities
grid = 1/(1+np.exp(-grid))

extent = [
    -delta[0], GRID_SIZE[0]*GRID_SCALE-delta[0],
    -delta[1], GRID_SIZE[1]*GRID_SCALE-delta[1],
]
plt.imshow(grid.T[::-1], vmin=0, vmax=1, cmap=plt.cm.Greys, extent=extent)
plt.plot(poses[:, 0], poses[:, 1], label="Robot trajectory")
plt.xlabel("X, m")
plt.ylabel("Y, m")
plt.legend(loc=4)
plt.grid()
plt.tight_layout()
plt.savefig("grid.png", dpi=300)
plt.show()
