from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# do not start from 0 because div by zero.
t1 = np.arange(0.1, 1, 0.1)
t2 = np.arange(0.1, 1, 0.1)

r1 = np.empty((len(t1), len(t1)))
r1.fill(79)

r2 = np.empty((len(t2), len(t2)))
r2.fill(67)

tt1, tt2 = np.meshgrid(t1, t2)

Z = tt1 * r1 + tt2 * r2
Z = Z / (tt1 + tt2)

surf = ax.plot_surface(tt1, tt2, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
ax.set_zlim(65, 80)
ax.set_xlabel('theta1')
ax.set_ylabel('theta2')
ax.set_zlabel('y(theta)')

plt.show()

