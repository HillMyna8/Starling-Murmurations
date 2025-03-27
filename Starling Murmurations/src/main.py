# Import libraries
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial.distance import pdist, squareform
from IPython.display import HTML


plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg' # Added to

# %%
# Initialize plot
L = 450
fig = plt.figure()
ax = plt.axes(xlim=(-L, L), ylim=(-L, L))
points, = ax.plot([], [], '*', color = 'blue')
plt.axis('off')


tsteps         = 200        # Time steps
n              = 300        # Number of birds
V0             = 20         # Initial velocity

wall_repulsion = 20         # https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html#Screen-edges
margin         = 20
max_speed      = 20
neighbors_dist = 70         # Community distance (At which distance they apply rule 1 and rule 3)

# Rule 1: COHESION
R = 0.1                     # velocity to center contribution

# Rule 2: SEPARATION
bird_repulsion = 7          # low values  make more "stains" of birds, nice visualization
privacy        = 14         #  Avoid each other ,length. When they see each other at this distance,

# Rule 3: ALIGNMENT
match_velocity = 3


x = np.zeros((n,tsteps))
y = np.zeros((n,tsteps))

# Randomize initial positions
x[:,0] = np.random.uniform(low=-L, high=L, size=(int(n),))
y[:,0] = np.random.uniform(low=-L, high=L, size=(int(n),))

# Randomize initial velocity
x[:,1] = x[:,0] + np.random.uniform(low=-V0, high=V0, size=(int(n),))
y[:,1] = y[:,0] + np.random.uniform(low=-V0, high=V0, size=(int(n),))


def moveToCenter(x0, y0, neighbors_dist, n, R):
    m = squareform(pdist(np.transpose([x0, y0])))
    idx = (m <= neighbors_dist)  # & (m!=0)

    center_x = np.zeros(n)
    center_y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    for i in range(0, n - 1):
        center_x[i] = np.mean(x0[idx[i,]])
        center_y[i] = np.mean(y0[idx[i,]])
        vx[i] = -(x0[i] - center_x[i]) * R
        vy[i] = -(y0[i] - center_y[i]) * R

    return vx, vy


def avoidOthers(x0, y0, n, privacy, bird_repulsion):
    dist = squareform(pdist(np.transpose([x0, y0])))

    idxmat = (dist < privacy) & (dist != 0)
    idx = np.transpose(np.array(np.where(idxmat)))

    vx = np.zeros(n)
    vy = np.zeros(n)

    vx[idx[:, 0]] = (x0[idx[:, 0]] - x0[idx[:, 1]]) * bird_repulsion
    vy[idx[:, 0]] = (y0[idx[:, 0]] - y0[idx[:, 1]]) * bird_repulsion
    return vx, vy


def matchVelocities(x_prev, y_prev, x0, y0, n, neighbors_dist, match_velocity):
    m = squareform(pdist(np.transpose([x_prev, y_prev])))
    idx = (m <= neighbors_dist)  # & (m!=0)

    vmean_x = np.zeros(n)
    vmean_y = np.zeros(n)
    for i in range(0, n - 1):
        vmean_x[i] = np.mean(x0[idx[i,]] - x_prev[idx[i,]])
        vmean_y[i] = np.mean(y0[idx[i,]] - y_prev[idx[i,]])

    return vmean_x * match_velocity, vmean_y * match_velocity


def move(x0, y0, x_prev, y_prev, n, neighbors_dist, R, privacy, bird_repulsion, match_velocity, L, margin,
         wall_repulsion, max_speed):
    vx1, vy1 = moveToCenter(x0, y0, neighbors_dist, n, R)
    vx2, vy2 = avoidOthers(x0, y0, n, privacy, bird_repulsion)
    vx3, vy3 = matchVelocities(x_prev, y_prev, x0, y0, n, neighbors_dist, match_velocity)

    vx = x0 - x_prev + vx1 + vx2 + vx3
    vy = y0 - y_prev + vy1 + vy2 + vy3

    # max speed limit
    # Matrix 2xn. Get the length of the velocity vector for each boid, and
    # scale it with the maximum value
    v_norm = np.zeros((2, n))
    v_vector = np.array([vx, vy])
    norm = np.linalg.norm(v_vector, axis=0)
    v_norm[:, norm != 0] = v_vector[:, norm != 0] / norm[norm != 0] * max_speed

    vx = v_norm[0, :]
    vy = v_norm[1, :]

    # Dump velocity when hits a wall
    right_border_dist = L - x0
    left_border_dist = x0 + L
    upper_border_dist = L - y0
    bottom_border_dist = y0 + L

    vx[right_border_dist < margin] = vx[right_border_dist < margin] - wall_repulsion
    vx[left_border_dist < margin] = vx[left_border_dist < margin] + wall_repulsion
    vy[upper_border_dist < margin] = vy[upper_border_dist < margin] - wall_repulsion
    vy[bottom_border_dist < margin] = vy[bottom_border_dist < margin] + wall_repulsion

    x1 = x0 + vx
    y1 = y0 + vy

    x1 = np.round(x1)
    y1 = np.round(y1)

    return x1, y1


for t in range(1, tsteps - 1):
    x[:, t + 1], y[:, t + 1] = move(x[:, t], y[:, t], x[:, t - 1], y[:, t - 1],
                                    n, neighbors_dist, R, privacy, bird_repulsion, match_velocity,
                                    L, margin, wall_repulsion, max_speed)


def init():
    points.set_data([], [])
    return points,


def animate(i):
    xx = x[:, i]
    yy = y[:, i]
    points.set_data(xx, yy)
    return points,


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=tsteps - 2, interval=80, blit=True)

HTML(anim.to_html5_video())

plt.show()