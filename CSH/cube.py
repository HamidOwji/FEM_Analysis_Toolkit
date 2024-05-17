import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Define the vertices of the cube
vertices = np.array([
    [0, 0, 0],  # Vertex 0
    [1, 0, 0],  # Vertex 1
    [1, 1, 0],  # Vertex 2
    [0, 1, 0],  # Vertex 3
    [0, 0, 1],  # Vertex 4
    [1, 0, 1],  # Vertex 5
    [1, 1, 1],  # Vertex 6
    [0, 1, 1]   # Vertex 7
])

# Define the list of sides by connecting vertices
sides = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Opposite side
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Front
    [vertices[0], vertices[3], vertices[7], vertices[4]]   # Back
]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot sides
ax.add_collection3d(Poly3DCollection(sides, 
 facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

# Plot vertices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="black")

# Label vertices
for i, vertex in enumerate(vertices):
    ax.text(vertex[0], vertex[1], vertex[2], f'{i}', color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
