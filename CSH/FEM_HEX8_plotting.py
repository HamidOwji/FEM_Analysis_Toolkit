import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh_3d(elements, node_coordinates):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = [coord[0] for coord in node_coordinates]
    y_coords = [coord[1] for coord in node_coordinates]
    z_coords = [coord[2] for coord in node_coordinates]
    max_range = np.array([max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords)]).max() / 2.0

    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5

    for element in elements:
        try:
            if all(node-1 < len(node_coordinates) and node-1 >= 0 for node in element['nodes']):
                coords = [node_coordinates[node-1] for node in element['nodes']]
                vertices = [coords[i] for i in range(8)]
                sides = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],
                    [vertices[4], vertices[5], vertices[6], vertices[7]],
                    [vertices[0], vertices[1], vertices[5], vertices[4]],
                    [vertices[2], vertices[3], vertices[7], vertices[6]],
                    [vertices[0], vertices[3], vertices[7], vertices[4]],
                    [vertices[1], vertices[2], vertices[6], vertices[5]]
                ]
                ax.add_collection3d(Poly3DCollection(sides, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))
                for idx, (x, y, z) in enumerate(coords):
                    ax.text(x, y, z, f'{element["nodes"][idx]}', color='black')
        except Exception as e:
            print(f"Error plotting element {element['nodes']}: {e}")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Element Mesh Plot')
    plt.show()

def plot_mesh_with_conditions_and_forces(elements, node_coordinates, fixed_nodes, applied_forces):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = [coord[0] for coord in node_coordinates]
    y_coords = [coord[1] for coord in node_coordinates]
    z_coords = [coord[2] for coord in node_coordinates]
    max_range = np.array([max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords)]).max() / 2.0

    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5

    for element in elements:
        try:
            if all(node-1 < len(node_coordinates) and node-1 >= 0 for node in element['nodes']):
                coords = [node_coordinates[node-1] for node in element['nodes']]
                vertices = [coords[i] for i in range(8)]
                sides = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],
                    [vertices[4], vertices[5], vertices[6], vertices[7]],
                    [vertices[0], vertices[1], vertices[5], vertices[4]],
                    [vertices[2], vertices[3], vertices[7], vertices[6]],
                    [vertices[0], vertices[3], vertices[7], vertices[4]],
                    [vertices[1], vertices[2], vertices[6], vertices[5]]
                ]
                ax.add_collection3d(Poly3DCollection(sides, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))
        except Exception as e:
            print(f"Error plotting element {element['nodes']}: {e}")

    for node in fixed_nodes:
        ax.scatter(*node_coordinates[node], color='red', s=50, label='Fixed Node')

    for i, force in enumerate(applied_forces):
        start_point = node_coordinates[i // 3]
        if force != 0:
            direction = np.zeros(3)
            direction_index = i % 3
            direction[direction_index] = force
            ax.quiver(start_point[0], start_point[1], start_point[2], direction[0], direction[1], direction[2], color='green', length=.02, normalize=False)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Element Mesh with Boundary Conditions and Forces')
    plt.legend()
    plt.show()

def plot_mesh_with_conditions_forces_and_deformation(elements, node_coordinates, fixed_nodes, applied_forces, U, deformation_scale=1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = [coord[0] for coord in node_coordinates]
    y_coords = [coord[1] for coord in node_coordinates]
    z_coords = [coord[2] for coord in node_coordinates]
    max_range = np.array([max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords)]).max() / 2.0

    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5

    for element in elements:
        if all(node-1 < len(node_coordinates) and node-1 >= 0 for node in element['nodes']):
            coords = [node_coordinates[node-1] for node in element['nodes']]
            vertices = [coords[i] for i in range(8)]
            sides = get_element_sides(vertices)
            ax.add_collection3d(Poly3DCollection(sides, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.2))
    
    deformed_node_coordinates = [np.add(coord, U[3*i:3*i+3] * deformation_scale) for i, coord in enumerate(node_coordinates)]
    for element in elements:
        if all(node-1 < len(deformed_node_coordinates) and node-1 >= 0 for node in element['nodes']):
            coords = [deformed_node_coordinates[node-1] for node in element['nodes']]
            vertices = [coords[i] for i in range(8)]
            sides = get_element_sides(vertices)
            ax.add_collection3d(Poly3DCollection(sides, facecolors='blue', linewidths=1, edgecolors='black', alpha=0.5))

    for node in fixed_nodes:
        ax.scatter(*node_coordinates[node], color='red', s=50, label='Fixed Node')

    for i, force in enumerate(applied_forces):
        if force != 0:
            start_point = node_coordinates[i // 3]
            direction = np.zeros(3)
            direction_index = i % 3
            direction[direction_index] = force / np.linalg.norm(force) if np.linalg.norm(force) != 0 else 0
            ax.quiver(start_point[0], start_point[1], start_point[2], direction[0], direction[1], direction[2], color='green', length=max_range*0.1, normalize=True)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Element Mesh with Boundary Conditions, Forces, and Deformation')
    plt.legend()
    plt.show()

def get_element_sides(vertices):
    """Utility function to get the sides of an element from its vertices."""
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]
