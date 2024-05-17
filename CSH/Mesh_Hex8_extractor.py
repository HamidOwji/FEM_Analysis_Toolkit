import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # This is important for 3D plotting

def divide_list_into_sublists(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_coordinates(lst, number_of_coordinates=3):
    """Assuming number_of_coordinates defaults to 3 for 3D models."""
    for i in range(0, len(lst), number_of_coordinates):
        yield lst[i:i + number_of_coordinates]

def generate_elements(node_coordinates, element_node_connectivity):
    elements = []
    for element in element_node_connectivity:
        # Assuming element ordering is correct for HE8, if not, adjust as per your element definition
        element_dict = {
            'nodes': element,
            'coords': np.array([node_coordinates[node - 1] for node in element])
        }
        elements.append(element_dict)
    return elements

def read_mesh_data(file_name):
    with h5py.File(file_name, 'r') as file:
        # Read coordinate data for 3D
        coo_dataset = file['ENS_MAA/Mesh_1/-0000000000000000001-0000000000000000001/NOE/COO']
        coo_data = coo_dataset[:]
        num_nodes = len(coo_data) // 3
        subcoord = list(extract_coordinates(coo_data, num_nodes))
        # Assuming every 3 values represent X, Y, Z coordinates for a node
        node_coordinates = [group for group in zip(*subcoord)]
        # print('node_coordinates:', node_coordinates)

        
        # Accessing QU8 element connectivity
        he8_dataset = file['ENS_MAA/Mesh_1/-0000000000000000001-0000000000000000001/MAI/HE8/NOD']
        he8_data = he8_dataset[:]
        num_quadrilaterals = len(he8_data) // 8
        sublists = list(divide_list_into_sublists(he8_data, num_quadrilaterals))
        element_node_connectivity = [group for group in zip(*sublists)]
        # print('element_node_connectivity:', element_node_connectivity)
    return node_coordinates, element_node_connectivity




def plot_mesh_3d(elements, node_coordinates):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the ranges for the x, y, and z coordinates to determine the aspect ratio
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
                sides = [[vertices[0], vertices[1], vertices[2], vertices[3]],
                         [vertices[4], vertices[5], vertices[6], vertices[7]],
                         [vertices[0], vertices[1], vertices[5], vertices[4]],
                         [vertices[2], vertices[3], vertices[7], vertices[6]],
                         [vertices[0], vertices[3], vertices[7], vertices[4]],
                         [vertices[1], vertices[2], vertices[6], vertices[5]]]

                ax.add_collection3d(Poly3DCollection(sides, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))
                
                for idx, (x, y, z) in enumerate(coords):
                    ax.text(x, y, z, f'{element["nodes"][idx]}', color='black')
                    
        except Exception as e:
            print(f"Error plotting element {element['nodes']}: {e}")

    # Setting the aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Alternatively, to center the plot around the mesh, adjust the axis limits manually
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

    # Calculate the ranges for the x, y, and z coordinates
    x_coords = [coord[0] for coord in node_coordinates]
    y_coords = [coord[1] for coord in node_coordinates]
    z_coords = [coord[2] for coord in node_coordinates]
    max_range = np.array([max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords)]).max() / 2.0

    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5

    # Plot the elements
    for element in elements:
        try:
            if all(node-1 < len(node_coordinates) and node-1 >= 0 for node in element['nodes']):
                coords = [node_coordinates[node-1] for node in element['nodes']]
                vertices = [coords[i] for i in range(8)]
                sides = [[vertices[0], vertices[1], vertices[2], vertices[3]],
                         [vertices[4], vertices[5], vertices[6], vertices[7]],
                         [vertices[0], vertices[1], vertices[5], vertices[4]],
                         [vertices[2], vertices[3], vertices[7], vertices[6]],
                         [vertices[0], vertices[3], vertices[7], vertices[4]],
                         [vertices[1], vertices[2], vertices[6], vertices[5]]]

                ax.add_collection3d(Poly3DCollection(sides, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))
        except Exception as e:
            print(f"Error plotting element {element['nodes']}: {e}")

    # Highlight fixed nodes (boundary conditions)
    for node in fixed_nodes:
        ax.scatter(*node_coordinates[node], color='red', s=50, label='Fixed Node')

    # Plot applied forces
    for i, force in enumerate(applied_forces):
        start_point = node_coordinates[i // 3]
        if force != 0:
            print('start point:', start_point)
            direction = np.zeros(3)  # Initialize a zero vector for the direction
            direction_index = i % 3  # Determine the direction (0=x, 1=y, 2=z) of the force based on the remainder
            direction[direction_index] = force  # Apply force in the determined direction
            print('direction:', direction)
            ax.quiver(start_point[0], start_point[1], start_point[2], direction[0], direction[1], direction[2], color='green', length=.02, normalize=False)


    # Setting the aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

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

    # Calculate the ranges for the x, y, and z coordinates
    x_coords = [coord[0] for coord in node_coordinates]
    y_coords = [coord[1] for coord in node_coordinates]
    z_coords = [coord[2] for coord in node_coordinates]
    max_range = np.array([max(x_coords) - min(x_coords), max(y_coords) - min(y_coords), max(z_coords) - min(z_coords)]).max() / 2.0

    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5

    # Plot the original mesh
    for element in elements:
        if all(node-1 < len(node_coordinates) and node-1 >= 0 for node in element['nodes']):
            coords = [node_coordinates[node-1] for node in element['nodes']]
            vertices = [coords[i] for i in range(8)]
            sides = get_element_sides(vertices)
            ax.add_collection3d(Poly3DCollection(sides, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.2))
    
    # Plot the deformed mesh
    deformed_node_coordinates = [np.add(coord, U[3*i:3*i+3] * deformation_scale) for i, coord in enumerate(node_coordinates)]
    for element in elements:
        if all(node-1 < len(deformed_node_coordinates) and node-1 >= 0 for node in element['nodes']):
            coords = [deformed_node_coordinates[node-1] for node in element['nodes']]
            vertices = [coords[i] for i in range(8)]
            sides = get_element_sides(vertices)
            ax.add_collection3d(Poly3DCollection(sides, facecolors='blue', linewidths=1, edgecolors='black', alpha=0.5))

    # Highlight fixed nodes (boundary conditions)
    for node in fixed_nodes:
        ax.scatter(*node_coordinates[node], color='red', s=50, label='Fixed Node')

    # Plot applied forces
    for i, force in enumerate(applied_forces):
        if force != 0:
            start_point = node_coordinates[i // 3]
            direction = np.zeros(3)
            direction_index = i % 3
            direction[direction_index] = force / np.linalg.norm(force) if np.linalg.norm(force) != 0 else 0
            ax.quiver(start_point[0], start_point[1], start_point[2], direction[0], direction[1], direction[2], color='green', length=max_range*0.1, normalize=True)

    # Setting the aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

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

if __name__ == "__main__":
    node_coordinates, element_node_connectivity = read_mesh_data('Mesh_1.med')
    elements = generate_elements(node_coordinates, element_node_connectivity)
    plot_mesh_3d(elements, node_coordinates)
