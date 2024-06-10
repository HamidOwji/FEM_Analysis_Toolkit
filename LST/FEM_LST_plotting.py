import matplotlib.pyplot as plt
import numpy as np

def plot_mesh(elements, node_coordinates):
    plt.figure(figsize=(10, 10))
    for element in elements:
        x_coords, y_coords = zip(*[node_coordinates[node-1] for node in element['nodes']])
        x_coords += (x_coords[0],)
        y_coords += (y_coords[0],)
        plt.plot(x_coords, y_coords, color="blue")
        
        # Plot node numbers
        for node in element['nodes']:
            node_x, node_y = node_coordinates[node-1]
            plt.text(node_x, node_y, str(node), color="gray", fontsize=8)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Element Mesh Plot')
    # plt.savefig('plot.png')
    plt.show()

def plot_displacements(node_coordinates, U, title, scale_factor=10000):
    plt.figure(figsize=(10, 10))
    if title:
        plt.title(title)
    
    for i, coord in enumerate(node_coordinates):
        displacement = U[2*i:2*i+2] * scale_factor  # Apply scale factor here
        plt.quiver(coord[0], coord[1], displacement[0], displacement[1], angles='xy', scale_units='xy', scale=100, color='r', label=f'Node {i+1}' if i == 0 else "")
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def plot_mesh_with_boundary_conditions(elements, node_coordinates, boundary_nodes, title='Element Mesh with Boundary Conditions'):
    plt.figure(figsize=(10, 10))
    for element in elements:
        x_coords, y_coords = zip(*[node_coordinates[node-1] for node in element['nodes']])
        x_coords += (x_coords[0],)
        y_coords += (y_coords[0],)
        plt.plot(x_coords, y_coords, color="blue")
        
        # Plot node numbers or boundary condition symbols
        for node in element['nodes']:
            node_x, node_y = node_coordinates[node-1]
            if node in boundary_nodes:
                plt.scatter(node_x, node_y, color="red", label="Fixed Node" if node == boundary_nodes[0] else "", marker='s')  # Mark boundary nodes
            plt.text(node_x, node_y, str(node), color="gray", fontsize=8)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def plot_loads(node_coordinates, loads, scale_factor=1, title='Loads on the Mesh'):
    plt.figure(figsize=(10, 10))
    max_load = np.max(np.abs(loads))
    for i, load in enumerate(loads):
        if not np.isclose(load, 0):  # Check if load is non-zero
            x, y = node_coordinates[i//2]
            if i % 2 == 0:  # X direction
                plt.quiver(x, y, load * scale_factor / max_load, 0, angles='xy', scale_units='xy', scale=1, color='green', width=0.003)
            else:  # Y direction
                plt.quiver(x, y, 0, load * scale_factor / max_load, angles='xy', scale_units='xy', scale=1, color='green', width=0.003)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.grid()
    plt.show()

def plot_mesh_with_loads(elements, node_coordinates, boundary_nodes, loads, title='Mesh with Loads and Boundary Conditions', scale_factor=10):
    plt.figure(figsize=(10, 10))
    max_load = np.max(np.abs(loads)) if np.max(np.abs(loads)) > 0 else 1  # Prevent division by zero

    # Plot elements and nodes
    for element in elements:
        x_coords, y_coords = zip(*[node_coordinates[node-1] for node in element['nodes']])
        x_coords += (x_coords[0],)
        y_coords += (y_coords[0],)
        plt.plot(x_coords, y_coords, color="blue")
        
        # Plot node numbers or boundary condition symbols
        for node in element['nodes']:
            node_x, node_y = node_coordinates[node-1]
            if node in boundary_nodes:
                plt.scatter(node_x, node_y, color="red", label="Fixed Node" if node == boundary_nodes[0] else "", marker='s')  # Mark boundary nodes
            plt.text(node_x, node_y, str(node), color="gray", fontsize=8)

    # Plot loads
    for i, load in enumerate(loads):
        if not np.isclose(load, 0):  # Check if load is non-zero
            node_index = i // 2
            x, y = node_coordinates[node_index]
            if i % 2 == 0:  # Load in X direction
                plt.quiver(x, y, load * scale_factor / max_load, 0, angles='xy', scale_units='xy', scale=1, color='red', width=0.005, headwidth=3, label='Load' if i == 0 else "")
            else:  # Load in Y direction
                plt.quiver(x, y, 0, load * scale_factor / max_load, angles='xy', scale_units='xy', scale=1, color='red', width=0.005, headwidth=3)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
