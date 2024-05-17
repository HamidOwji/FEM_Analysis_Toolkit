import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def divide_list_into_sublists(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_coordinates(lst, number_of_coordinates):
    for i in range(0, len(lst), number_of_coordinates):
        yield lst[i:i + number_of_coordinates]

def generate_elements_for_stiffness(node_coordinates, element_node_connectivity):
    elements = []
    for element in element_node_connectivity:
        # reordered_element = [element[0], element[4], element[1], element[5], element[2], element[6], element[3], element[7]]
        element_dict = {
            'nodes': element,
            'coords': np.array([node_coordinates[node - 1] for node in element])
        }
        elements.append(element_dict)
        # print(element_dict)
    return elements

def generate_elements_for_plot(node_coordinates, element_node_connectivity):
    elements = []
    for element in element_node_connectivity:
        reordered_element = [element[0], element[4], element[1], element[5], element[2], element[6], element[3], element[7]]
        element_dict = {
            'nodes': reordered_element,
            'coords': np.array([node_coordinates[node - 1] for node in reordered_element])
        }
        elements.append(element_dict)
        # print(element_dict)
    return elements


def read_mesh_data(file_name):
    with h5py.File(file_name, 'r') as file:
        # Read coordinate data
        coo_dataset = file['ENS_MAA/Mesh_3/-0000000000000000001-0000000000000000001/NOE/COO']
        coo_data = coo_dataset[:]
        num_nodes = len(coo_data) // 2
        subcoord = list(extract_coordinates(coo_data, num_nodes))
        node_coordinates = [group for group in zip(*subcoord)]

        # Accessing QU8 element connectivity
        qu8_dataset = file['ENS_MAA/Mesh_3/-0000000000000000001-0000000000000000001/MAI/QU8/NOD']
        qu8_data = qu8_dataset[:]
        num_quadrilaterals = len(qu8_data) // 8
        sublists = list(divide_list_into_sublists(qu8_data, num_quadrilaterals))
        element_node_connectivity = [group for group in zip(*sublists)]

    return node_coordinates, element_node_connectivity

def plot_mesh(elements, node_coordinates):
    plt.figure(figsize=(10, 10))
    for element in elements:
        try:
            # Ensure all node indices are within the valid range for node_coordinates
            if all(node-1 < len(node_coordinates) and node-1 >= 0 for node in element['nodes']):
                x_coords, y_coords = zip(*[node_coordinates[node-1] for node in element['nodes']])
                x_coords += (x_coords[0],)  # Close the loop for plotting
                y_coords += (y_coords[0],)
                plt.plot(x_coords, y_coords, color="blue")
                for node in element['nodes']:
                    node_x, node_y = node_coordinates[node-1]
                    plt.text(node_x, node_y, str(node), color="red", fontsize=8)
            else:
                print(f"Element {element['nodes']} contains out-of-range node indices.")
        except Exception as e:
            print(f"Error plotting element {element['nodes']}: {e}")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Element Mesh Plot')
    plt.axis('equal')
    plt.show()



def plot_displacements(node_coordinates, U, title, scale_factor=1):
    """Plot nodal displacements or stress on the mesh with a scaling factor."""
    plt.figure(figsize=(10, 10))
    if title:
        plt.title(title)
    
    for i, coord in enumerate(node_coordinates):
        displacement = U[2*i:2*i+2] * scale_factor  # Apply scale factor here
        plt.quiver(coord[0], coord[1], displacement[0], displacement[1], angles='xy', scale_units='xy', scale=1, color='r', label=f'Node {i+1}' if i == 0 else "")
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# This function should be called from solver
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

# This function should be called from solver
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


# This function should be called from solver
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



if __name__ == "__main__":
    node_coordinates, element_node_connectivity = read_mesh_data('Mesh_3.med')
    elements = generate_elements_for_plot(node_coordinates, element_node_connectivity)
    plot_mesh(elements, node_coordinates)
