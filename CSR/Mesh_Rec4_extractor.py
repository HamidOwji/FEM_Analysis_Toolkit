import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import FEM_CSR_plotting  # Import the new plotting module

def divide_list_into_sublists(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_coordinates(lst, number_of_coordinates):
    for i in range(0, len(lst), number_of_coordinates):
        yield lst[i:i + number_of_coordinates]

def generate_elements(node_coordinates, element_node_connectivity):
    elements = []
    for element in element_node_connectivity:
        element_dict = {
            'nodes': element,
            'coords': np.array([node_coordinates[node - 1] for node in element])
        }
        elements.append(element_dict)
    return elements

def read_mesh_data(file_name):
    with h5py.File(file_name, 'r') as file:
        # Read coordinate data
        coo_dataset = file['ENS_MAA/Mesh_2/-0000000000000000001-0000000000000000001/NOE/COO']
        coo_data = coo_dataset[:]
        num_nodes = len(coo_data) // 2
        subcoord = list(extract_coordinates(coo_data, num_nodes))
        node_coordinates = [group for group in zip(*subcoord)]

        # Read QU4/NOD dataset for QU4 elements
        qu4_dataset = file['ENS_MAA/Mesh_2/-0000000000000000001-0000000000000000001/MAI/QU4/NOD']
        qu4_data = qu4_dataset[:]
        num_quadrilaterals = len(qu4_data) // 4
        sublists = list(divide_list_into_sublists(qu4_data, num_quadrilaterals))
        element_node_connectivity = [group for group in zip(*sublists)]

    return node_coordinates, element_node_connectivity

if __name__ == "__main__":
    node_coordinates, element_node_connectivity = read_mesh_data('Mesh_2.med')
    elements = generate_elements(node_coordinates, element_node_connectivity)
    FEM_CSR_plotting.plot_mesh(elements, node_coordinates)
