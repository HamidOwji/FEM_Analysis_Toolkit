import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import FEM_HEX8_plotting  # Import the new FEM_HEX8_plotting module

def divide_list_into_sublists(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_coordinates(lst, number_of_coordinates=3):
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
        coo_dataset = file['ENS_MAA/Mesh_1/-0000000000000000001-0000000000000000001/NOE/COO']
        coo_data = coo_dataset[:]
        num_nodes = len(coo_data) // 3
        subcoord = list(extract_coordinates(coo_data, num_nodes))
        node_coordinates = [group for group in zip(*subcoord)]

        he8_dataset = file['ENS_MAA/Mesh_1/-0000000000000000001-0000000000000000001/MAI/HE8/NOD']
        he8_data = he8_dataset[:]
        num_quadrilaterals = len(he8_data) // 8
        sublists = list(divide_list_into_sublists(he8_data, num_quadrilaterals))
        element_node_connectivity = [group for group in zip(*sublists)]
    return node_coordinates, element_node_connectivity

if __name__ == "__main__":
    node_coordinates, element_node_connectivity = read_mesh_data('Mesh_1.med')
    elements = generate_elements(node_coordinates, element_node_connectivity)
    FEM_HEX8_plotting.plot_mesh_3d(elements, node_coordinates)
