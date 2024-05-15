import h5py
import numpy as np

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
        coo_dataset = file['ENS_MAA/Mesh_1/-0000000000000000001-0000000000000000001/NOE/COO']
        coo_data = coo_dataset[:]
        num_nodes = len(coo_data) // 2
        subcoord = list(extract_coordinates(coo_data, num_nodes))
        node_coordinates = [group for group in zip(*subcoord)]

        # Read TRIA3/NOD dataset for TRIA3 elements
        tr3_dataset = file['ENS_MAA/Mesh_1/-0000000000000000001-0000000000000000001/MAI/TR3/NOD']
        tr3_data = tr3_dataset[:]
        num_triangles = len(tr3_data) // 3
        sublists = list(divide_list_into_sublists(tr3_data, num_triangles))
        element_node_connectivity = [group for group in zip(*sublists)]

    return node_coordinates, element_node_connectivity
