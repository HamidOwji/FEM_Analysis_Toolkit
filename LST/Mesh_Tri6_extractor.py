import h5py
import numpy as np
import FEM_LST_plotting

def divide_list_into_sublists(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_coordinates(lst, number_of_coordinates):
    for i in range(0, len(lst), number_of_coordinates):
        yield lst[i:i + number_of_coordinates]

def generate_elements_for_stiffness(node_coordinates, element_node_connectivity):
    elements = []
    for element in element_node_connectivity:
        element_dict = {
            'nodes': element,
            'coords': np.array([node_coordinates[node - 1] for node in element])
        }
        elements.append(element_dict)
    return elements

def generate_elements_for_plot(node_coordinates, element_node_connectivity):
    elements = []
    for element in element_node_connectivity:
        reordered_element = [element[0], element[3], element[1], element[4], element[2], element[5]]
        element_dict = {
            'nodes': reordered_element,
            'coords': np.array([node_coordinates[node - 1] for node in reordered_element])
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

        # Read TRIA6/NOD dataset for TRIA6 elements
        tr6_dataset = file['ENS_MAA/Mesh_2/-0000000000000000001-0000000000000000001/MAI/TR6/NOD']
        tr6_data = tr6_dataset[:]
        num_triangles = len(tr6_data) // 6
        sublists = list(divide_list_into_sublists(tr6_data, num_triangles))
        element_node_connectivity = [group for group in zip(*sublists)]

    # node_coordinates=  [(0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, 0.0), (1.5, 0.5), (0.5, 0.5)]
    # element_node_connectivity = [(1, 2, 3, 4, 5, 6)]
    # elements=  [{'nodes': (1, 2, 3, 4, 5, 6), 'coords': ([[ 0.,  0.], [ 2., 0.], [1., 1.], [1., 0.], [1.5, 0.5], [0.5, 0.5]])}]

    return node_coordinates, element_node_connectivity

if __name__ == "__main__":
    node_coordinates, element_node_connectivity = read_mesh_data('Mesh_2.med')
    elements = generate_elements_for_plot(node_coordinates, element_node_connectivity)
    FEM_LST_plotting.plot_mesh(elements, node_coordinates)
