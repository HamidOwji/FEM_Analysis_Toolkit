import numpy as np
from Mesh_Tri3_extractor import generate_elements, read_mesh_data
from FEM_CST_plotting import (plot_mesh, plot_displacements, plot_mesh_with_boundary_conditions, plot_loads, plot_mesh_with_loads)

# Role of B Matrix: The B matrix effectively bridges the gap between the physical displacements of the nodes and the strains in the material.
# By including the B matrix in the stiffness calculation, the stiffness matrix is able to capture how the material's internal stresses are related to nodal displacements.

def compute_area_and_B_matrix(coords):
    """Compute the area and B matrix for a CST element."""
    A = 0.5 * abs(coords[0][0]*(coords[1][1]-coords[2][1]) +
                  coords[1][0]*(coords[2][1]-coords[0][1]) +
                  coords[2][0]*(coords[0][1]-coords[1][1]))

    b = np.array([coords[1][1]-coords[2][1], coords[2][1]-coords[0][1], coords[0][1]-coords[1][1]])
    c = np.array([coords[2][0]-coords[1][0], coords[0][0]-coords[2][0], coords[1][0]-coords[0][0]])

    B = np.zeros((3, 6))
    B[0, ::2] = b
    B[1, 1::2] = c
    B[2, ::2] = c
    B[2, 1::2] = b
    B /= (2*A)

    return A, B

def compute_D_matrix(E, nu):
    """Compute the D matrix (material matrix)."""
    return E / (1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

# The elemental stiffness matrix is calculated using the formula k = A * Bᵀ * D * B, where
def compute_stiffness_matrix(coords, D):
    """Compute the stiffness matrix for a CST element."""
    A, B = compute_area_and_B_matrix(coords)
    return A * np.dot(B.T, np.dot(D, B))

def assemble_global_stiffness(elements, D):
    """Assemble the global stiffness matrix."""
    num_nodes = max([node for elem in elements for node in elem['nodes']])
    K_global = np.zeros((2*num_nodes, 2*num_nodes))

    for elem in elements:
        k = compute_stiffness_matrix(elem['coords'], D)
        for i in range(3):
            for j in range(3):
                m, n = elem['nodes'][i], elem['nodes'][j]
                K_global[2*m-2:2*m, 2*n-2:2*n] += k[2*i:2*i+2, 2*j:2*j+2]

    return K_global

def compute_global_forces(K_global, U):
    """Compute the global forces from the global stiffness matrix and nodal displacements."""
    return np.dot(K_global, U)

def compute_element_forces(element, U, D):
    """Compute the element forces from the element stiffness matrix and nodal displacements."""
    k = compute_stiffness_matrix(element['coords'], D)
    u_element = np.array([U[2*node-2:2*node] for node in element['nodes']]).flatten()
    return np.dot(k, u_element)

def compute_element_stress(element, U, D):
    """Compute the stress within an element."""
    u_element = np.array([U[2*node-2:2*node] for node in element['nodes']]).flatten()
    A, B = compute_area_and_B_matrix(element['coords'])
    epsilon = np.dot(B, u_element)
    sigma = np.dot(D, epsilon)
    return sigma

def compute_principal_stresses_and_angles(sigma):
    """Compute the principal stresses from the stress vector."""
    sigma_x, sigma_y, tau_xy = sigma
    sigma_avg = 0.5 * (sigma_x + sigma_y)
    R = np.sqrt(((sigma_x - sigma_y) * 0.5)**2 + tau_xy**2)
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R
    theta_rad = 0.5 * np.arctan2(2 * tau_xy, sigma_x - sigma_y)
    theta_deg = np.degrees(theta_rad)
    return sigma_1, sigma_2, theta_deg

# Example usage
E = 2.1e6  # Modulus of elasticity in Pa (for steel)
nu = 0.3  # Poisson's ratio (for steel)
D = compute_D_matrix(E, nu)  # Compute D matrix once and reuse

node_coordinates, element_node_connectivity = read_mesh_data('Mesh_1.med')
elements = generate_elements(node_coordinates, element_node_connectivity)

K_global = assemble_global_stiffness(elements, D)
# print(K_global)

# Identify boundary nodes and apply conditions. This gives us index not node number
left_boundary_nodes = [node + 1 for node, coord in enumerate(node_coordinates) if coord[0] == 0]
# print('left boundary: ', left_boundary_nodes)

# Initialize the displacement vector with zeros (as a starting assumption)
U = np.zeros(2 * len(node_coordinates))

# Identify nodes with x=140 and apply a load of 1000 in the x direction
F_external = np.zeros_like(U)

# Node index where the load will be applied (Python uses 0-based indexing, so node 3 is indexed as 2)
node_index = 3 - 1 # Adjust for 0-based indexing by subtracting 1

# Apply a load of 1000 in the x direction to node 3
F_external[2*node_index + 1] = -1000  # Apply load in x direction. For y direction, use 2*node_index + 1

# print('f external: ', F_external)

fixed_dof = []
for node in left_boundary_nodes:  # assuming left_boundary_nodes contain fixed nodes
    fixed_dof.extend([2*(node -1), 2*(node -1) +1])
    
K_reduced = np.delete(K_global, fixed_dof, axis=0)  # Remove rows
K_reduced = np.delete(K_reduced, fixed_dof, axis=1)  # Remove columns
F_reduced = np.delete(F_external, fixed_dof)
# print('f reduced: ', F_reduced)
U_reduced = np.linalg.solve(K_reduced, F_reduced)
U_full = np.zeros_like(U)
free_dof = set(range(len(U))) - set(fixed_dof)
free_dof = list(free_dof)
U_full[free_dof] = U_reduced


F_global_calculated = compute_global_forces(K_global, U_full)

# Uncomment the following lines for debugging:
# print("Global Forces:")
# print(F_global_calculated)

# Uncomment the following blocks for debugging:
# for elem in elements:
#     F_element = compute_element_forces(elem, U_full, D)  
#     print(f"Element Forces for nodes {elem['nodes']}:")
#     print(F_element)

# for elem in elements:
#     sigma_element = compute_element_stress(elem, U_full, D)  
#     print(f"Element Stress for nodes {elem['nodes']}:")
#     print(sigma_element)

# for elem in elements:
#     sigma_element = compute_element_stress(elem, U_full, D) 
#     sigma_1, sigma_2, theta_deg = compute_principal_stresses_and_angles(sigma_element)
#     print(f"Principal Stresses for element with nodes {elem['nodes']}:")
#     print(f"Maximum (sigma_1): {sigma_1}")
#     print(f"Minimum (sigma_2): {sigma_2}")
#     print(f"Angle of Maximum Stress (degrees): {theta_deg}")

# for i, coord in enumerate(node_coordinates):
#     x, y = coord  # Unpack the x and y coordinates of the node
#     dx = U_full[2*i]   # x displacement of node i
#     dy = U_full[2*i+1] # y displacement of node i
#     print(f"Node {i+1}: x = {x:.6f}, y = {y:.6f}, dx = {dx:.6f}, dy = {dy:.6f}")

# print('max dis= ', max(U_full))
plot_displacements(node_coordinates, U_full, 'Nodal Displacements', scale_factor=500)
# print(U_full)
# plot_mesh(elements, node_coordinates)
# plot_displacements(node_coordinates, U, 'stress')


# plot_mesh_with_boundary_conditions(elements, node_coordinates, left_boundary_nodes)
# plot_loads(node_coordinates, F_external, 1)
# plot_mesh_with_loads(elements, node_coordinates, left_boundary_nodes, F_external)

# Assuming U_full is already defined and contains the displacements for each node
displacement_magnitudes = np.sqrt(U_full[::2]**2 + U_full[1::2]**2)

max_disp_node_index = np.argmax(displacement_magnitudes)

max_disp_magnitude = displacement_magnitudes[max_disp_node_index]

max_disp_node_number = max_disp_node_index + 1

max_disp_x = U_full[2*max_disp_node_index]
max_disp_y = U_full[2*max_disp_node_index + 1]

# Print the information
print(f"Node with Maximum Displacement: Node {max_disp_node_number}")
print(f"Displacement in X: {max_disp_x:.6f}")
print(f"Displacement in Y: {max_disp_y:.6f}")
print(f"Total Displacement Magnitude: {max_disp_magnitude:.6f}")
