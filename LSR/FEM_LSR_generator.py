import numpy as np
from FEM_Analysis_Toolkit.LSR.Mesh_Rec8_extractor import ( generate_elements_for_stiffness,
                                 generate_elements_for_plot,
                                read_mesh_data, plot_mesh,
                                plot_displacements,
                                plot_mesh_with_boundary_conditions,
                                plot_loads,
                                plot_mesh_with_loads)

from Stiffness_LSR import compute_stiffness_matrix_for_8node_element, compute_B_matrix_for_8node_element

# Material properties (E: Young's modulus, nu: Poisson's ratio)
E = 210e4  # Example: 210 GPa for steel
nu = 0.3
t = 1  # Thickness of the element, assuming 1 cm for a 2D plane stress problem

def assemble_global_stiffness(elements, E, nu, t):
    """Assemble the global stiffness matrix."""
    num_nodes = max([node for elem in elements for node in elem['nodes']])
    K_global = np.zeros((2*num_nodes, 2*num_nodes))

    for elem in elements:
        k = compute_stiffness_matrix_for_8node_element(elem['coords'], E, nu, t)
        for i in range(8):
            for j in range(8):
                m, n = elem['nodes'][i], elem['nodes'][j]
                K_global[2*m-2:2*m, 2*n-2:2*n] += k[2*i:2*i+2, 2*j:2*j+2]

    return K_global

def compute_global_forces(K_global, U):
    """Compute the global forces from the global stiffness matrix and nodal displacements."""
    return np.dot(K_global, U)

def compute_element_forces(element, U, D):
    """Compute the element forces from the element stiffness matrix and nodal displacements."""
    k = compute_stiffness_matrix_for_8node_element(element['coords'], D)
    u_element = np.array([U[2*node-2:2*node] for node in element['nodes']]).flatten()
    return np.dot(k, u_element)

def compute_element_stress(element, U, D, t):
    """
    Compute the stress within a rectangular element.
    
    Parameters:
    - element: Dictionary containing element information ('coords' and 'nodes').
    - U: Global displacement vector.
    - D: Material elasticity matrix.
    - t: Thickness of the element.
    
    Returns:
    - sigma: Stress vector (sigma_x, sigma_y, tau_xy) averaged over the element.
    """
    # Initialize the stress components
    sigma_x = sigma_y = tau_xy = 0
    
    # Gauss quadrature points for integration over the element
    gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                    (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
    weights = [1, 1, 1, 1]
    
    # Extract nodal displacements for the element
    u_element = np.array([U[2*node-2:2*node] for node in element['nodes']]).flatten()

    # Loop over Gauss points to integrate stress over the element
    for gp, weight in zip(gauss_points, weights):
        B = compute_B_matrix_for_8node_element(element['coords'], gp)
        epsilon = np.dot(B, u_element)  # Strain vector at this Gauss point
        sigma = np.dot(D, epsilon)  # Stress vector at this Gauss point
        
        # Accumulate weighted stress components
        sigma_x += sigma[0] * weight
        sigma_y += sigma[1] * weight
        tau_xy += sigma[2] * weight

    # Average the stress components over the Gauss points
    sigma_x /= len(gauss_points)
    sigma_y /= len(gauss_points)
    tau_xy /= len(gauss_points)

    return np.array([sigma_x, sigma_y, tau_xy])

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

node_coordinates, element_node_connectivity = read_mesh_data('Mesh_3.med')
elements_for_stiffness = generate_elements_for_stiffness(node_coordinates, element_node_connectivity)
elements_for_plot = generate_elements_for_plot(node_coordinates, element_node_connectivity)

K_global = assemble_global_stiffness(elements_for_stiffness, E, nu, t)
# print(K_global)

# Identify boundary nodes and apply conditions. This gives us index not node number
left_boundary_nodes = [node + 1 for node, coord in enumerate(node_coordinates) if coord[0] == 0]
# print('left boundary: ', left_boundary_nodes)

# Initialize the displacement vector with zeros (as a starting assumption)
U = np.zeros(2 * len(node_coordinates))
# for distributed load
# #######################
# # Define and apply loads (Example: Vertical load at right boundary)
# total_load = -1000  # Total load value in Newtons
# load_per_node = total_load / len(right_boundary_nodes)  # Distribute load evenly

# F_external = np.zeros_like(U)
# for node in right_boundary_nodes:
#     F_external[2*node-1] = load_per_node Applying distributed load in vertical direction
#########################

# Identify nodes with x=140 and apply a load of 1000 in the x direction
F_external = np.zeros_like(U)

# for node, coord in enumerate(node_coordinates):
#     if coord[0] == 140 and coord[1] == 0:  # Assuming the x-coordinate is the first element in coord
#         # F_external[2*node] = 1000  # Apply load in x direction
#         F_external[2*node - 2] = -1000 # F_external[2*(3)-2] = total_load in x direction for y direction F_external[2*(3)-1] = total_load

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


# Use U_full for further calculations
# print("Global Forces:")
F_global_calculated = compute_global_forces(K_global, U_full)
# print(F_global_calculated)

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

    # After obtaining U_full from solving the system of equations
# print("Node Displacements:")
# for i, coord in enumerate(node_coordinates):
#     x, y = coord  # Unpack the x and y coordinates of the node
#     dx = U_full[2*i]   # x displacement of node i
#     dy = U_full[2*i+1] # y displacement of node i
#     print(f"Node {i+1}: x = {x:.6f}, y = {y:.6f}, dx = {dx:.6f}, dy = {dy:.6f}")

# print('max dis= ', max(U_full))
# plot_displacements(node_coordinates, U_full, 'Nodal Displacements', scale_factor=1000)
# print(U_full)
plot_mesh(elements_for_plot, node_coordinates)
plot_displacements(node_coordinates, U, 'stress')


# plot_mesh_with_boundary_conditions(elements, node_coordinates, left_boundary_nodes)
# plot_loads(node_coordinates, F_external, 1)
# plot_mesh_with_loads(elements, node_coordinates, left_boundary_nodes, F_external)

# Assuming U_full is already defined and contains the displacements for each node
# Calculate displacement magnitude for each node
displacement_magnitudes = np.sqrt(U_full[::2]**2 + U_full[1::2]**2)

# Find the index of the node with the maximum displacement
max_disp_node_index = np.argmax(displacement_magnitudes)

# Calculate the maximum displacement magnitude
max_disp_magnitude = displacement_magnitudes[max_disp_node_index]

# Node number (assuming node indexing starts from 1)
max_disp_node_number = max_disp_node_index + 1

# Displacements in x and y directions for the node with maximum displacement
max_disp_x = U_full[2*max_disp_node_index]
max_disp_y = U_full[2*max_disp_node_index + 1]

# Print the information
print(f"Node with Maximum Displacement: Node {max_disp_node_number}")
print(f"Displacement in X: {max_disp_x:.6f}")
print(f"Displacement in Y: {max_disp_y:.6f}")
print(f"Total Displacement Magnitude: {max_disp_magnitude:.6f}")
