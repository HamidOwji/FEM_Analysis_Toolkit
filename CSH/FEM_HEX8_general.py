import numpy as np
from Mesh_HEX8_extractor import generate_elements, read_mesh_data
import FEM_HEX8_plotting  # Import the new FEM_HEX8_FEM_HEX8_plotting module
from Stiffness_CSH import compute_stiffness_matrix_for_8node_element, compute_B_matrix

# Material properties (E: Young's modulus, nu: Poisson's ratio)
E = 210e4  # Example: 210 GPa for steel
nu = 0.3

def assemble_global_stiffness(elements, E, nu):
    """Assemble the global stiffness matrix for 3D."""
    num_nodes = max([node for elem in elements for node in elem['nodes']])
    K_global = np.zeros((3*num_nodes, 3*num_nodes))  # Adjusted for 3 DoFs per node

    for elem_idx, elem in enumerate(elements):
        k = compute_stiffness_matrix_for_8node_element(elem['coords'], E, nu)
        for i in range(8):
            for j in range(8):
                m, n = elem['nodes'][i], elem['nodes'][j]
                K_global[3*m-3:3*m, 3*n-3:3*n] += k[3*i:3*i+3, 3*j:3*j+3]  # Adjusted for 3 DoFs

    return K_global

def compute_global_forces(K_global, U):
    """Compute the global forces from the global stiffness matrix and nodal displacements."""
    return np.dot(K_global, U)

def compute_element_forces(element, U, D):
    """Compute the element forces for a 3D hexahedral element."""
    k = compute_stiffness_matrix_for_8node_element(element['coords'], D)
    u_element = np.array([U[3*node-3:3*node] for node in element['nodes']]).flatten()
    return np.dot(k, u_element)

def compute_element_stress(element, U, D):
    """
    Compute the stress within a 3D hexahedral element.
    """
    sigma = np.zeros(6)
    sqrt3 = np.sqrt(3)
    gauss_points = [(-1/sqrt3, -1/sqrt3, -1/sqrt3), (1/sqrt3, -1/sqrt3, -1/sqrt3),
                    (-1/sqrt3, 1/sqrt3, -1/sqrt3), (1/sqrt3, 1/sqrt3, -1/sqrt3),
                    (-1/sqrt3, -1/sqrt3, 1/sqrt3), (1/sqrt3, -1/sqrt3, 1/sqrt3),
                    (-1/sqrt3, 1/sqrt3, 1/sqrt3), (1/sqrt3, 1/sqrt3, 1/sqrt3)]
    weights = [1, 1, 1, 1, 1, 1, 1, 1]

    total_weight = 0
    
    for gp, weight in zip(gauss_points, weights):
        xi, eta, zeta = gp
        B, det_J = compute_B_matrix(element['coords'], xi, eta, zeta)
        u_element = np.array([U[3*node-3:3*node] for node in element['nodes']]).flatten()
        epsilon = np.dot(B, u_element)
        local_sigma = np.dot(D, epsilon)

        sigma += local_sigma * weight * det_J
        total_weight += weight * det_J

    sigma /= total_weight

    return sigma

def compute_principal_stresses(sigma):
    """
    Compute the principal stresses and their directions from the 3D stress state.
    """
    stress_tensor = np.array([
        [sigma[0], sigma[3], sigma[5]],  # [sigma_x, tau_xy, tau_xz]
        [sigma[3], sigma[1], sigma[4]],  # [tau_xy, sigma_y, tau_yz]
        [sigma[5], sigma[4], sigma[2]]   # [tau_xz, tau_yz, sigma_z]
    ])
    
    principal_stresses, principal_directions = np.linalg.eig(stress_tensor)

    return principal_stresses, principal_directions

node_coordinates, element_node_connectivity = read_mesh_data('Mesh_1.med')
elements = generate_elements(node_coordinates, element_node_connectivity)

K_global = assemble_global_stiffness(elements, E, nu)

left_boundary_nodes = [i for i, coord in enumerate(node_coordinates) if coord[0] == 0]

U = np.zeros(3 * len(node_coordinates))

for node in left_boundary_nodes:
    U[3*node:3*node+3] = 0

fixed_dofs = []
for node in left_boundary_nodes:
    fixed_dofs.extend([3*node, 3*node+1, 3*node+2])

F_external = np.zeros_like(U)

node_nums = [18, 24]
force_z = -1000

for node_num in node_nums:
    node_index = node_num - 1
    F_external[3*node_index + 2] = force_z

K_reduced = np.delete(K_global, fixed_dofs, axis=0)
K_reduced = np.delete(K_reduced, fixed_dofs, axis=1)
F_reduced = np.delete(F_external, fixed_dofs)
U_reduced = np.linalg.solve(K_reduced, F_reduced)
U_full = np.zeros_like(U)
free_dof = set(range(len(U))) - set(fixed_dofs)
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
#     x, y = coord
#     dx = U_full[2*i]
#     dy = U_full[2*i+1]
#     print(f"Node {i+1}: x = {x:.6f}, y = {y:.6f}, dx = {dx:.6f}, dy = {dy:.6f}")

# print('max dis= ', max(U_full))

FEM_HEX8_plotting.plot_mesh_3d(elements, node_coordinates)
# FEM_HEX8_plotting.plot_mesh_with_conditions_forces(elements, node_coordinates)
FEM_HEX8_plotting.plot_mesh_with_conditions_forces_and_deformation(elements, node_coordinates, left_boundary_nodes, F_external, U_full, deformation_scale=2000)

displacement_magnitudes = np.sqrt(U_full[::3]**2 + U_full[1::3]**2 + U_full[2::3]**2)

max_disp_node_index = np.argmax(displacement_magnitudes)

max_disp_magnitude = displacement_magnitudes[max_disp_node_index]

max_disp_node_number = max_disp_node_index + 1

max_disp_x = U_full[3*max_disp_node_index]
max_disp_y = U_full[3*max_disp_node_index + 1]
max_disp_z = U_full[3*max_disp_node_index + 2]

print(f"Node with Maximum Displacement: Node {max_disp_node_number}")
print(f"Displacement in X: {max_disp_x:.6f}")
print(f"Displacement in Y: {max_disp_y:.6f}")
print(f"Displacement in Z: {max_disp_z:.6f}")
print(f"Total Displacement Magnitude: {max_disp_magnitude:.6f}")
