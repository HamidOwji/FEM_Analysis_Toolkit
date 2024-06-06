from Mesh_Tri6_extractor import ( generate_elements_for_stiffness,
                                read_mesh_data, plot_mesh,
                                plot_displacements,
                                plot_mesh_with_boundary_conditions,
                                plot_loads,
                                plot_mesh_with_loads)
from sympy import symbols, diff, Matrix, lambdify, Rational
import numpy as np
import FEM_LST_plotting

# Define symbolic variables for coordinates
x, y = symbols('x y')
x1, y1, x2, y2, x3, y3 = symbols('x1 y1 x2 y2 x3 y3')

# Calculate the centroid (x_c, y_c) of the triangle formed by the first three nodes
x_c = Rational(1, 3) * (x1 + x2 + x3)
y_c = Rational(1, 3) * (y1 + y2 + y3)

# Placeholder for symbolic shape functions of an LST element
# You need to replace these with actual shape function expressions
L1, L2, L3, L4, L5, L6 = symbols('L1 L2 L3 L4 L5 L6')  # Placeholder

# Assume L1 to L6 are already defined here
L1 = ((x - x3)*(y2 - y3) - (x2 - x3)*(y - y3))/(x1*(y2 - y3) - 1.0*x2*(y1 - y3) + x3*(y1 - y2))
L2 = (-(x - x3)*(y1 - y3) + (x1 - x3)*(y - y3))/(x1*(y2 - y3) - 1.0*x2*(y1 - y3) + x3*(y1 - y2))
L3 = 1.0*(x*y1 - x*y2 - x1*y + x1*y2 + x2*y - x2*y1)/(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)
L4 = -4*((x - x3)*(y1 - y3) - (x1 - x3)*(y - y3))*((x - x3)*(y2 - y3) - (x2 - x3)*(y - y3))/(x1*(y2 - y3) - 1.0*x2*(y1 - y3) + x3*(y1 - y2))**2
L5 = -4.0*((x - x3)*(y1 - y3) - (x1 - x3)*(y - y3))*(x*y1 - x*y2 - x1*y + x1*y2 + x2*y - x2*y1)/((x1*(y2 - y3) - 1.0*x2*(y1 - y3) + x3*(y1 - y2))*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2))
L6 = 4.0*((x - x3)*(y2 - y3) - (x2 - x3)*(y - y3))*(x*y1 - x*y2 - x1*y + x1*y2 + x2*y - x2*y1)/((x1*(y2 - y3) - 1.0*x2*(y1 - y3) + x3*(y1 - y2))*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2))

# Symbolic derivatives of shape functions with respect to x and y
# Replace the content of diff() with actual differentiation of L1 to L6 with respect to x and y
dL_dx = [diff(L, x) for L in [L1, L2, L3, L4, L5, L6]]
dL_dy = [diff(L, y) for L in [L1, L2, L3, L4, L5, L6]]

# Assembly of the B matrix symbolically
B = Matrix(3, 12, lambda i, j: 0)  # Initialize a 3x12 matrix filled with zeros

for i in range(6):
    B[0, 2*i] = dL_dx[i]  # dL/dx contributes to εx
    B[1, 2*i + 1] = dL_dy[i]  # dL/dy contributes to εy
    B[2, 2*i] = dL_dy[i]  # dL/dy also contributes to γxy
    B[2, 2*i + 1] = dL_dx[i]  # dL/dx also contributes to γxy

# Convert the symbolic B matrix to a function that can evaluate it for given nodal coordinates
B_func = lambdify((x1, y1, x2, y2, x3, y3, x, y), B, 'numpy')

def compute_area_and_B_matrix(coords):
    """
    Compute the B matrix for an LST element given its nodal coordinates.
    
    coords: A list of tuples, where each tuple represents the (x, y) coordinates of a node.
            The nodes should be ordered as: three corner nodes followed by three mid-side nodes.
    """
    A = 0.5 * abs(coords[0][0]*(coords[1][1]-coords[2][1]) +
                  coords[1][0]*(coords[2][1]-coords[0][1]) +
                  coords[2][0]*(coords[0][1]-coords[1][1]))
    
    # Calculate the centroid (middle point) of the triangle
    x_c = Rational(1, 3) * sum(coord[0] for coord in coords)
    y_c = Rational(1, 3) * sum(coord[1] for coord in coords)

    # Flatten the list of coordinates for input to B_func
    # coords = [(1, 2), (3, 4), (5, 6)] => coords_flat = [1, 2, 3, 4, 5, 6]
    coords_flat = [val for coord in coords for val in coord]
    
    # Evaluate the B matrix for the given coordinates
    B = B_func(*coords_flat, x_c, y_c)
    # print('A: ', A)
    # print('B: ', B)
    return A, B

def compute_D_matrix(E, nu):
    """Compute the D matrix (material matrix)."""
    return E / (1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

# The elemental stiffness matrix is calculated using the formula k = A * Bᵀ * D * B
def compute_stiffness_matrix(coords, D):
    """Compute the stiffness matrix for a CST element."""
    # print('coords', coords[0:3])
    # print('D', D)
    A, B = compute_area_and_B_matrix((coords[0], coords[2], coords[4]))
    return A * np.dot(B.T, np.dot(D, B))

def assemble_global_stiffness(elements, D):
    """Assemble the global stiffness matrix."""
    num_nodes = max([node for elem in elements for node in elem['nodes']])
    K_global = np.zeros((2*num_nodes, 2*num_nodes))

    for elem in elements:
        k = compute_stiffness_matrix(elem['coords'], D)
        k = np.asarray(k, dtype=np.float64) # to convert object comes from symbolic calculations to numerical dataset
        # print('elem[coords] : ', elem['coords'])
        # print('elem[nodes] : ', elem['nodes'])
        # print('k', k)
        for i in range(6):
            for j in range(6):
                m, n = elem['nodes'][i], elem['nodes'][j]
                K_global[2*m-2:2*m, 2*n-2:2*n] += k[2*i:2*i+2, 2*j:2*j+2]
    # print(K_global)
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

node_coordinates, element_node_connectivity = read_mesh_data('Mesh_2.med')
elements = generate_elements_for_stiffness(node_coordinates, element_node_connectivity)

K_global = assemble_global_stiffness(elements, D)
# print(K_global)

# Identify boundary nodes and apply conditions. This gives us index not node number
left_boundary_nodes = [node + 1 for node, coord in enumerate(node_coordinates) if coord[0] == 0]
print('left boundary: ', left_boundary_nodes)

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
FEM_LST_plotting.plot_displacements(node_coordinates, U_full, 'Nodal Displacements', scale_factor=1000)
# print(U_full)
# plot_mesh(elements, node_coordinates)
# plot_displacements(node_coordinates, U, 'stress')


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
