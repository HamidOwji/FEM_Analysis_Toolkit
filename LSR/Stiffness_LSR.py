import numpy as np


# Shape function derivatives in local coordinates (dN/dxi and dN/deta)
def shape_func_derivatives(xi, eta):

    return np.array([
        [ -0.25*(eta - 1)*(eta + 2*xi), -0.25*(2*eta + xi)*(xi - 1) ],
        [ 0.25*(eta - 1)*(eta - 2*xi), 0.25*(2*eta - xi)*(xi + 1)],
        [ 0.25*(eta + 1)*(eta + 2*xi), 0.25*(2*eta + xi)*(xi + 1)],
        [ 0.25*(-eta + 2*xi)*(eta + 1), 0.25*(-2*eta + xi)*(xi - 1)],
        [ 1.0*xi*(eta - 1), 0.5*xi**2 - 0.5],
        [ 0.5 - 0.5*eta**2, -1.0*eta*(xi + 1)],
        [ -1.0*xi*(eta + 1), 0.5 - 0.5*xi**2],
        [ 0.5*eta**2 - 0.5, eta*(xi - 1)]
    ])

def compute_jacobian(dN_dxi, dN_deta, coords):
    J = np.zeros((2, 2))
    for i in range(8):
        J[0, 0] += dN_dxi[i] * coords[i][0]  # dX/dxi
        J[0, 1] += dN_dxi[i] * coords[i][1]  # dY/dxi
        J[1, 0] += dN_deta[i] * coords[i][0]  # dX/deta
        J[1, 1] += dN_deta[i] * coords[i][1]  # dY/deta
    return J

def compute_B_matrix(dN_dxi, dN_deta, coords):
    J = compute_jacobian(dN_dxi, dN_deta, coords)
    J_inv = np.linalg.inv(J)
    det_J = np.linalg.det(J)

    B = np.zeros((3, 16))
    for i in range(8):
        dN_dx, dN_dy = np.dot(J_inv, [dN_dxi[i], dN_deta[i]])
        B[0, 2*i] = dN_dx  # Strain in x direction
        B[1, 2*i + 1] = dN_dy  # Strain in y direction
        B[2, 2*i] = dN_dy  # Shear strain
        B[2, 2*i + 1] = dN_dx
    return B, det_J

def compute_stiffness_matrix_for_8node_element(coords, E, nu, t):
    # Elasticity matrix for plane stress
    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    
    # Define Gauss points and weights for 2D integration
    gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                    (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
    weights = [1, 1, 1, 1]
    
    K = np.zeros((16, 16))  # Initialize stiffness matrix for 8 nodes with 2 DOFs each

    for gp, w in zip(gauss_points, weights):
        xi, eta = gp
        dN_dxi_eta = shape_func_derivatives(xi, eta)
        dN_dxi = dN_dxi_eta[:, 0]
        dN_deta = dN_dxi_eta[:, 1]
        
        B, det_J = compute_B_matrix(dN_dxi, dN_deta, coords)
        
        K += t * w * det_J * np.dot(B.T, np.dot(D, B))
    
    return K


def compute_B_matrix_for_8node_element(coords, xi, eta):
    # Compute derivatives of shape functions with respect to xi and eta
    dN_dxi_eta = shape_func_derivatives(xi, eta)
    dN_dxi = dN_dxi_eta[:, 0]
    dN_deta = dN_dxi_eta[:, 1]
    
    J = compute_jacobian(dN_dxi, dN_deta, coords)
    J_inv = np.linalg.inv(J)
    det_J = np.linalg.det(J)
    
    B = np.zeros((3, 16))  # Initialize B matrix for 8 nodes with 2 DOFs each
    for i in range(8):
        # Calculate derivatives of N with respect to x and y using the inverse Jacobian
        dN_dx, dN_dy = np.dot(J_inv, [dN_dxi[i], dN_deta[i]])
        
        # Populate the B matrix
        B[0, 2*i] = dN_dx  # Strain in x direction due to displacement in x
        B[1, 2*i + 1] = dN_dy  # Strain in y direction due to displacement in y
        B[2, 2*i] = dN_dy  # Shear strain due to displacement in x
        B[2, 2*i + 1] = dN_dx  # Shear strain due to displacement in y

    return B, det_J




# Example coordinates for an 8-node element, replace with actual element coordinates
coords_example = np.array([
    [0, 0],  # Node 1
    [2, 0],  # Node 2
    [2, 2],  # Node 3
    [0, 2],  # Node 4
    [1, 0],  # Node 5 (midside)
    [2, 1],  # Node 6 (midside)
    [1, 2],  # Node 7 (midside)
    [0, 1]   # Node 8 (midside)
])

# Material properties
E = 210e9  # Young's modulus in Pa
nu = 0.3   # Poisson's ratio
t = 0.1    # Thickness in meters

# Compute stiffness matrix for the example element
K = compute_stiffness_matrix_for_8node_element(coords_example, E, nu, t)
# print("Stiffness Matrix (K):")
# print(K)

# Example usage within the stiffness matrix computation or for stress calculation
xi_example, eta_example = -1/np.sqrt(3), -1/np.sqrt(3)  # Example Gauss point
B_example, det_J_example = compute_B_matrix_for_8node_element(coords_example, xi_example, eta_example)
