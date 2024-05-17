import numpy as np


# Shape function derivatives in local coordinates (dN/dxi and dN/deta)
def shape_func_derivatives(xi, eta):
    return np.array([
        [-(1 - eta) / 4, -(1 - xi) / 4],
        [(1 - eta) / 4, -(1 + xi) / 4],
        [(1 + eta) / 4, (1 + xi) / 4],
        [-(1 + eta) / 4, (1 - xi) / 4]
    ])


def compute_stiffness_matrix_for_rectangular_element(coords, E, nu, t):
    # Elasticity matrix for plane stress
    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])

    # Gauss quadrature points and weights for 2D integration
    gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                    (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
    weights = [1, 1, 1, 1]

    K = np.zeros((8, 8))  # For 4 nodes with 2 DOFs each

    for (xi, eta), w in zip(gauss_points, weights):
        dN_dxi_eta = shape_func_derivatives(xi, eta)
        
        J = np.zeros((2, 2))
        for i in range(4):
            x_i = coords[i, 0]
            y_i = coords[i, 1]
            J[0, 0] += dN_dxi_eta[i, 0] * x_i
            J[0, 1] += dN_dxi_eta[i, 0] * y_i
            J[1, 0] += dN_dxi_eta[i, 1] * x_i
            J[1, 1] += dN_dxi_eta[i, 1] * y_i

        J_inv = np.linalg.inv(J)
        det_J = np.linalg.det(J)
        
        dN_dx_dy = dN_dxi_eta @ J_inv

        B = np.zeros((3, 8))
        B[0, 0::2] = dN_dx_dy[:, 0]
        B[1, 1::2] = dN_dx_dy[:, 1]
        B[2, 0::2] = dN_dx_dy[:, 1]
        B[2, 1::2] = dN_dx_dy[:, 0]

        K += t * w * det_J * (B.T @ D @ B)
    return K

def compute_B_matrix_for_rectangular_element(coords, gauss_point):
    xi, eta = gauss_point  # Use the specific Gauss point passed as an argument

    # Compute derivatives of shape functions with respect to xi and eta
    dN_dxi_eta = shape_func_derivatives(xi, eta)
    
    # Initialize Jacobian matrix
    J = np.zeros((2, 2))
    for i in range(4):  # Iterate over each node to sum contributions
        x_i = coords[i, 0]
        y_i = coords[i, 1]
        J[0, 0] += dN_dxi_eta[i, 0] * x_i  # ∂x/∂xi
        J[0, 1] += dN_dxi_eta[i, 0] * y_i  # ∂y/∂xi
        J[1, 0] += dN_dxi_eta[i, 1] * x_i  # ∂x/∂eta
        J[1, 1] += dN_dxi_eta[i, 1] * y_i  # ∂y/∂eta

    # Calculate the inverse and determinant of the Jacobian
    J_inv = np.linalg.inv(J)
    
    # Correctly transform derivatives of shape functions to x-y space using J_inv
    dN_dx_dy = np.dot(dN_dxi_eta, J_inv)

    # Calculate B matrix in physical coordinates
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx_dy[:, 0]  # dN/dx for x-direction displacements
    B[1, 1::2] = dN_dx_dy[:, 1]  # dN/dy for y-direction displacements
    B[2, 0::2] = dN_dx_dy[:, 1]  # dN/dy for x-y coupling in shear
    B[2, 1::2] = dN_dx_dy[:, 0]  # dN/dx for y-x coupling in shear

    return B

# print("Stiffness Matrix (K):")
# print(K)
