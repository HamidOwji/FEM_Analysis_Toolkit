import numpy as np

# Shape function derivatives in local coordinates (dN/dxi and dN/deta)
def shape_func_derivatives(xi, eta):
    return np.array([
        [4*xi - 1, 0],
        [0, 4*eta - 1],
        [4*(eta - xi), 4*(xi - eta)],
        [4*(1 - 2*xi - eta), -4*xi],
        [4*eta, 4*xi],
        [-4*eta, 4*(1 - xi - 2*eta)]
    ])

def compute_jacobian(dN_dxi, dN_deta, coords):
    J = np.zeros((2, 2))
    for i in range(6):
        J[0, 0] += dN_dxi[i] * coords[i][0]  # dX/dxi
        J[0, 1] += dN_dxi[i] * coords[i][1]  # dY/dxi
        J[1, 0] += dN_deta[i] * coords[i][0]  # dX/deta
        J[1, 1] += dN_deta[i] * coords[i][1]  # dY/deta
    return J

def compute_B_matrix(dN_dxi, dN_deta, coords):
    J = compute_jacobian(dN_dxi, dN_deta, coords)
    J_inv = np.linalg.inv(J)
    det_J = np.linalg.det(J)

    B = np.zeros((3, 12))
    for i in range(6):
        dN_dx, dN_dy = np.dot(J_inv, [dN_dxi[i], dN_deta[i]])
        B[0, 2*i] = dN_dx  # Strain in x direction
        B[1, 2*i + 1] = dN_dy  # Strain in y direction
        B[2, 2*i] = dN_dy  # Shear strain
        B[2, 2*i + 1] = dN_dx
    return B, det_J

def compute_stiffness_matrix_for_6node_element(coords, E, nu, t):
    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    
    gauss_points = [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)]
    weights = [1/6, 1/6, 1/6]
    
    K = np.zeros((12, 12))

    for (xi, eta), w in zip(gauss_points, weights):
        dN_dxi_eta = shape_func_derivatives(xi, eta)
        dN_dxi = dN_dxi_eta[:, 0]
        dN_deta = dN_dxi_eta[:, 1]
        
        B, det_J = compute_B_matrix(dN_dxi, dN_deta, coords)
        
        K += t * w * det_J * np.dot(B.T, np.dot(D, B))
    
    return K

def compute_B_matrix_for_6node_element(coords, xi, eta):
    dN_dxi_eta = shape_func_derivatives(xi, eta)
    dN_dxi = dN_dxi_eta[:, 0]
    dN_deta = dN_dxi_eta[:, 1]
    
    J = compute_jacobian(dN_dxi, dN_deta, coords)
    J_inv = np.linalg.inv(J)
    det_J = np.linalg.det(J)
    
    B = np.zeros((3, 12))
    for i in range(6):
        dN_dx, dN_dy = np.dot(J_inv, [dN_dxi[i], dN_deta[i]])
        B[0, 2*i] = dN_dx
        B[1, 2*i + 1] = dN_dy
        B[2, 2*i] = dN_dy
        B[2, 2*i + 1] = dN_dx

    return B, det_J


if __name__ == "__main__":

    # Example coordinates for a 6-node triangular element
    coords_example = np.array([
        [0, 0], [1, 0], [0, 1],
        [0.5, 0], [0.5, 0.5], [0, 0.5]
    ])
    # Material properties
    E = 210e9  # Young's modulus in Pa
    nu = 0.3   # Poisson's ratio
    t = 0.1    # Thickness in meters

    # Compute stiffness matrix for the example element
    K = compute_stiffness_matrix_for_6node_element(coords_example, E, nu, t)
    print("Stiffness Matrix (K):")
    print(K)

    # Example usage within the stiffness matrix computation or for stress calculation
    xi_example, eta_example = 1/6, 1/6  # Example Gauss point
    B_example, det_J_example = compute_B_matrix_for_6node_element(coords_example, xi_example, eta_example)
    print("B Matrix (B_example):")
    print(B_example)
    print("Determinant of Jacobian (det_J_example):")
    print(det_J_example)
