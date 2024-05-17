import numpy as np


# Shape function derivatives in local coordinates (dN/dxi and dN/deta) in clockwise
def shape_func_derivatives(xi, eta, zeta):

    return np.array([
    [-0.125*(eta - 1)*(zeta - 1), -0.125*(xi - 1)*(zeta - 1), -0.125*(eta - 1)*(xi - 1)],
    [0.125*(eta + 1)*(zeta - 1), 0.125*(xi - 1)*(zeta - 1), 0.125*(eta + 1)*(xi - 1)], 
    [-0.125*(eta + 1)*(zeta - 1), -0.125*(xi + 1)*(zeta - 1), -0.125*(eta + 1)*(xi + 1)], 
    [0.125*(eta - 1)*(zeta - 1), 0.125*(xi + 1)*(zeta - 1), 0.125*(eta - 1)*(xi + 1)], 
    [0.125*(eta - 1)*(zeta + 1), 0.125*(xi - 1)*(zeta + 1), 0.125*(eta - 1)*(xi - 1)], 
    [-0.125*(eta + 1)*(zeta + 1), -0.125*(xi - 1)*(zeta + 1), -0.125*(eta + 1)*(xi - 1)],
    [0.125*(eta + 1)*(zeta + 1), 0.125*(xi + 1)*(zeta + 1), 0.125*(eta + 1)*(xi + 1)], 
    [-0.125*(eta - 1)*(zeta + 1), -0.125*(xi + 1)*(zeta + 1), -0.125*(eta - 1)*(xi + 1)], 
    ])

def compute_jacobian(dN_dxi, dN_deta, dN_dzeta, coords):
    J = np.zeros((3, 3))
    for i in range(8):
        J[0, 0] += dN_dxi[i] * coords[i][0]  # dX/dxi
        J[0, 1] += dN_deta[i] * coords[i][0]  # dX/deta
        J[0, 2] += dN_dzeta[i] * coords[i][0]  # dX/dzeta
        J[1, 0] += dN_dxi[i] * coords[i][1]  # dY/dxi
        J[1, 1] += dN_deta[i] * coords[i][1]  # dY/deta
        J[1, 2] += dN_dzeta[i] * coords[i][1]  # dY/dzeta
        J[2, 0] += dN_dxi[i] * coords[i][2]  # dZ/dxi
        J[2, 1] += dN_deta[i] * coords[i][2]  # dZ/deta
        J[2, 2] += dN_dzeta[i] * coords[i][2]  # dZ/dzeta
    return J


def compute_B_matrix(dN_dxi, dN_deta, dN_dzeta, coords):
    J = compute_jacobian(dN_dxi, dN_deta, dN_dzeta, coords)
    J_inv = np.linalg.inv(J)
    det_J = np.linalg.det(J)

    B = np.zeros((6, 24))  # For 3D stress-strain, 8 nodes, 3 DOFs each
    for i in range(8):
        dN_dxyz = np.dot(J_inv, [dN_dxi[i], dN_deta[i], dN_dzeta[i]])
        B[0, 3*i] = dN_dxyz[0]  # Strain in x direction due to displacement in x
        B[1, 3*i+1] = dN_dxyz[1]  # Strain in y direction due to displacement in y
        B[2, 3*i+2] = dN_dxyz[2]  # Strain in z direction due to displacement in z
        B[3, 3*i] = dN_dxyz[1]  # Shear strain xy
        B[3, 3*i+1] = dN_dxyz[0]  # Shear strain xy
        B[4, 3*i+1] = dN_dxyz[2]  # Shear strain yz
        B[4, 3*i+2] = dN_dxyz[1]  # Shear strain yz
        B[5, 3*i] = dN_dxyz[2]  # Shear strain xz
        B[5, 3*i+2] = dN_dxyz[0]  # Shear strain xz
    return B, det_J

def compute_stiffness_matrix_for_8node_element(coords, E, nu):
    # Elasticity matrix for 3D elasticity
    D = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ])
    
    # Gauss points and weights for 3D integration (2 points in each direction)
    sqrt3 = np.sqrt(3)
    gauss_points = [(-1/sqrt3, -1/sqrt3, -1/sqrt3), (1/sqrt3, -1/sqrt3, -1/sqrt3),
                    (-1/sqrt3, 1/sqrt3, -1/sqrt3), (1/sqrt3, 1/sqrt3, -1/sqrt3),
                    (-1/sqrt3, -1/sqrt3, 1/sqrt3), (1/sqrt3, -1/sqrt3, 1/sqrt3),
                    (-1/sqrt3, 1/sqrt3, 1/sqrt3), (1/sqrt3, 1/sqrt3, 1/sqrt3)]
    weights = [1, 1, 1, 1, 1, 1, 1, 1]
    K = np.zeros((24, 24))  # Initialize stiffness matrix for 8 nodes with 3 DOFs each

    for gp, w in zip(gauss_points, weights):
        xi, eta, zeta = gp
        dN_dxi_eta_zeta = shape_func_derivatives(xi, eta, zeta)
        # Directly use the array returned by shape_func_derivatives
        B, det_J = compute_B_matrix(dN_dxi_eta_zeta[:, 0], dN_dxi_eta_zeta[:, 1], dN_dxi_eta_zeta[:, 2], coords)
        
        K += w * det_J * np.dot(B.T, np.dot(D, B))
    # print('coords:', coords)
    # print("\nComputed Stiffness Matrix for 8-Node Element:\n", K)
    return K



# Example coordinates for an 8-node element, replace with actual element coordinates
coords_example = np.array([
    [0, 0, 0],  # Node 1
    [1, 0, 0],  # Node 2
    [1, 1, 0],  # Node 3
    [0, 1, 0],  # Node 4
    [0, 0, 1],  # Node 5
    [1, 0, 1],  # Node 6
    [1, 1, 1],  # Node 7
    [0, 1, 1]   # Node 8
])

# Material properties
E = 210e9  # Young's modulus in Pa
nu = 0.3   # Poisson's ratio

if __name__ == "__main__":

    # Compute stiffness matrix for the example element
    K = compute_stiffness_matrix_for_8node_element(coords_example, E, nu)
    # print("Stiffness Matrix (K):")
    print(K)
