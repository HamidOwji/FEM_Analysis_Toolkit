import sympy as sp

# Define symbolic variables for xi and eta
xi, eta = sp.symbols('xi eta')

# Define the shape functions for an 8-node square element
N = [
    1/4 * (1-xi) * (1-eta) * (-xi-eta-1),
    1/4 * (1+xi) * (1-eta) * (xi-eta-1),
    1/4 * (1+xi) * (1+eta) * (xi+eta-1),
    1/4 * (1-xi) * (1+eta) * (-xi+eta-1),
    1/2 * (1-xi**2) * (1-eta),
    1/2 * (1+xi) * (1-eta**2),
    1/2 * (1-xi**2) * (1+eta),
    1/2 * (1-xi) * (1-eta**2)
]

# Calculate the derivatives of the shape functions with respect to xi and eta
dN_dxi = [sp.diff(N_i, xi) for N_i in N]
dN_deta = [sp.diff(N_i, eta) for N_i in N]

# Print the derivatives
print("Derivatives with respect to xi:")
for i, dN in enumerate(dN_dxi, start=1):
    print(f"N{i}_xi: {sp.simplify(dN)}")

print("\nDerivatives with respect to eta:")
for i, dN in enumerate(dN_deta, start=1):
    print(f"N{i}_eta: {sp.simplify(dN)}")
