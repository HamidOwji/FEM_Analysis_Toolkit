import sympy as sp

# Define symbolic variables for xi, eta, and zeta
xi, eta, zeta = sp.symbols('xi eta zeta')

# Define the shape functions for an 8-node hexahedral element
N = [
    1/8 * (1-xi) * (1-eta) * (1-zeta),
    1/8 * (1+xi) * (1-eta) * (1-zeta),
    1/8 * (1+xi) * (1+eta) * (1-zeta),
    1/8 * (1-xi) * (1+eta) * (1-zeta),
    1/8 * (1-xi) * (1-eta) * (1+zeta),
    1/8 * (1+xi) * (1-eta) * (1+zeta),
    1/8 * (1+xi) * (1+eta) * (1+zeta),
    1/8 * (1-xi) * (1+eta) * (1+zeta),
]

# Calculate the derivatives of the shape functions with respect to xi, eta, and zeta
dN_dxi = [sp.diff(N_i, xi) for N_i in N]
dN_deta = [sp.diff(N_i, eta) for N_i in N]
dN_dzeta = [sp.diff(N_i, zeta) for N_i in N]

# Calculate the derivatives of the shape functions
derivatives = [
    [
        sp.simplify(sp.diff(N_i, xi)),
        sp.simplify(sp.diff(N_i, eta)),
        sp.simplify(sp.diff(N_i, zeta))
    ]
    for N_i in N
]

# For displaying the derivatives in a structured format
# for i, dN in enumerate(derivatives, start=1):
#     print(f"N{i}: {{dxi: {dN[0]}, deta: {dN[1]}, dzeta: {dN[2]}}}")

# Print the derivatives
# print("Derivatives with respect to xi:")
# for i, dN in enumerate(dN_dxi, start=1):
#     print(f"N{i}_xi: {sp.simplify(dN)}")

# print("\nDerivatives with respect to eta:")
# for i, dN in enumerate(dN_deta, start=1):
#     print(f"N{i}_eta: {sp.simplify(dN)}")

# print("\nDerivatives with respect to zeta:")
# for i, dN in enumerate(dN_dzeta, start=1):
#     print(f"N{i}_zeta: {sp.simplify(dN)}")
print(derivatives)