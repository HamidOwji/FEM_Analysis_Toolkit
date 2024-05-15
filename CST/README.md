# Finite Element Method with CST Elements

This repository contains Python modules for performing finite element analysis using Constant Strain Triangle (CST) elements. The project includes mesh generation, stiffness matrix computation, and plotting of results.

## Modules

### `FEM_CST_general.py`

This module contains the main functions for finite element analysis, including the computation of stiffness matrices, assembling the global stiffness matrix, and solving for nodal displacements.

### `Mesh_Tri3_extractor.py`

This module includes functions for reading mesh data from a file and generating elements from the mesh data.

### `FEM_CST_plotting.py`

This module contains functions for plotting the mesh, nodal displacements, boundary conditions, and loads.

## Usage

To use these modules, follow these steps:

1. Ensure you have the required dependencies installed:

   ```bash
   pip install numpy matplotlib h5py
   ```

    Run the FEM_CST_general.py script with the appropriate mesh file and parameters.

## Example

An example of how to use these modules can be found in the FEM_CST_general.py script. The script reads mesh data, computes stiffness matrices, applies boundary conditions and loads, solves for nodal displacements, and plots the results.
License

## This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

This project uses the following libraries:

    NumPy
    Matplotlib
    h5py

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.