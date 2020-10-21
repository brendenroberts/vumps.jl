# vumps.jl
This is a Julia implementation of the ``vumps'' algorithm of Zauner-Stauber, Vanderstraeten, Fishman, Verstraete, and Haegeman (https://doi.org/10.1103/PhysRevB.97.045145), using Haegeman's TensorOperations (https://github.com/Jutho/TensorOperations.jl).
The design is modular, with the central loop accepting a specification of the Hamiltonian via MPO tensor.
Included is a driver code implmenting an ``adiabatic'' procedure of optimizing a state for a particular Hamiltonian and then using this as the initial state for a new Hamiltonian with parameters tuned slightly.

The function `vumps` takes the following parameters:
1. The MPO tensor, a four-dimensional `Array` of `Real` or `Complex` types
1. A `Dict` with `String` keys and values each a `Vector` of three-dimensional `Array` objects. It should include keys `"AC"`, `"AL"`, and `"AR"`, which are the complete unit cell of MPS tensors for the initial state.
The `"AC"` tensors are two-leg objects, but in the interest of really specific typing one should add a dummy dimension to these tensors.
The datatype of these tensors should match the type of the MPO tensor.
1. A vector of `HermitianOp` objects, which is a compound type containing a `name::String` and a `Vector` of `Matrices` named `op`.
These onsite operators will be measured in the MPS by the algorithm as it runs, allowing for tracking of spontaneous magnetic ordering.
1. A `Float64` specifying the stopping criterion for the convergence of the `B` tensors.
This is convergence of the state, and the convergence of the energy will be stronger.
1. A `String` specifying the base for the output log file. The file will have `.log` appended.

The driver code is in `scan_Z3xZ3.jl`, which can be run from the command line and takes the following command-line arguments:
1. `N_uc`: number of sites in the unit cell
1. `chi`: MPS bond dimension
1. `pfile`: an ASCII file containing parameters for the Hamiltonian tuning.
See file `params.txt` for an example of the format.
1. `tag`: a string uniquely identifying this particular scan.
This allows for multiple arbitrary scans through the same phase diagram to live in the same data file.
1. `fname`: name of the overall data file.
1. `init.jld`: optional JLD file containing initial state for the scan
The output of this driver will be a single data file as well as the vumps log for every value of parameters specified in `pfile` and a JLD file (which is essentially HDF5) containing each optimized wavefunction.

Also included is similar code which takes only a single set of the `d` and `K` values that parameterize this particular Hamiltonian and a file containing a list of bond dimensions.
This code will optimize an MPS for each bond dimension, then use it to grow a state at a larger bond dimension.
To see the command line arguments for this code, run it without any arguments. 
