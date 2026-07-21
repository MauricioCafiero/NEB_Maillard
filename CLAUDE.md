#  find transition states for maillard reactions

##  create code that, given two structures (a reactant and a product) will use the NEB approach to find a transition state.

steps:
- create an environment (python or UV)
- install tblite (for the GFN2 semi-empirical method, which we will use), ASE for calculations (energy, geometry, vibrational modes), RDKit for general chemistry functionality, and any packages you may need to implement nudged elastic band.
- you will find reactant and product output files for Gaussian 16 in this folder. Extract the final optimized structures from those to use as a test. 
