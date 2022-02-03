Development tasks

# todo
- KALLE: Make a better README
- ~~KALLE: lompe.Data - remove nans~~
- KALLE: In clean version: lompe.Data - remove Hall/Pedersen; remove cmodel stuff, 
- ALL: Remove references to ext_factor (replace with perimeter_width if needed)
- ~~KALLE: reimplment ext_factor instead of hard-coded 10 cell boundary~~
- ~~KALLE: Adjust lompeplot size to aspect ratios~~
- ~~KALLE: Make the lompeplot polar plot nicer~~
- KALLE: Run for Nina's dates: 2014-01-21, 2014-12-18, 2014-12-19
- MICHAEL/KALLE ++ work on inversion (improve regularization)
    - penalize deviation from previous timestep? Add prior model vector to run_inversion()
    - transition from east-west to plain gradient or amplitude from 70-80 mlat?
    - minimize gradient
- JONE + KALLE Conductance ideas:
    - use Knight relation on FACs from previous timestep?
    - Use ovation prime in addition to EUV?
    - Find optimal example with SSUSI etc. 
- SIMON Data class (almost just a placeholder now)
    - add capability to save
    - find out how to design (xarray? pandas?)
    - make checks on input data
    - include uncertainty (e.g. AMPERE provides a +- range)
- find a better solution for the return_shape keyword (it is handled in decorator now)
- fix demo notebooks
- SIMON add functionality to store output to xarray / cdf
- KALLE: Implement Fukushima's correction for inclined field
- KALLE: Take into account cell distortion in regularization (minimize volume charge density, not sum of line charge densities)
- KALLE/SIMON - include magnetic coordinates in display
- write paper

# DONE
- KALLE do grid lines form great circles? Yes they do
(- KALLE: Normalize the calculations so that units of magnetic field, electric field, etc. are comparable to avoid numerical crap)
- KALLE: Check if the radial component / divergence free part of B works in space (also: is there any information in this component from Iridium?)
- KALLE: Spatial weights: Vector measurements are counted once per component (I think) - fix that
- KALLE: Make differentiation matrices sparse and include arbitrary stencil sizes
- test if the ground magnetic field actually works, using Biot-Savart or something like that


# long-term todo
- nonlinear solver (solve simultaneously for conductance and electric field)
- add option to solve for neutral wind
- do something clever with induced magnetic fields (mirror currents?)
- do something clever with boundary conditions
- take into account non-radial main magnetic field (see note for recipe)


# minor
- what is the effect of the "global curl/divergence" model parameters? They are set to 0 now. 
