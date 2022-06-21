Development tasks

# todo
- MICHAEL/KALLE ++ work on inversion (improve regularization)
    - penalize deviation from previous timestep? Add prior model vector to run_inversion()
    - transition from east-west to plain gradient or amplitude from 70-80 mlat?
    - minimize gradient
    - boundary condition: HMB
    - Take into account cell distortion in regularization (minimize volume charge density, not sum of line charge densities)
    - Minimize E dot J instead of charge?
- JONE + KALLE Conductance ideas:
    - use Knight relation on FACs from previous timestep?
    - use electric field (Cousins / Merkin)
    - Use ovation prime in addition to EUV?
    - Find optimal example with SSUSI etc. 
    - Empirical orthogonal functions based on auroral images?
- SIMON: do something clever with induced magnetic fields (mirror currents?)
- SIMON Data class
    - add capability to save
    - find out how to design (xarray? pandas?)
    - make checks on input data
    - include uncertainty (e.g. AMPERE provides a +- range)
- find a better solution for the return_shape keyword (it is handled in decorator now)
- KALLE: Implement Fukushima's correction for inclined field
- KALLE/SIMON - include magnetic coordinates in display
- KALLE: Visualization - scatter plots of model data misfits
- KALLE: Plotting tool for cubed sphere projections
- KALLE: Run for Nina's dates: 2014-01-21, 2014-12-18, 2014-12-19
- AMALIE: Add function to map convection data
- AMALIE: Helper function to get east, north orientation of LOS vector
- Add height option for magnetometer station coordinates
- Extend the methods in conductance to have options for coordinate input 
(magnetic, geographic, etc.) 
- ~~write paper~~


# long-term todo
- model residuals from empirical model to reduce boundary effects (HV's idea)
- nonlinear solver (solve simultaneously for conductance and electric field)
- add option to solve for neutral wind
- expand to 3D
