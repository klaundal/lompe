Development tasks

# todo
- Inversion:
    - Reduce east/west regularization towards the pole
    - Explore Michael's method to determine regularization parameters
    - penalize deviation from previous timestep? Add prior model vector to run_inversion()
    - transition from east-west to plain gradient or amplitude from 70-80 mlat?
    - minimize gradient
    - boundary condition: HMB
    - Take into account cell distortion in regularization (minimize volume charge density, not sum of line charge densities)
    - Minimize E dot J instead of charge?
- Conductance ideas:
    - use Knight relation on FACs from previous timestep?
    - use electric field (Cousins / Merkin)
    - Use ovation prime in addition to EUV?
    - Find optimal example with SSUSI etc. 
    - Empirical orthogonal functions based on auroral images?
- Data handling:
    - add capability to save
    - find out how to design (xarray? pandas?)
    - make checks on input data
- SECS:
    - Implement Fukushima's correction for inclined field
    - Handle induced magnetic field in a better way (mirror current, Liisa's method, ...)
- User friendliness:
    - include magnetic coordinates in interactive display
    - Plotting tool for cubed sphere projections - like polplot
    - Helper function to solve M-I coupling Poisson equation
- Code organization:
    - find a better solution for the return_shape keyword (it is handled in decorator now)
- Run for Nina's dates: 2014-01-21, 2014-12-18, 2014-12-19
- Mapping of convection data in data.py or dataloader.py (with apexpy mapping functions?) 
- Handle radius coordinate from satellite measurements/magnetometers in a 
better way (geocentric/geodetic)
- Calculate resolution with resolution matrix
- Allow for FAC at any position (not just gridded) as input
- Add option to specify neutral wind


# long-term todo
- Take into account inclination of magnetic field (but still height-integrated)
- model residuals from empirical model to reduce boundary effects (HV's idea)
- nonlinear solver (solve simultaneously for conductance and electric field)
    - Possible approach: Use principal component analysis for conductance, or scaling parameter for conductance patterns
- add option to solve for neutral wind
- expand to 3D
