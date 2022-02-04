Development tasks

# todo
- MICHAEL/KALLE ++ work on inversion (improve regularization)
    - penalize deviation from previous timestep? Add prior model vector to run_inversion()
    - transition from east-west to plain gradient or amplitude from 70-80 mlat?
    - minimize gradient
- JONE + KALLE Conductance ideas:
    - use Knight relation on FACs from previous timestep?
    - Use ovation prime in addition to EUV?
    - Find optimal example with SSUSI etc. 
- SIMON Data class
    - add capability to save
    - find out how to design (xarray? pandas?)
    - make checks on input data
    - include uncertainty (e.g. AMPERE provides a +- range)
- find a better solution for the return_shape keyword (it is handled in decorator now)
- SIMON add functionality to store output to xarray / cdf
- KALLE: Implement Fukushima's correction for inclined field
- KALLE: Take into account cell distortion in regularization (minimize volume charge density, not sum of line charge densities)
- KALLE/SIMON - include magnetic coordinates in display
- KALLE: Run for Nina's dates: 2014-01-21, 2014-12-18, 2014-12-19
- ~~write paper~~


# long-term todo
- nonlinear solver (solve simultaneously for conductance and electric field)
- add option to solve for neutral wind
- do something clever with induced magnetic fields (mirror currents?)

