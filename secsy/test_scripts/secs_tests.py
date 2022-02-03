""" plot SECS currents and magnetic fields to make sure that vectors point 
    in the right direction. Plotting on cubedsphere grid
"""
import numpy as np
import matplotlib.pyplot as plt
from secsy import CSprojection, CSgrid, get_SECS_B_G_matrices, get_SECS_J_G_matrices

RE = 6371.2 * 1e3
H  = 110.e3

secs_lats = np.array([60, 61])
secs_lons = np.array([10, 20])
df_m = np.array([1, -1]).reshape((-1, 1))
cf_m = np.array([1, -1]).reshape((-1, 1))


lat0, lon0 = 60.5, 15 # projection center

projection = CSprojection((lon0, lat0), 0)
grid = CSgrid(projection, 1200e3, 1200e3, 7e3, 7e3, R = RE + H)
SKIP = 10 # how many grid points to skip when plotting vector fields
grid_vlat, grid_vlon = grid.lat[::SKIP, ::SKIP].flatten(), grid.lon[::SKIP, ::SKIP].flatten()

G_B_df_e, G_B_df_n, _ = get_SECS_B_G_matrices(grid_vlat, grid_vlon,
                                              RE, secs_lats, secs_lons, 
                                              current_type = 'divergence_free')
G_B_cf_e, G_B_cf_n, _ = get_SECS_B_G_matrices(grid_vlat, grid_vlon,
                                              RE+2*H, secs_lats, secs_lons, 
                                              current_type = 'curl_free')
_, _, G_B_df_u = get_SECS_B_G_matrices(grid.lat.flatten(), grid.lon.flatten(),
                                      RE, secs_lats, secs_lons, 
                                      current_type = 'divergence_free')
_, _, G_B_cf_u = get_SECS_B_G_matrices(grid_vlat, grid_vlon,
                                       RE+2*H, secs_lats, secs_lons, 
                                       current_type = 'curl_free')
G_J_df_e, G_J_df_n = get_SECS_J_G_matrices(grid_vlat, grid_vlon,
                                           secs_lats, secs_lons, 
                                           current_type = 'divergence_free', RI = RE + H)
G_J_cf_e, G_J_cf_n = get_SECS_J_G_matrices(grid_vlat, grid_vlon,
                                           secs_lats, secs_lons, 
                                           current_type = 'curl_free', RI = RE + H)

je_df, jn_df = G_J_df_e.dot(df_m), G_J_df_n.dot(df_m)
je_cf, jn_cf = G_J_cf_e.dot(cf_m), G_J_cf_n.dot(cf_m)

Be_df, Bn_df = G_B_df_e.dot(df_m), G_B_df_n.dot(df_m)
Be_cf, Bn_cf = G_B_cf_e.dot(cf_m), G_B_cf_n.dot(cf_m)


fig, axes = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 8))

# divergence-free currents:
for i, ax in enumerate(axes[0]):
    x, y = projection.geo2cube(secs_lons, secs_lats)

    pos = df_m.flatten() > 0
    ax.scatter(x[ pos], y[ pos], c = 'C1', cmap = plt.cm.bwr, marker = '$+$', zorder = 10)
    ax.scatter(x[~pos], y[~pos], c = 'C1', cmap = plt.cm.bwr, marker = '$-$', zorder = 10)

    if i == 0:
        xx, yy, jx, jy = projection.vector_cube_projection(je_df, jn_df, grid_vlon, grid_vlat, return_xi_eta = True)
        ax.quiver(xx, yy, jx, jy)

    if i == 1:
        xx, yy, Bx, By = projection.vector_cube_projection(Be_df, Bn_df, grid_vlon, grid_vlat, return_xi_eta = True)
        ax.quiver(xx, yy, Bx, By)

        Bu = G_B_df_u.dot(df_m).reshape(grid.shape)
        Bum = np.max(np.abs(Bu))
        ax.contourf(grid.xi, grid.eta, Bu, zorder = 0, cmap = plt.cm.bwr, levels = np.linspace(-Bum, Bum, 23))

    ax.set_axis_off()
    ax.set_aspect('equal')


# curl-free currents:
for i, ax in enumerate(axes[1]):
    x, y = projection.geo2cube(secs_lons, secs_lats)
    pos = cf_m.flatten() > 0
    ax.scatter(x[ pos], y[ pos], c = 'C1', cmap = plt.cm.bwr, marker = '$+$', zorder = 10)
    ax.scatter(x[~pos], y[~pos], c = 'C1', cmap = plt.cm.bwr, marker = '$-$', zorder = 10)

    if i == 0:
        xx, yy, jx, jy = projection.vector_cube_projection(je_cf, jn_cf, grid_vlon, grid_vlat, return_xi_eta = True)
        ax.quiver(xx, yy, jx, jy)

    if i == 1:
        xx, yy, Bx, By = projection.vector_cube_projection(Be_cf, Bn_cf, grid_vlon, grid_vlat, return_xi_eta = True)
        ax.quiver(xx, yy, Bx, By)

    ax.set_axis_off()
    ax.set_aspect('equal')

axes[0, 0].set_title('Divergence-free j')
axes[0, 1].set_title('Divergence-free B on ground')

axes[1, 0].set_title('Curl-free j')
axes[1, 1].set_title('Curl-free B in space')


plt.show()