""" THIS IS CODE COPIED FROM model.py TO BE SAVED AND USED DIFFERENTLY """
    # def save_output(self, time, parameter='all', path=False, overwrite=False):
    #     """
    #     Parameters
    #     ----------
    #     time : datetime
    #         The date and time of the conditions being modelled.
    #     parameter : str/list/numpy.ndarray/dict/dict_keys, optional
    #         Used to pick which model outputs to save and when using the dictionary can implement different functions. 
    #         The default is 'all' which saves all model outputs using the default functions in self.save_options.

    #     Raises
    #     ------
    #     ValueError
    #         Will be raised if the parameter argument isn't of a valid type.

    #     Returns
    #     -------
    #     Due to the evaluation grid for each model output being two different grids
    #     that depend on the parameter being modelled there are two xarray datasets that
    #     are returned.
        
    #     xarray1:
    #         Model outputs evaluated on self.grid.
    #     xarray2:
    #         Model outputs evaluated on self.grid_E.
    #     dict:
    #         If parameter is 'all' or grid is chosen as a parameter then a dictionary of 
    #         the grid properties is returned that can be used to recreate the cubed sphere grid objects
    #         self.grid and self.grid_E
    #     """
    #     try:
    #         from progressbar import progressbar
    #     except:
    #         print('progressbar package not installed\nrunning script without')
    #         def progressbar(iterable, **kwargs):
    #                 if 'prefix' in kwargs:
    #                     print(kwargs['prefix'])
    #                 return iterable
    #     if isinstance(parameter, str) and parameter =='all':
    #         parameter= self.save_options.keys()
    #     if isinstance(parameter, (list, tuple, np.ndarray, type({}.keys()))):
    #         grid1= [self.save_options[key.lower()] for key in \
    #                 progressbar(parameter, max_value=len(parameter), prefix= 'Collecting functions (1/2) |') if key.lower() in \
    #                     ['velocity', 'potential', 'current', 'conductance', 'electric']]
    #         grid_E= [self.save_options[key.lower()] for key in \
    #                 progressbar(parameter, max_value=len(parameter), prefix= 'Collecting functions (2/2) |') if key.lower() in \
    #                     ['amplitudes', 'ground_magnetic', 'space_magnetic']]
    #         import xarray as xr
    #         g1={}
    #         g2={}
    #         for func in progressbar(grid1, max_value=len(grid1), prefix='Running functions and creating xarray datasets (1/2) |'): 
    #             if 'data_vars' in g1:
    #                 x= func(self, time)
    #                 g1['data_vars'].update(x['data_vars'])
    #                 g1['coords'].update(x['coords'])
    #             else:
    #                 g1.update(func(self, time))
    #         for func in progressbar(grid_E, max_value=len(grid_E), prefix='Running functions and creating xarray datasets (2/2) |'): 
    #             if 'data_vars' in g2:
    #                 x= func(self, time)
    #                 g2['data_vars'].update(x['data_vars'])
    #                 g2['coords'].update(x['coords'])
    #             else:
    #                 g2.update(func(self, time))
    #         DS=[xr.Dataset(**g1),
    #             xr.Dataset(**g2)]
    #         if 'grid' in parameter:
    #            DS.append(self.save_options['grid'](self))
    #         if path:
    #             g1= DS[0]
    #             g2= DS[1]
    #             # g1['grid_str']= list(DS[2]['grid'].keys())
    #             # g1['grid']= list(DS[2]['grid'].values())
    #             g1.attrs.update(DS[2]['grid'])
    #             g2.attrs.update(DS[2]['projection'])
    #             # g1['projection_str']= list(DS[2]['projection'].keys())
    #             # print(list(DS[2]['projection'].values()))
    #             # g1['projection']= list(DS[2]['projection'].values())
                
                
                
                
    #             # g1.attrs.update({'grid': np.array([[key, DS[2]['grid'][key]] for key in DS[2]['grid'].keys()]).flatten()})
    #             # g1.attrs.update({'projection':np.array([[key, DS[2]['projection'][key]] for key in DS[2]['projection'].keys()]).flatten()})
    #             # g1.attrs.update({'grid': [[key, DS[2]['grid'][key]] for key in DS[2]['grid'].keys()], 'projection':\
    #             #     [[key, DS['projection'[2]]] for key in DS[2]['projection'].keys()]})
    #             # g1['grid'], g1['projection']=[DS[2]['grid']], [DS[2]['projection']]
    #             # g2['grid'], g2['projection']=[DS[2]['grid']], [DS[2]['projection']]
    #             # print(g1)
    #             # return g1, g2
                
    #             # Merge with pre existing files
    #             import os
    #             if os.path.isfile(path+'grid1.ncdf4') and not overwrite:
    #                 g1= xr.concat((g1, xr.load_dataset(path+'grid1.ncdf4', engine= 'netcdf4')), dim='date')
    #             if os.path.isfile(path+'grid_E.ncdf4') and not overwrite:
    #                 g2= xr.concat((g2, xr.load_dataset(path+'grid_E.ncdf4', engine= 'netcdf4')), dim='date')
    #             g1.to_netcdf(path+'grid1.ncdf4')
    #             g2.to_netcdf(path+'grid_E.ncdf4')
    #             DS=DS[:2]
    #         return DS
    #     elif isinstance(parameter, dict):
    #         grid1= {key.lower():parameter[key] for key in progressbar(parameter.keys()) if key.lower() in ['velocity', 'potential', 'current', 'conductance']}
    #         grid_E= {key.lower():parameter[key] for key in progressbar(parameter.keys()) if key.lower() in ['amplitudes', 'ground_magnetic', 'space_magnetic']}
    #     else:
    #         raise ValueError(f'parameter argument is not a valid type it must be either {str}, {list}, {tuple}, {np.ndarray} or {dict}. \
    #                          parameter is a {type(parameter)}. Please try again!')
        


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:45:26 2021

@author: simon
"""

import numpy as np
# from .varcheck import check_input
# def save_SECS_J_G(model, lon=None, lat=None):
#     import numpy as np
#     from lompe.model.model import get_SECS_J_G_matrices
#     if lat is None:            
#         lat = model.lat
#     if lon is None:            
#         lon= model.lon
#     Bu= model.Bu
#     B0= model.B0
#     return get_SECS_J_G_matrices(lat, lon, model.lat2, model.lon2,
#                                        current_type = 'curl_free',
#                                        RI = model.grid.R,
#                                        singularity_limit = model.secs_singularity_limit)
def save_grid(model):
    projection={'position':model.grid.projection.position, 'orientation':model.grid.projection.orientation}
    grid= {'L':model.grid.L, 'W':model.grid.W, 'Lres':model.grid.Lres, 'Wres':model.grid.Wres, 'R':model.grid.R}
        
    return {'projection': projection, 'grid': grid}


def save_amplitudes(model, time):
    if not isinstance(time, (list, np.ndarray)):
        if isinstance(time, tuple):
            time= list(time)
        else:
            time= [time]
    # ds = xr.Dataset(data_vars= {'Amplitude': (['date', 'xi', 'eta'], np.array([model.m.reshape(model.grid2.shape).T]))}, 
    #                 coords={"date": (["date"], time),
    #                         "xi": (["xi"], model.grid2.xi[0]),
    #                         "eta": (["eta"], model.grid2.eta[:,0])})
    # return ds
    return dict(data_vars= {'Amplitude': (['date', 'xi', 'eta'], np.array([model.m.reshape(model.grid2.shape).T]))}, 
                    coords={"date": (["date"], time),
                            "xi": (["xi"], model.grid2.xi[0]),
                            "eta": (["eta"], model.grid2.eta[:,0])})


def save_velocity(model, time):
    if not isinstance(time, (list, np.ndarray)):
        if isinstance(time, tuple):
            time= list(time)
        else:
            time= [time]
    v_e, v_n= model.v()    
    # ds = xr.Dataset(data_vars= {'Velocity_East': (['date', 'xi', 'eta'], np.array([v_e.reshape(model.grid.shape).T])),
    #                             'Velocity_North':(['date', 'xi', 'eta'], np.array([v_n.reshape(model.grid.shape).T]))},
    #                 coords={"date": (["date"], time), 
    #                         "xi": (["xi"], model.grid.xi[0]),
    #                         "eta": (["eta"], model.grid.eta[:,0])})
    # return ds
    return dict(data_vars= {'Velocity_East': (['date', 'xi', 'eta'], np.array([v_e.reshape(model.grid.shape).T])),
                                'Velocity_North':(['date', 'xi', 'eta'], np.array([v_n.reshape(model.grid.shape).T]))},
                    coords={"date": (["date"], time), 
                            "xi": (["xi"], model.grid.xi[0]),
                            "eta": (["eta"], model.grid.eta[:,0])})


def save_electric(model, time):
    if not isinstance(time, (list, np.ndarray)):
        if isinstance(time, tuple):
            time= list(time)
        else:
            time= [time]
    E_e, E_n= model.E()    
    # ds = xr.Dataset(data_vars= {'Electric_East': (['date', 'xi', 'eta'], np.array([E_e.reshape(model.grid.shape).T])),
    #                             'Electric_North':(['date', 'xi', 'eta'], np.array([E_n.reshape(model.grid.shape).T]))},
    #                 coords={"date": (["date"], time), 
    #                         "xi": (["xi"], model.grid.xi[0]),
    #                         "eta": (["eta"], model.grid.eta[:,0])})
    return dict(data_vars= {'Electric_East': (['date', 'xi', 'eta'], np.array([E_e.reshape(model.grid.shape).T])),
                                'Electric_North':(['date', 'xi', 'eta'], np.array([E_n.reshape(model.grid.shape).T]))},
                    coords={"date": (["date"], time), 
                            "xi": (["xi"], model.grid.xi[0]),
                            "eta": (["eta"], model.grid.eta[:,0])})


def save_electric_pot(model, time):
    if not isinstance(time, (list, np.ndarray)):
        if isinstance(time, tuple):
            time= list(time)
        else:
            time= [time]
    E_pot= model.E_pot()    
    # ds = xr.Dataset(data_vars= {'Potential': (['date', 'xi', 'eta'], np.array([E_pot.reshape(model.grid.shape).T]))},
    #                 coords={"date": (["date"], time), 
    #                         "xi": (["xi"], model.grid.xi[0]),
    #                         "eta": (["eta"], model.grid.eta[:,0])})
    # return ds
    return dict(data_vars= {'Potential': (['date', 'xi', 'eta'], np.array([E_pot.reshape(model.grid.shape).T]))},
                    coords={"date": (["date"], time), 
                            "xi": (["xi"], model.grid.xi[0]),
                            "eta": (["eta"], model.grid.eta[:,0])})


def save_groundB(model, time):
    if not isinstance(time, (list, np.ndarray)):
        if isinstance(time, tuple):
            time= list(time)
        else:
            time= [time]
    Be, Bn, Bu= model.B_ground()
    # ds = xr.Dataset(data_vars= {'GroundB_East': (['date', 'xi', 'eta'], np.array([Be.reshape(model.grid2.shape).T])),
    #                             'GroundB_North':(['date', 'xi', 'eta'], np.array([Bn.reshape(model.grid2.shape).T])),
    #                             'GroundB_Radial':(['date', 'xi', 'eta'],np.array([Bu.reshape(model.grid2.shape).T]))},
    #                 coords={"date": (["date"], time), 
    #                         "xi": (["xi"], model.grid2.xi[0]),
    #                         "eta": (["eta"], model.grid2.eta[:,0])})
    # return ds
    return dict(data_vars= {'GroundB_East': (['date', 'xi', 'eta'], np.array([Be.reshape(model.grid2.shape).T])),
                                'GroundB_North':(['date', 'xi', 'eta'], np.array([Bn.reshape(model.grid2.shape).T])),
                                'GroundB_Radial':(['date', 'xi', 'eta'],np.array([Bu.reshape(model.grid2.shape).T]))},
                    coords={"date": (["date"], time), 
                            "xi": (["xi"], model.grid2.xi[0]),
                            "eta": (["eta"], model.grid2.eta[:,0])})


def save_current(model, time):
    if not isinstance(time, (list, np.ndarray)):
        if isinstance(time, tuple):
            time= list(time)
        else:
            time= [time]
    j_e, j_n= model.j()
    # ds = xr.Dataset(data_vars= {'Sheet_Current_East': (['date', 'xi', 'eta'], np.array([j_e.reshape(model.grid.shape).T])),
    #                             'Sheet_Current_North':(['date', 'xi', 'eta'], np.array([j_n.reshape(model.grid.shape).T])),
    #                             'Field_Aligned_Current':(['date', 'xi','eta'],np.array([model.FAC().reshape(model.grid.shape).T]))},
    #                 coords={"date": (["date"], time), 
    #                         "xi": (["xi"], model.grid.xi[0]),
    #                         "eta": (["eta"], model.grid.eta[:,0])})
    return dict(data_vars= {'Sheet_Current_East': (['date', 'xi', 'eta'], np.array([j_e.reshape(model.grid.shape).T])),
                                'Sheet_Current_North':(['date', 'xi', 'eta'], np.array([j_n.reshape(model.grid.shape).T])),
                                'Field_Aligned_Current':(['date', 'xi','eta'],np.array([model.FAC().reshape(model.grid.shape).T]))},
                    coords={"date": (["date"], time), 
                            "xi": (["xi"], model.grid.xi[0]),
                            "eta": (["eta"], model.grid.eta[:,0])})


def save_spaceB(model, time):
    if not isinstance(time, (list, np.ndarray)):
        if isinstance(time, tuple):
            time= list(time)
        else:
            time= [time]
    Be, Bn, Bu= model.B_space()
    # ds = xr.Dataset(data_vars= {'SpaceB_East': (['date', 'xi', 'eta'], np.array([Be.reshape(model.grid2.shape).T])),
    #                             'SpaceB_North':(['date', 'xi', 'eta'], np.array([Bn.reshape(model.grid2.shape).T])),
    #                             'SpaceB_Radial':(['date', 'xi', 'eta'], np.array([Bu.reshape(model.grid2.shape).T]))},
    #                 coords={"date": (["date"], time), 
    #                         "xi": (["xi"], model.grid2.xi[0]),
    #                         "eta": (["eta"], model.grid2.eta[:,0])})
    return dict(data_vars= {'SpaceB_East': (['date', 'xi', 'eta'], np.array([Be.reshape(model.grid2.shape).T])),
                                'SpaceB_North':(['date', 'xi', 'eta'], np.array([Bn.reshape(model.grid2.shape).T])),
                                'SpaceB_Radial':(['date', 'xi', 'eta'], np.array([Bu.reshape(model.grid2.shape).T]))},
                    coords={"date": (["date"], time), 
                            "xi": (["xi"], model.grid2.xi[0]),
                            "eta": (["eta"], model.grid2.eta[:,0])})


def save_conductance(model, time):
    if not isinstance(time, (list, np.ndarray)):
        if isinstance(time, tuple):
            time= list(time)
        else:
            time= [time]
    # ds= xr.Dataset(data_vars={'Hall_Conductance': (['date', 'xi', 'eta'], np.array([model.hall_conductance().reshape(model.grid.shape).T])),
    #                           'Pedersen_Conductance':(['date', 'xi', 'eta'], np.array([model.pedersen_conductance().reshape(model.grid.shape).T]))},
    #                coords={"date": (["date"], time), 
    #                         "xi": (["xi"], model.grid.xi[0]),
    #                         "eta": (["eta"], model.grid.eta[:,0])})
    return dict(data_vars={'Hall_Conductance': (['date', 'xi', 'eta'], np.array([model.hall_conductance().reshape(model.grid.shape).T])),
                              'Pedersen_Conductance':(['date', 'xi', 'eta'], np.array([model.pedersen_conductance().reshape(model.grid.shape).T]))},
                   coords={"date": (["date"], time), 
                            "xi": (["xi"], model.grid.xi[0]),
                            "eta": (["eta"], model.grid.eta[:,0])})


def unused_args(args, **kwargs):
    if args and kwargs:
        print(f'Loading from save model args: {args} and kwargs: {list(kwargs.keys())} are unused')
    elif args:
        print(f'Loading from save model args: {args} are unused')
    elif kwargs:
        print(f'Loading from save model kwargs: {list(kwargs.keys())} are unused')


class load_model(object):
    def __init__(self, path):
        import xarray as xr
        self.grid1_dataset= xr.load_dataset(path+'grid1.ncdf4', engine='netcdf4')
        self.grid2_dataset= xr.load_dataset(path+'grid2.ncdf4', engine='netcdf4')
        from secsy import cubedsphere as cs 
        self.grid= cs.CSgrid(cs.CSprojection(**self.grid2_dataset.attrs), **self.grid1_dataset.attrs)
        xi_e  = np.hstack((self.grid.xi_mesh[0]    , self.grid.xi_mesh [0 , - 1] + self.grid.dxi )) - self.grid.dxi /2 
        eta_e = np.hstack((self.grid.eta_mesh[:, 0], self.grid.eta_mesh[-1,   0] + self.grid.deta)) - self.grid.deta/2 
        self.grid2 = cs.CSgrid(cs.CSprojection(self.grid.projection.position, self.grid.projection.orientation),
                               self.grid.L + self.grid.Lres, self.grid.W + self.grid.Wres, self.grid.Lres, self.grid.Wres, 
                               edges = (xi_e, eta_e), R = self.grid.R)

        self.lat , self.lon  = np.ravel( self.grid.lat), np.ravel(self.grid.lon)
        self.lat2, self.lon2 = np.ravel( self.grid2.lat), np.ravel(self.grid2.lon)
    def E_pot(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid1_dataset.sel(date=self.time)['Potential'].values.T
    def E(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid1_dataset.sel(date=self.time)['Electric_East'].values.T, self.grid1_dataset.sel(date=self.time)['Electric_North'].values.T
    def v(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid1_dataset.sel(date=self.time)['Velocity_East'].values.T, self.grid1_dataset.sel(date=self.time)['Velocity_North'].values.T
    def j(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid1_dataset.sel(date=self.time)['Sheet_Current_East'].values.T, self.grid1_dataset.sel(date=self.time)['Sheet_Current_North'].values.T
    def B_space_FAC(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid2_dataset.sel(date=self.time)['SpaceB_East'].values.T, self.grid2_dataset.sel(date=self.time)['SpaceB_North'].values.T, self.grid2_dataset.sel(date=self.time)['SpaceB_Radial'].values.T
    def FAC(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid1_dataset.sel(date=self.time)['Field_Aligned_Current'].values.T
    def hall_conductance(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid1_dataset.sel(date=self.time)['Hall_Conductance'].values.T
    def pedersen_conductance(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid1_dataset.sel(date=self.time)['Pedersen_Conductance'].values.T
    def B_ground(self, *args, **kwargs):
        unused_args(args, **kwargs)
        return self.grid2_dataset.sel(date=self.time)['GroundB_East'].values.T, self.grid2_dataset.sel(date=self.time)['GroundB_North'].values.T, self.grid2_dataset.sel(date=self.time)['GroundB_Radial'].values.T