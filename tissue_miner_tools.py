import sqlite3 as lite
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import pandas.io.parsers as pp

#from sets import Set 
import os.path
import time

def angle_difference(angle2, angle1):
    """
    Return the angle difference constrained to [-pi, pi] interval! CAUTION: Use only for small angle differences. Maybe todo: add warning when not equal to angle2 - angle1?
    """
    return np.angle(np.exp(1j*(angle2 - angle1)))

def subset_dataframe(df, columns, conditions):
    """
    Subset a DataFrame object on columns in columns list
    by values is conditions list
    """ 
    t= df.copy()
    for column, condition in zip(columns, conditions):
        t= t[t[column] == condition]
    return t                        

def fill_zeros(s,n):
    """
    Add zeros to a string s until
    it reaches length n.
    """
    while len(s) < n:
        s= ''.join(('0',s))
    return s

def smooth_data(x, NSmooth=10, mode= 'valid'):
    """
    Returns the original data smoothed over a
    window of lenght NSmooth.
    """
    return np.convolve(x, 1.*np.ones(NSmooth)/NSmooth, mode= mode)


def make_directory(path):
    """
    Create all directories in a proposed path
    - unlike simple os.makedirs(path) for which all
    higher level directories have to exist.
    """
    path_list= path.rstrip().rstrip('/').split('/')
    making_path= ''
    if path_list[0] == '':
        making_path+= '/'
        path_list.pop(0)
    for segment in path_list:
        making_path+= segment
        if not os.path.exists(path):
            os.makedirs(path)
        making_path+='/'

def cells_total_area(rc):
    """
    Calculates sum of cell areas in each frame. 
    Intednded for use on 'cells' or subsets of 'cells' tables.
    """    
    return rc[['frame', 'area']].groupby('frame').agg(np.sum).reset_index().sort('frame')['area'].values

def cells_average_area(rc):
    """
    Calculates average of cell areas in each frame. 
    Intednded for use on 'cells' or subsets of 'cells' tables.
    """
    return rc[['frame', 'area']].groupby('frame').agg(np.mean).reset_index().sort('frame')['area'].values
    #return rc[['frame', 'area']].groupby('frame').agg(np.mean).reset_index().sort_values('frame')['area'].values

def cells_average_elongation(rc):
    """
    Calculates average _xx and _xy components of the elongation nematic in each frame.
    Intedend for use on 'cells' or subsets of 'cells' tables. NOT RECOMMENDED IF TRIANGLE ELONGATION IS AVALIABLE.
    """
    Q_xx= rc[['frame', 'elong_xx']].groupby('frame').agg(np.mean).reset_index().sort('frame')['elong_xx'].values
    Q_xy= rc[['frame', 'elong_xy']].groupby('frame').agg(np.mean).reset_index().sort('frame')['elong_xy'].values
    return Q_xx, Q_xy

def cells_number(rc):
    """
    Calculates number of cells in each frame.
    Intedend for use on 'cells' or subsets of 'cells' tables.
    """    
    return rc[['frame', 'cell_id']].groupby('frame').agg(len).reset_index().sort('frame')['cell_id'].values


def region_shape_nematic(rc_in):
    rc= rc_in.copy()
    rc['x_area']= rc['center_x']*rc['area']
    rc['xx_area']= rc['center_x']*rc['center_x']*rc['area']
    rc['xy_area']= rc['center_x']*rc['center_y']*rc['area']
    rc['yy_area']= rc['center_y']*rc['center_y']*rc['area']
    rc['y_area']= rc['center_y']*rc['area']    
    
    v_x= rc[['x_area', 'frame']].groupby('frame').agg(np.sum)['x_area'].values
    v_y= rc[['y_area', 'frame']].groupby('frame').agg(np.sum)['y_area'].values
    v_xx= rc[['xx_area', 'frame']].groupby('frame').agg(np.sum)['xx_area'].values
    v_xy= rc[['xy_area', 'frame']].groupby('frame').agg(np.sum)['xy_area'].values
    v_yy= rc[['yy_area', 'frame']].groupby('frame').agg(np.sum)['yy_area'].values
    av_a= rc[['area', 'frame']].groupby('frame').agg(np.sum)['area'].values

    m_xx= (v_xx - v_x*v_x/av_a)
    m_xy= (v_xy - v_x*v_y/av_a)
    m_yy= (v_yy - v_y*v_y/av_a)

    norm= m_xx*m_yy-m_xy**2
    m_xx= m_xx/norm
    m_yy= m_yy/norm
    m_xy= m_xy/norm
    s= 0.5*np.log(m_xx*m_yy - m_xy**2)
    Q = np.arcsinh(0.5*np.sqrt((m_xx-m_yy)**2.+(2*m_xy)**2.)/np.exp(s))
    twophi = np.arctan2((2*m_xy),(m_xx-m_yy))
    return Q*np.cos(twophi), Q*np.sin(twophi), s


def calculate_trinagle_state_properties(cr):
    """
    Calculates tringale state properties given triangle coordinates in format 'n_alpha' n= 1, 2, 3 and alpha= x, y ordered counter-clockwise.
    """
    cr['area']= 0.5*(-cr['2_x']*cr['1_y'] +
                     cr['3_x']*cr['1_y'] +
                     cr['1_x']*cr['2_y'] -
                     cr['3_x']*cr['2_y'] -
                     cr['1_x']*cr['3_y'] +
                     cr['2_x']*cr['3_y'])
    xx= cr['2_x'] - cr['1_x']
    xy= cr['3_x'] - cr['1_x']
    yx= cr['2_y'] - cr['1_y']
    yy= cr['3_y'] - cr['1_y']
    cr['xx']= xx
    cr['xy']= xy
    cr['yx']= yx
    cr['yy']= yy    
    scalingFactor= 1/(2*3**(1./4))
    cr['S_xx']= scalingFactor*(-1.*(xx + xy))
    cr['S_xy']= scalingFactor*np.sqrt(3)*(xx - xy)
    cr['S_yx']= scalingFactor*(-1.*(yx + yy))
    cr['S_yy']= scalingFactor*np.sqrt(3)*(yx - yy)
    cr['Q_kk']= np.log(cr['S_xx']*cr['S_yy'] - cr['S_xy']*cr['S_yx'])
    cr['theta']= np.arctan2(cr['S_yx'] - cr['S_xy'], cr['S_xx'] + cr['S_yy'])
    cr['two_phi']= (cr['theta'] +
                    np.arctan2(cr['S_xy'] + cr['S_yx'], cr['S_xx'] - cr['S_yy']))
    cr['Q']= np.arcsinh(0.5*
                        np.sqrt((cr['S_xx']-cr['S_yy'])**2.+(cr['S_xy']+cr['S_yx'])**2.)/
                        np.exp(0.5*cr['Q_kk']))


def table_with_triangle_coordinates(cr):
    ''' Requires table with (at least) columns 'tri_id', 'tri_order', 'center_x' -> other info is lost
        This is by default satisfiend by intermediate_state[] tables'''
    x= cr.pivot(index= 'tri_id',
                columns= 'tri_order',
                values= 'center_x')
    y= cr.pivot(index= 'tri_id',
                columns= 'tri_order',
                values= 'center_y')
    r= x.merge(y, left_index= True, right_index= True)
    return r.dropna()
