import matplotlib as mpl
import sqlite3 as lite
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import pandas.io.parsers as pp
# import rpy2.robjects as robjects
# import rpy2.robjects as ro
import pyreadr
import matplotlib.cm as cm
from matplotlib import collections as mc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path

import time

import tissue_miner_tools as tlm

class Movie:
    """
    Movie class is intended to contain all the data of a single time-lapse movie.

    It also contains methods for loading the data from SQL database and extra .RData
    tables provided by TissueMiner workflow.

    Note that it will create pickle (.pkl) format of tables provided in RData files
    to significantly reduce data loading time. Currently this is done trough intermediate
    step creating .csv file, which seems to be faster than using rpy2 library to
    directly read RData files into python for large files.
    Each time SQL database and/or RData files are updated piclke files will be updated.
    """
    def __init__(self, name, path, ROI_path= '', ROI_filename= '', refresh= False):
        self.name= name
        self.DB_path= path
        self.con= lite.connect(self.DB_path+name+'/'+name+'.sqlite')
        #self.con= lite.connect(self.DB_path+'/'+name+'.sqlite')
        self.loaded= set()
        self.DB_table= {}
        self.intermediate_state= {}
        self.triangle_state= {}
        self.pupalWing_loaded= False
        self.ROI_path= ROI_path
        self.ROI_filename= ROI_filename
        try:
            print('Loading table frames from database ' + self.name + '...')             
            self.DB_table['frames']= psql.read_sql("SELECT * FROM frames;", self.con)
            self.loaded.add('frames')
            self.load_DB_table('cells', refresh)
            self.frames= self.DB_table['cells']['frame'].unique()
            self.NFrames= len(self.frames)            
            self.time= self.DB_table['frames']['time_sec'].values/3600.
            self.time= self.time[:self.NFrames]             
            self.dt= self.time[1:] - self.time[:-1]
        except:
            print('Table frames not available in ' + self.name + ' database.')
        try:
            pupalWings= pp.read_csv(self.DB_path+'PupalWingMovies.csv', sep=',')
            self.pupalWing_loaded= True
        except:
            print('PupalWingMovies.csv not found in: '+self.DB_path)
        try:
            self.time_shift= float(np.array(pupalWings[pupalWings['name']==name]['time_shift_sec'])[0])/3600.
            if np.isnan(self.time_shift):
                print('While loading ' + name + ' time shift is NAN')
            else:    
                self.time+= 15. + self.time_shift
        except:
            print('While loading '+ name+'. No time shift provided!.')  
        try:
            self.load_roiBT(path= self.ROI_path, filename= self.ROI_filename)            
        except:
            print('ROI file is not available!')
            
    def load_DB_table(self, table_name, refresh= False):
        """
        Loads a table table_name from the SQL database associated with the class
        instance on initialization.
        If pickle format of the table exists and is newer than the database file,
        it will load the table from pickle file instead.

        Also checks whether the table is already loaded.
        """
        if not table_name in self.loaded:
            file_pickle= self.DB_path + self.name + '/' + table_name + '.pkl'
            file_sql= self.DB_path + self.name + '/' + self.name + '.sqlite'
            if ((not os.path.isfile(file_pickle)) or (os.path.getmtime(file_sql) > os.path.getmtime(file_pickle)) or (refresh)):
                print('Loading table ' + table_name + ' from database ' + self.name + '...')
                if table_name in ('vertices', 'bonds', 'frames', 'directed_bonds') :
                    self.DB_table[table_name]= psql.read_sql('SELECT * FROM ' + table_name + ';', self.con)
                else:
                    self.DB_table[table_name]= psql.read_sql('SELECT * FROM ' + table_name + ';', self.con)

                print('Writing table ' + table_name + ' to pickle file ' + self.name)
                self.DB_table[table_name].to_pickle(file_pickle)
            else:
                print('Loading table ' + table_name + ' from pickle file...') 
                self.DB_table[table_name]= pd.read_pickle(file_pickle)
            #if table_name == 'cells':
            #    self.DB_table[table_name]= self.DB_table[table_name][['frame', 'cell_id', 'center_x', 'center_y', 'area', 'elong_xx', 'elong_xy']]                
            self.loaded.add(table_name)
            
    def database_tables(self):
        """
        Returns a list of tables of the SQL database as DataFrame.
        """
        return psql.read_sql("SELECT name FROM sqlite_master WHERE type='table';", self.con)
    
    def RData_to_pickle(self, table, file_RData, file_pickle):
        """
        Writes a table from RData file to a pickle format if RData file is newer
        that the pickle file, or pickle file does not exist.
        """
        
        file_csv= file_RData[:-6] + '.csv'
        if ((not os.path.isfile(file_pickle)) or
            (os.path.getmtime(file_RData) > os.path.getmtime(file_pickle))):
            R_dic=pyreadr.read_r(file_RData)
            df=R_dic[table]
            df.to_csv(file_csv)
            # print('Converting \n'+ file_RData + ' to \n' + file_pickle)
            # print('Creating temporary .csv file...')            
            # ro.r('load("'+file_RData+'")')
            # ro.r('write.csv('+table+', "' + file_csv + '")')
            print('.csv created, writing to pickle')
            print('Reading temporary .csv file...')
            temp_csv= pd.read_csv(file_csv)
            print('Writing to pickle...')
            temp_csv.to_pickle(file_pickle)
        
    def load_roiBT(self, path, filename):
        """
        Loads ROI table.
        """
        if not 'roiBT' in self.loaded:
            file_pickle= self.DB_path + self.name + '/' + path + filename + '.pkl'
            file_RData= self.DB_path + self.name + '/' + path + filename + '.RData'
            print('Loading roiBT ...')
            self.RData_to_pickle(filename, file_RData, file_pickle)
            self.roiBT= pd.read_pickle(file_pickle)
            self.loaded.add('roiBT')
            self.regions= self.roiBT['roi'].unique()

    def load_cellshapes(self):
        """
        Loads cellshapes table.
        """
        if not 'cellshapes' in self.loaded:
            file_pickle= self.DB_path + self.name + '/cellshapes.pkl'
            file_RData= self.DB_path + self.name + '/cellshapes.RData'
            print('Loading cellshapes of ' + self.name + '...')
            self.RData_to_pickle('cellshapes', file_RData, file_pickle)
            self.cellshapes= pd.read_pickle(file_pickle)
            self.loaded.add('cellshapes')
            del self.cellshapes[self.cellshapes.columns[0]]

    def load_cellNeighbors(self):
        """
        Loads cellNeighbors table.
        """
        if not 'cellNeighbors' in self.loaded:
            file_pickle= self.DB_path + self.name + '/cellNeighbors.pkl'
            file_RData= self.DB_path + self.name + '/topochanges/cellNeighbors.RData'
            print('Loading cellNeighbors of ' + self.name + '...')
            self.RData_to_pickle('cellNeighbors', file_RData, file_pickle)
            self.cellNeighbors= pd.read_pickle(file_pickle)
            self.loaded.add('cellNeighbors')
            del self.cellNeighbors[self.cellNeighbors.columns[0]]
            
    def load_intermediate_state(self, state):
        """
        Loads precalculated triangle intermediate state table.
        """
        if state not in ['first', 'snd', 'third']:
            print('State should be one of the: "first", "snd", "third"')
            return -1
        if not state + 'Int' in self.loaded:
            file_pickle= self.DB_path + self.name + '/shear_contrib/' +state + 'Int.pkl'
            file_RData= self.DB_path + self.name + '/shear_contrib/' + state + 'Int.RData'
            print('Loading intermediate state ' + state + 'Int of ' + self.name + '...')
            self.RData_to_pickle(state+'Int', file_RData, file_pickle)
            self.intermediate_state[state]= pd.read_pickle(file_pickle)
            print('Done!')
            self.loaded.add(state+'Int')
            del self.intermediate_state[state][self.intermediate_state[state].columns[0]]

    def load_triangle_state(self, region='whole_tissue'):
        """
        Loads precalculated triangles states in a given region.
        """
        self.load_roiBT(self.ROI_path, self.ROI_filename)        
        if ((region == 'whole_tissue') and
            (not ('whole_tissue' in self.regions))):
            region= 'whole_wing'
        if ((not ('Ta_t_'+region in self.loaded)) and
            (region in self.regions)):
            file_pickle= self.DB_path+self.name+'/shear_contrib/'+region+'/Ta_t.pkl'
            file_RData= self.DB_path+self.name+'/shear_contrib/'+region+'/Ta_t.RData'    
            print('Loading triangle state table in ' + region + ' region of ' + self.name + '...')
            self.RData_to_pickle('Ta_t',file_RData, file_pickle)
            self.triangle_state[region]= pd.read_pickle(file_pickle)
            print('Done!')
            self.loaded.add('Ta_t_' + region)
            del self.triangle_state[region][self.triangle_state[region].columns[0]]
        if not (region in self.regions):
            print('Region ' + region + ' is not available')

    def load_cellinfoDB(self):
        """
        Loads cellinfo table.
        """
        if not 'cellinfoDB' in self.loaded:
            file_pickle= self.DB_path + self.name + '/cellsinfoDB.pkl'
            file_RData= self.DB_path + self.name + '/cellinfoDB.RData'
            print('Loading cellinfoDB of ' + self.name + '...')
            self.RData_to_pickle('cellinfoDB', file_RData, file_pickle)
            self.cellinfoDB= pd.read_pickle(file_pickle)
            self.loaded.add('cellinfoDB')
            del self.cellinfoDB[self.cellinfoDB.columns[0]]
                
    def load_triangle_list(self):
        """
        Loads triangle list table.
        """
        if not 'triangle_list' in self.loaded:
            file_pickle= self.DB_path+self.name+'/shear_contrib/triList.pkl'
            file_RData= self.DB_path+self.name+'/shear_contrib/triList.RData'
            print('Loading triList...')
            self.RData_to_pickle('triList', file_RData, file_pickle)
            self.triangle_list= pd.read_pickle(file_pickle)
            print('Done!')            
            self.loaded.add('triangle_list')
            del self.triangle_list[self.triangle_list.columns[0]]
            self.load_DB_table('cells')
            self.triangle_list= self.triangle_list.merge(self.DB_table['cells'][['frame', 'cell_id', 'center_x', 'center_y']], on= ['frame', 'cell_id'])            
    def load_triangle_categories(self):
        if not 'triangle_categories' in self.loaded:
            file_pickle= self.DB_path+self.name+'/tri_categories/triangleCategories.pkl'
            file_RData= self.DB_path+self.name+'/tri_categories/triangleCategories.RData'
            print('Loading triangle categories table...')
            self.RData_to_pickle('triangleCategories',file_RData, file_pickle)
            self.triangle_categories= pd.read_pickle(file_pickle)
            print('Done!')
            self.loaded.add('triangle_categories')

    def load_topo_change_summary(self):
        if not 'topo_change_summary' in self.loaded:
            file_pickle= self.DB_path+self.name+'/topochanges/topoChangeSummary.pkl'
            file_RData= self.DB_path+self.name+'/topochanges/topoChangeSummary.RData'
            print('Loading topo_change_summary table...')
            self.RData_to_pickle('topoChangeSummary',file_RData, file_pickle)
            self.topo_change_summary= pd.read_pickle(file_pickle)
            print('Done!')
            self.loaded.add('topo_change_summary')            

    def load_T1_data_filt(self):
        if not 'T1_data_filt' in self.loaded:
            file_pickle= self.DB_path+self.name+'/topochanges/t1DataFilt.pkl'
            file_RData= self.DB_path+self.name+'/topochanges/t1DataFilt.RData'
            print('Loading T1_data_filt table...')
            self.RData_to_pickle('t1DataFilt',file_RData, file_pickle)
            self.T1_data_filt= pd.read_pickle(file_pickle)
            print('Done!')
            self.loaded.add('T1_data_filt')            


    def show_image(self, frame):
        """
        Shows original (rotated) image of the given frame.
        """
#        im_path= self.DB_path + self.name + '/Segmentation/' + self.name + '_' + tlm.fill_zeros(str(frame), 3) + '/original_trafo.png'
        im_path= self.DB_path + self.name + '/Segmentation/' + self.name + '_' + tlm.fill_zeros(str(frame), 3) + '/original.png'
        im= plt.imread(im_path)
        plt.imshow(im)
                                    
    def region_cells(self, region):
        """
        Returns cells DataFrame with cells belonging to the region ROI.
        """
        self.load_DB_table('cells')
        if region not in self.regions:
            raise Exception('Region '+region+' is not defined in this movie!')
        else:
            return self.DB_table['cells'][self.DB_table['cells']['cell_id'].isin(self.roiBT[self.roiBT['roi']==region]['cell_id'])]
                
    def plot_frame_cells(self, frame, coll_df, color_column, c_min= 0., c_max= 1., n_ticks= 5, figsize= (6, 10), polygon_lw= .1, color_map= cm.afmhot, title= ''):
        """
        Plots a collection of polygons provided in coll_df DataFrame in 'plot_vertices' column.
        Color is assigned based on values in color_column column of the coll_df DataFrame.
        c_min and c_max control the range of the colormap.
        Colormap can be provided by user and is set to afmhot by default.
        """
        plt.figure(figsize= figsize)
        plt.title(title, fontsize= 25)
        self.show_image(frame)
        plt.gca().autoscale_view()
        plt.gca().set_aspect('equal')
        colors= color_map((coll_df[color_column].values-c_min)/(c_max - c_min)) 
        coll= mc.PolyCollection(coll_df['plot_vertices'].values, lw= polygon_lw)
        coll.set(facecolors= colors)
        plt.gca().add_collection(coll)
        plt.xlim(0, 1800)
        plt.ylim(0, 1800)
        plt.gca().invert_yaxis()
        plt.axis('off')
        divider= make_axes_locatable(plt.gca())
        cax= divider.append_axes('right', size= '5%', pad= 0.05)
        mm= cm.ScalarMappable(cmap= color_map)
        mm.set_array(colors)
        cbar= plt.colorbar(mm, cax= cax, cmap= color_map, ticks= np.linspace(0, 1, n_ticks + 1))
        cbar.ax.set_yticklabels(np.linspace(c_min, c_max, n_ticks + 1))
    
    
    def region_deform_tensor(self, roi_name):
        """
        Loads precalculated deformation data.
        """
        if 'avgDeformTensorsWide.tsv' in os.listdir(self.DB_path+self.name+'/shear_contrib/'+roi_name):
            df_DB_shear= pp.read_csv(self.DB_path+self.name+'/shear_contrib/'+roi_name+'/avgDeformTensorsWide.tsv', sep='\t')
        return df_DB_shear

    #def subset_table_by_region(self, df, region= 'blade', on_column= 'cell_id'):
    def subset_table_by_region(self, df, region= 'vein', on_column= 'cell_id'):

        """
        Returns subset of DataFrame table with the rows corresponding to
        'cell_id' values in a given region.
        """
        rc= self.region_cells(region)['cell_id'].unique()
        return df[df[on_column].isin(rc)]

    def cell_number(self, region= 'whole_tissue'):
        """
        Calculates number of cells in a region for each frame.
        """
        region_cells= self.region_cells(region)
        nr_cells= region_cells[['frame', 'cell_id']].groupby('frame').agg(len).reset_index().rename(columns= {'cell_id': 'nr_cells'})
        return nr_cells

    def add_dbonds_coordinates(self):
        """
        Adds positions, vectors and vectors of left bond to the dbonds table
        """
        self.load_DB_table('vertices')
        self.load_DB_table('directed_bonds')
        self.bonds_full= pd.merge(pd.merge(self.DB_table['directed_bonds'],
                                           self.DB_table['directed_bonds'][['vertex_id', 'conj_dbond_id', 'cell_id']],
                                           left_on= 'dbond_id',
                                           right_on= 'conj_dbond_id'),
                                self.DB_table['vertices'][['vertex_id', 'x_pos', 'y_pos']],
                                left_on= 'vertex_id_x',
                                right_on= 'vertex_id')
        self.bonds_full= pd.merge(self.bonds_full,
                                  self.DB_table['vertices'][['vertex_id', 'x_pos', 'y_pos']],
                                  left_on= 'vertex_id_y',
                                  right_on= 'vertex_id', suffixes= ['', '_2'])
        self.bonds_full['dx']= self.bonds_full['x_pos_2'] - self.bonds_full['x_pos']
        self.bonds_full['dy']= self.bonds_full['y_pos_2'] - self.bonds_full['y_pos']
        self.dbonds= pd.merge(self.DB_table['directed_bonds'],
                              self.bonds_full[['dbond_id', 'x_pos', 'y_pos', 'dx', 'dy', 'cell_id_y']],
                              left_on= 'dbond_id',
                              right_on= 'dbond_id').rename(columns= {'cell_id_y': 'conj_cell_id'})
        del self.bonds_full
        self.dbonds['length']= np.sqrt(self.dbonds['dx']**2. + self.dbonds['dy']**2.)
        self.dbonds['angle']= np.arctan2(self.dbonds['dy'],
                                         self.dbonds['dx'])

    #def shear_decomposition(self, region= 'blade', N_steps= 100):
    def shear_decomposition(self, region= 'vein', N_steps= 100):

        """
        Calculates shear decomposition components and adds the corresponding variables to the movie instance.
        """
        #### loading trinagle states and intermediate states
        self.load_triangle_state(region)
        self.load_intermediate_state('first')
        self.load_intermediate_state('snd')
        self.load_intermediate_state('third')

        #### calculating triangle state properties
        self.firstInt= self.subset_table_by_region(self.intermediate_state['first'], region)
        self.first_r= tlm.table_with_triangle_coordinates(self.firstInt)
        tlm.calculate_trinagle_state_properties(self.first_r)
        self.sndInt= self.subset_table_by_region(self.intermediate_state['snd'], region)
        self.snd_r= tlm.table_with_triangle_coordinates(self.sndInt)
        tlm.calculate_trinagle_state_properties(self.snd_r)

        self.thirdInt= self.subset_table_by_region(self.intermediate_state['third'], region)
        self.third_r= tlm.table_with_triangle_coordinates(self.thirdInt)
        tlm.calculate_trinagle_state_properties(self.third_r)

        self.state_r= self.triangle_state[region]

        #### adding frame info
        self.first_triFrame= self.firstInt[self.firstInt['tri_order'] == 1][['frame', 'tri_id']]
        self.first_r= self.first_r.reset_index().merge(self.first_triFrame, on= 'tri_id', how= 'left')

        self.snd_triFrame= self.sndInt[self.sndInt['tri_order'] == 1][['frame', 'tri_id']]
        self.snd_r= self.snd_r.reset_index().merge(self.snd_triFrame, on= 'tri_id', how= 'left')

        self.third_triFrame= self.thirdInt[self.thirdInt['tri_order'] == 1][['frame', 'tri_id']]
        self.third_r= self.third_r.reset_index().merge(self.third_triFrame, on= 'tri_id', how= 'left')

        self.load_triangle_list()
        self.state_triFrame= self.triangle_list[self.triangle_list['tri_order'] == 1][['frame', 'tri_id']]
        self.state_r= self.state_r.merge(self.state_triFrame, on= 'tri_id', how= 'left')

        #### calculating average elongation
        self.state_r['Q_xx_w']= self.state_r['Q_a']*np.cos(self.state_r['two_phi_a'])*self.state_r['tri_area']
        self.first_r['Q_xx_w']= self.first_r['Q']*np.cos(self.first_r['two_phi'])*self.first_r['area']
        self.snd_r['Q_xx_w']= self.snd_r['Q']*np.cos(self.snd_r['two_phi'])*self.snd_r['area']
        self.third_r['Q_xx_w']= self.third_r['Q']*np.cos(self.third_r['two_phi'])*self.third_r['area']
        self.state_Qxx_avg= self.state_r[['frame', 'Q_xx_w']].groupby('frame').agg(np.sum)['Q_xx_w'].values/self.state_r[['frame', 'tri_area']].groupby('frame').agg(np.sum)['tri_area'].values
        self.first_Qxx_avg= self.first_r[['frame', 'Q_xx_w']].groupby('frame').agg(np.sum)['Q_xx_w'].values/self.first_r[['frame', 'area']].groupby('frame').agg(np.sum)['area'].values
        self.snd_Qxx_avg= self.snd_r[['frame', 'Q_xx_w']].groupby('frame').agg(np.sum)['Q_xx_w'].values/self.snd_r[['frame', 'area']].groupby('frame').agg(np.sum)['area'].values
        self.third_Qxx_avg= self.third_r[['frame', 'Q_xx_w']].groupby('frame').agg(np.sum)['Q_xx_w'].values/self.third_r[['frame', 'area']].groupby('frame').agg(np.sum)['area'].values

        self.state_r['Q_xy_w']= self.state_r['Q_a']*np.sin(self.state_r['two_phi_a'])*self.state_r['tri_area']
        self.first_r['Q_xy_w']= self.first_r['Q']*np.sin(self.first_r['two_phi'])*self.first_r['area']
        self.snd_r['Q_xy_w']= self.snd_r['Q']*np.sin(self.snd_r['two_phi'])*self.snd_r['area']
        self.third_r['Q_xy_w']= self.third_r['Q']*np.sin(self.third_r['two_phi'])*self.third_r['area']
        self.state_Qxy_avg= self.state_r[['frame', 'Q_xy_w']].groupby('frame').agg(np.sum)['Q_xy_w'].values/self.state_r[['frame', 'tri_area']].groupby('frame').agg(np.sum)['tri_area'].values
        self.first_Qxy_avg= self.first_r[['frame', 'Q_xy_w']].groupby('frame').agg(np.sum)['Q_xy_w'].values/self.first_r[['frame', 'area']].groupby('frame').agg(np.sum)['area'].values
        self.snd_Qxy_avg= self.snd_r[['frame', 'Q_xy_w']].groupby('frame').agg(np.sum)['Q_xy_w'].values/self.snd_r[['frame', 'area']].groupby('frame').agg(np.sum)['area'].values
        self.third_Qxy_avg= self.third_r[['frame', 'Q_xy_w']].groupby('frame').agg(np.sum)['Q_xy_w'].values/self.third_r[['frame', 'area']].groupby('frame').agg(np.sum)['area'].values    
    
        #### total shear and correlation terms
        self.firstInt= self.subset_table_by_region(self.intermediate_state['first'], region)
        self.first_r= tlm.table_with_triangle_coordinates(self.firstInt)
        tlm.calculate_trinagle_state_properties(self.first_r)

        self.sndInt= self.subset_table_by_region(self.intermediate_state['snd'], region)
        self.snd_r= tlm.table_with_triangle_coordinates(self.sndInt)
        tlm.calculate_trinagle_state_properties(self.snd_r)

        triData= self.intermediate_state['first'][self.intermediate_state['first']['tri_order'] == 1][['tri_id', 'frame']]

        av_j_xx, av_j_xy= 0, 0
        av_J_xx, av_J_xy= 0, 0
        av_ukk_qxx, av_ukk_qxy= 0, 0
        av_Ukk_Qxx, av_Ukk_Qxy= 0, 0
        av_N_xx, av_N_xy= 0, 0

        for i in range(N_steps):
            print (i)
            if i > 0:
                first= snd
            else:
                first= self.first_r + (self.snd_r - self.first_r)*1.*i/N_steps
                tlm.calculate_trinagle_state_properties(first)
                first['Qxx']= first['Q']*np.cos(first['two_phi'])
                first['Qxy']= first['Q']*np.sin(first['two_phi'])        
            snd= self.first_r + (self.snd_r - self.first_r)*(i+1.)/N_steps
            tlm.calculate_trinagle_state_properties(snd)
        
            snd['Qxx']= snd['Q']*np.cos(snd['two_phi'])
            snd['Qxy']= snd['Q']*np.sin(snd['two_phi'])    
            fs= first[['area', 'Q_kk', 'Q', 'theta', 'two_phi', 'Qxx', 'Qxy', 'xx', 'xy', 'yx', 'yy']].merge(snd[['area','Q_kk', 'Q', 'theta', 'two_phi', 'Qxx', 'Qxy', 'xx', 'xy', 'yx', 'yy']], left_index= True, right_index= True, how= 'inner', suffixes= ('_f', '_s'))

            fs['T_denom']= -fs['xy_f']*fs['yx_f'] + fs['xx_f']*fs['yy_f']
            fs['T_xx']= (-fs['xy_s']*fs['yx_f'] + fs['xx_s']*fs['yy_f'])/fs['T_denom']
            fs['T_xy']= (fs['xy_s']*fs['xx_f'] - fs['xx_s']*fs['xy_f'])/fs['T_denom']
            fs['T_yx']= (-fs['yy_s']*fs['yx_f'] + fs['yx_s']*fs['yy_f'])/fs['T_denom']
            fs['T_yy']= (fs['yy_s']*fs['xx_f'] - fs['yx_s']*fs['xy_f'])/fs['T_denom']
            fs['N_xx']= .5*(fs['T_xx'] - fs['T_yy'])
            fs['N_xy']= .5*(fs['T_xy'] + fs['T_yx'])
    
            fs['c']= np.tanh(2*fs['Q_f'])/(2*fs['Q_f'])
            fs['delta_phi']= .5*tlm.angle_difference(fs['two_phi_s'],
                                                 fs['two_phi_f'])
            fs['delta_theta']= tlm.angle_difference(fs['theta_s'],
                                                fs['theta_f'])
            fs['delta_psi']= tlm.angle_difference(fs['delta_phi'],
                                              tlm.angle_difference(fs['delta_phi'],
                                                               fs['delta_theta'])*np.cosh(2*fs['Q_f']))
            fs['jxx']= 2*(fs['c']*fs['delta_psi'] +
                          (1. - fs['c'])*fs['delta_phi'])*fs['Qxy_f']
            fs['jxy']= -2*(fs['c']*fs['delta_psi'] +
                           (1. - fs['c'])*fs['delta_phi'])*fs['Qxx_f']
            fs['delta_ukk']= np.log(fs['area_s']/fs['area_f'])
            fs.dropna(inplace= True) ######### this should not be neccessary if triangulation was perfect and triangles in initial and last frames properly selected and all triangles were of finite area :)
            fsf= fs.merge(triData, left_index= True, right_on= 'tri_id')
            gg= fsf.groupby('frame')
            area_norm_first= gg['area_f'].transform('sum')
            area_norm_snd= gg['area_s'].transform('sum')
            fsf['N_xx_wa']= fsf['N_xx']/area_norm_first*fsf['area_f']
            fsf['N_xy_wa']= fsf['N_xy']/area_norm_first*fsf['area_f']   
            fsf['Qxx_f_wa']= fsf['Qxx_f']/area_norm_first*fsf['area_f']
            fsf['Qxy_f_wa']= fsf['Qxy_f']/area_norm_first*fsf['area_f']
            fsf['Qxx_s_wa']= fsf['Qxx_s']/area_norm_snd*fsf['area_s']
            fsf['Qxy_s_wa']= fsf['Qxy_s']/area_norm_snd*fsf['area_s']
            fsf['jxx_wa']= fsf['jxx']/area_norm_first*fsf['area_f']
            fsf['jxy_wa']= fsf['jxy']/area_norm_first*fsf['area_f']
            fsf['delta_ukk_qxx_wa']= fsf['delta_ukk']*fsf['Qxx_f_wa']
            fsf['delta_ukk_qxy_wa']= fsf['delta_ukk']*fsf['Qxy_f_wa']
            fsf['delta_psi_wa']= fsf['delta_psi']/area_norm_first*fsf['area_f']
            fsf['delta_ukk_wa']= fsf['delta_ukk']/area_norm_first*fsf['area_f']
            av_N_xx+= gg['N_xx_wa'].sum()
            av_N_xy+= gg['N_xy_wa'].sum()            
            av_Qxx_f= gg['Qxx_f_wa'].sum()
            av_Qxy_f= gg['Qxy_f_wa'].sum()
            av_Q_f= np.sqrt(av_Qxx_f**2 + av_Qxy_f**2)
            av_phi_f= .5*np.angle(av_Qxx_f + 1j*av_Qxy_f)
            av_Qxx_s= gg['Qxx_s_wa'].sum()
            av_Qxy_s= gg['Qxy_s_wa'].sum()
            av_Q_s= np.sqrt(av_Qxx_f**2 + av_Qxy_f**2)
            av_phi_s= .5*np.angle(av_Qxx_s + 1j*av_Qxy_s)
            av_delta_phi= tlm.angle_difference(av_phi_s, av_phi_f)
            av_delta_psi= gg['delta_psi_wa'].sum()
            av_C= np.tanh(2*av_Q_f)/(2*av_Q_f)
            av_Ukk= gg['delta_ukk_wa'].sum()
            av_ukk_qxx+= gg['delta_ukk_qxx_wa'].sum()
            av_Ukk_Qxx+= av_Ukk*av_Qxx_f
            av_ukk_qxy+= gg['delta_ukk_qxy_wa'].sum()
            av_Ukk_Qxy+= av_Ukk*av_Qxy_f    
            av_j_xx+= gg['jxx_wa'].sum()
            av_j_xy+= gg['jxy_wa'].sum()    
            av_J_xx+= 2*(av_C*av_delta_psi +
                         (1 - av_C)*av_delta_phi)*av_Qxy_f
            av_J_xy+= -2*(av_C*av_delta_psi +
                          (1 - av_C)*av_delta_phi)*av_Qxx_f

        self.total_shear_xx= av_N_xx
        self.shear_cell_elongation_xx= self.state_Qxx_avg[1:] - self.state_Qxx_avg[:-1]
        self.shear_T1_xx= self.snd_Qxx_avg - self.third_Qxx_avg[1:]
        self.shear_cell_division_xx= (self.third_Qxx_avg - self.state_Qxx_avg)[1:]
        self.shear_cell_extrusion_xx= (self.state_Qxx_avg - self.first_Qxx_avg)[:-1]
        self.corotational_xx= av_J_xx
        self.rotation_elongation_correlation_xx= av_j_xx - av_J_xx
        self.area_growth_correlation_elongation_correlation_xx= -(av_ukk_qxx - av_Ukk_Qxx)
        self.total_shear_xy= av_N_xy
        self.shear_cell_elongation_xy= self.state_Qxy_avg[1:] - self.state_Qxy_avg[:-1]
        self.shear_T1_xy= self.snd_Qxy_avg - self.third_Qxy_avg[1:]
        self.shear_cell_division_xy= (self.third_Qxy_avg - self.state_Qxy_avg)[1:]
        self.shear_cell_extrusion_xy= (self.state_Qxy_avg - self.first_Qxy_avg)[:-1]
        self.corotational_xy= av_J_xy
        self.rotation_elongation_correlation_xy= av_j_xy - av_J_xy
        self.area_growth_correlation_elongation_correlation_xy= -(av_ukk_qxy - av_Ukk_Qxy) 
        
