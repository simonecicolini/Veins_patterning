#packages to import
import pandas as pd
import numpy as np
import scipy
import tissue_miner_tools as tml
import tissue_miner as tm
import scipy.spatial.distance
import matplotlib as mpl
# matplotlib.use('agg') #this command is necessary on the cluster
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as mc
import os 
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from scipy.optimize import curve_fit
import math
pd.options.mode.chained_assignment = None
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##run this block to simulate WT veins
Neighbors_info='./wing_movies/200923/cellNeighbors.csv' #data_base containing neighbors relationships
data_base='./wing_movies/200923/DB.csv' #data_base containing all the relevant quantities for each cell at each time frame, like DSRF level, cell centroids, lineages,...
DB=pd.read_csv(data_base);DB=DB.drop(['Unnamed: 0'],axis=1)
movieDatabaseDir='./wing_movies/' 
name='200923'
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##run this block to simulate Dumpy mutant veins
# Neighbors_info='./wing_movies/200924_wing1/cellNeighbors.csv' #data_base containing neighbors relationships
# data_base='./wing_movies/200924_wing1/DBcells2.csv' #data_base containing all the relevant quantities for each cell at each time frame, like DSRF level, cell centroids, lineages,...
# DB=pd.read_csv(data_base);DB=DB.drop(['Unnamed: 0'],axis=1)
# movieDatabaseDir='./wing_movies/' 
# name='200924_wing1'
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

##import movie info using the tissue_miner class
movieRoiPath= 'roi_bt/' 
movieRoiFile= 'lgRoiSmoothed'            
movie=tm.Movie(name, path= movieDatabaseDir, ROI_path= movieRoiPath, ROI_filename= movieRoiFile)
movie.load_cellshapes()
Neighbors=pd.read_csv(Neighbors_info)

#initialization
DB['u']=0;DB['diffusion_term']=0;DB['kernel_term']=0;DB['signaling_force']=0;DB['mcherry']=0
#The cell 10 000 is not an actual cell but the border of the tissue:
DB=DB[DB.cell_id!=10000]

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##This block calculate the distance from vein L3 (Run it only to simulate the cross vein)
# DB_cells2=DB
# th=500 #threshold of DSRF concentration
# # we use a time varying paralelogram to select cells around a vein
# y10=55 
# y1f=55
# y20=110
# y2f=110
# L0=37.4
# Lf=37.4
# xM=325.6
# xm=0
# X=np.linspace(xm,xM,int(xM-xm))
# Y=np.zeros((234,int(xM-xm)))
# Theta=np.zeros((234,int(xM-xm)))#we store the angle that the curve is making for the rotation later
# for fr in range(234):
#     print(fr)
#     y2=y20+fr/233*(y2f-y20)
#     y1=y10+fr/233*(y1f-y10)
#     L=L0+fr/233*(Lf-L0)
    
#     xtest = (DB_cells2.center_x>=xm) & (DB_cells2.center_x<=xM)
#     ytestm = (DB_cells2.center_y >= y1+(DB_cells2.center_x-xm)*(y2-y1)/(xM-xm)-L)
#     ytestM = (DB_cells2.center_y <= y1+(DB_cells2.center_x-xm)*(y2-y1)/(xM-xm)+L)
    
#     Vein=DB_cells2[(DB_cells2.frame==fr) & (DB_cells2.DSRF_Conc<th) & xtest & ytestm & ytestM]
#     p=np.polyfit(Vein.center_x,Vein.center_y,3)
#     Y[fr,:]=p[0]*X**3+p[1]*X**2+p[2]*X+p[3]

# DB_cells2['dist']=0
# for fr in range(234):
#     print(fr)
#     cx=DB_cells2[DB_cells2.frame==fr].center_x.to_numpy()
#     cy=DB_cells2[DB_cells2.frame==fr].center_y.to_numpy()
#     Dist=np.subtract.outer(cx,X)**2+np.subtract.outer(cy,Y[fr,:])**2
#     Arg=np.argmin(Dist,axis=1)
#     #inverting sign of minimum distance based on sign of cy-Y[fr,argmin]
#     Sgn=(cy-Y[fr,Arg])<0
#     #the angle to rotate is the angle of the fit curve at the point that minimizes the distance
#     #Theta_cell=Theta[fr,Arg]
#     Distmin=np.sqrt(Dist[range(Dist.shape[0]),Arg])
#     Distmin[Sgn]=-Distmin[Sgn]
#     DB_cells2.loc[DB_cells2.frame==fr,'dist']=Distmin

# DB=DB_cells2
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#parameters (time scales are expressed in frames (1 frame=5 min=1/12 hours) and length scales in microns)
#dx is the average distance between cells
#kl is the ratio between the intercellular distance dx and the basal protrusions length l
#rho is the strength of notch inhibition (called J^I in the paper)
#here D_tau/dx^2 is the activation strength (called J^A in the paper)
#sigma is a parameter asscociated to the strength of the Langevin noise
#tauprime half-time mcherry reporter (Notch activity)
#u_I and alpha_I are associated to the threshold for Delta-Notch cis-inhibition
kl=2; r=0.1; rho=0.4; tau=1.5; D=2.5/tau; alpha_I=0.075; u_I=0.52; tauprime=20*12; D_tau=D*tau; sigma=0.0026

dx=4 # average distance between cells np.sqrt(DB.area.mean()*2/(np.sqrt(3)));
l0=kl*dx;
dt=0.2 #time step
StepsPerFrame=5
Boundary='Simulation'

## m and M are respectively the first and the second peak in the histogram of DSRF level
##++++++++++++++++++++++++++++++++++++++++++++++++
##use these values for WT:
m=401. #WT_200923 
M=1007. #WT_200923
##use these values are for Dumpy mutant:
#m=448. #DUMPY
#M=1128. #DUMPY
##++++++++++++++++++++++++++++++++++++++++++++++++


#inital condition for the simulation
DB.loc[DB.frame==0,'u']=(1-(DB[DB.frame==0].DSRF_Conc-m)/(M-m))
##initial condition for cross vein
#DB.loc[(DB.frame==0)&(DB.dist>0)&(DB.dist<60)&(DB.center_x>80)&(DB.center_x<130),'u']=1 #large cross vein
#DB.loc[(DB.frame==0)&(DB.dist>0)&(DB.dist<60)&(DB.center_x>90)&(DB.center_x<120),'u']=1 #small cross vein

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##folder to store 
param='tau='+str(tau)+'_r='+str(r)+'Dtau='+str(D_tau)+'_kl=2_alpha_I=0.075_rho='+str(rho)+'u_I=0.52'
if Boundary not in os.listdir('./'):
    os.mkdir(Boundary)
if param not in os.listdir('./'+Boundary):
    os.mkdir('./'+Boundary+'/'+param)

## Functions 
#autonomous dynamics part
def f(x,r)  :
      return (r - x)*(-1 + x)*x

#This function returns the distance between each cell of the tissue at a given time point 
def get_distances(DataB,FRAME_VALUE) :
    DataB_f=DataB.loc[DataB.frame==FRAME_VALUE]
    dist=scipy.spatial.distance.cdist(DataB_f[['center_x','center_y']].to_numpy(),DataB_f[['center_x','center_y']].to_numpy())
    return dist

#This function calculates the diffusion term for a given cell given its neighbors : 
def calculate_diffusion_term(CELL,NEIGHBORS,DataB_f): 
    diff=DataB_f[DataB_f.cell_id.isin(NEIGHBORS)].u.sum()
    diff+=-len(NEIGHBORS)*DataB_f[DataB_f.cell_id==CELL].u.values[0]
    diff=D*tau*diff/dx**2
    return diff

#This function calculates the kernel term (Notch inhibition) for a given cell given the state of all other cells: 
def calculate_kernel_term(CELL,DataB_f):
    ##get the row number of the cell in the database
    cell_row=DataB_f[DataB_f.cell_id==CELL].iloc[0]
    ## twisted way used to match with the distances array that has indices and not cell ids
    index_cell=DataB_f.index.get_loc(cell_row.name)
    s=np.sum(np.exp(-distances[index_cell,:]/l0)*DataB_f.u.values)/np.sum(np.exp(-distances[index_cell,:]/l0))
    #function I(u) to account for the effect of cis-inhibition
    cis_inhib=(1-math.tanh((DataB_f[DataB_f.cell_id==CELL].u.values-u_I)/alpha_I))/2
    return -rho*cis_inhib*s


def plot_frame_cells(MOVIE, frame, location, coll_df, color_column, c_min= 0., c_max= 1., n_ticks= 5, figsize= (6, 10), polygon_lw= .1, color_map= cm.afmhot, title= ''):
        """
        Plots a collection of polygons provided in coll_df DataFrame in 'plot_vertices' column.
        Color is assigned based on values in color_column column of the coll_df DataFrame.
        c_min and c_max control the range of the colormap.
        Colormap can be provided by user and is set to afmhot by default.
        """
        plt.figure(figsize= figsize);
        plt.title(title, fontsize= 25)
        MOVIE.show_image(frame)
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
        #divider= make_axes_locatable(plt.gca())
        #cax= divider.append_axes('right', size= '5%', pad= 0.05)
        mm= cm.ScalarMappable(cmap= color_map)
        mm.set_array(colors)
        #cbar= plt.colorbar(mm, cax= cax, cmap= color_map, ticks= np.linspace(0, 1, n_ticks + 1))
        #cbar.ax.set_yticklabels(np.linspace(c_min, c_max, n_ticks + 1))
        plt.savefig(location+str(frame)+'.tif',format='tif',dpi=120) 
        plt.close()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##functions to fit the second peak (M) in the histogram of DSRF concentration
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)
##use this definition for WT:
def bimodal(x, sigma1, A1, mu2, sigma2, A2): #WT
    return gauss(x,401,sigma1,A1)+gauss(x,mu2,sigma2,A2) #the number 401 is the DSRF level of the first peak, obtained by fitting a sum of two gaussians to the DSRF histogram containing all the time frames.
##use this definition for Dumpy:
# def bimodal(x, sigma1, A1, mu2, sigma2, A2): #DPY
#     return gauss(x,448,sigma1,A1)+gauss(x,mu2,sigma2,A2) #the number 448 is the DSRF level of the first peak, obtained by fitting a sum of two gaussians to the DSRF histogram containing all the time frames.
##initial guess for fitting M (second peak in DSRF histogram)
expected=(50,90,1000,350,50)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#find cells in the border      
FRAME=0
frame_cellshapes=tml.subset_dataframe(movie.cellshapes, ['frame'], [FRAME])    
DB_f1=DB.loc[DB.frame==FRAME] #subset of the database for this frame
Neighbors_f=Neighbors.loc[Neighbors.frame==FRAME]
DB_f1['neighbors']=DB_f1['cell_id'].transform(lambda x: Neighbors_f[(Neighbors_f.cell_id==x)].neighbor_cell_id.unique())
DB_f1['NeighborsCount']=DB_f1['neighbors'].apply(lambda x: len(x))
#This next part selects the cells at the border of the tissue (within 3 cell layers) to set the boundary condition:
Differences=[]
for cell in DB_f1.cell_id.unique():
    Differences.append((-DB_f1[DB_f1.cell_id==cell].NeighborsCount.values+frame_cellshapes[frame_cellshapes.cell_id==cell].bond_order.max())[0])
DB_f1['Border']=Differences
Border_cells=DB_f1[DB_f1.Border!=0].cell_id.unique()
Border_And_Neighbors=list(Neighbors_f[Neighbors_f.cell_id.isin(Border_cells)].neighbor_cell_id.unique())
Border_And_Neighborsx2=list(Neighbors_f[Neighbors_f.cell_id.isin(Border_And_Neighbors)].neighbor_cell_id.unique())


for FRAME in tqdm(range(233)):
    #------------------------------------------------------------------------
    #this part plot the simulated tissue for this frame:
    frame_cellshapes=tml.subset_dataframe(movie.cellshapes, ['frame'], [FRAME])    
    frame_polygons_0=frame_cellshapes.groupby('cell_id').apply(lambda x: list(zip(x['x_pos'].values, x['y_pos'].values))).reset_index().rename(columns= {0: 'plot_vertices'})
    frame_polygons=DB[DB.frame==FRAME][[u'frame', u'cell_id', u'center_x', u'center_y', u'DSRF',u'DSRF_Conc', u'u']].merge(frame_polygons_0, on= 'cell_id')
    plot_frame_cells(movie,FRAME,'./'+Boundary+'/'+param+'/frame_', frame_polygons, title= 'u', color_column= 'u', c_min= -0.1, c_max= 1.05, color_map=cm.plasma) #cm.gist_rainbow)
    #------------------------------------------------------------------------

    DB_f=DB_f1 #subset of the database for this frame
    distances=get_distances(DB,FRAME)

    #perform the simulation for cells inside the tissue (not in the border)
    DB_f_in=DB_f[~DB_f.cell_id.isin(Border_And_Neighborsx2)]
    if FRAME<53: # frame corresponding to 21h
        for i in range(StepsPerFrame):                                
                DB_f_in['u']=DB_f_in['u']
                DB_f.loc[~DB_f.cell_id.isin(Border_And_Neighborsx2),'u']=DB_f_in['u']
    else:
        for i in range(StepsPerFrame): 
                length_u=len(DB_f_in['u'])
                noise=np.random.normal(0,np.sqrt(sigma*dt),length_u) #noise 
 
                DB_f_in['diffusion_term']=DB_f_in.apply(lambda x: calculate_diffusion_term(x.cell_id,x.neighbors,DB_f),axis=1)
                DB_f_in['kernel_term']=DB_f_in['cell_id'].apply(lambda x : calculate_kernel_term(x,DB_f))
                DB_f_in['signaling_force']=dt*1./tau*(DB_f_in['diffusion_term']+DB_f_in['kernel_term']+f(DB_f_in.u,r))
                DB_f_in['mcherry']=DB_f_in['mcherry']+dt*(-DB_f_in['kernel_term']-DB_f_in['mcherry']/tauprime)
                
                DB_f_in['u']=DB_f_in['u']+DB_f_in['signaling_force']+noise
                DB_f.loc[~DB_f.cell_id.isin(Border_And_Neighborsx2),'u']=DB_f_in['u']
            

    DB_f[~DB_f.cell_id.isin(Border_And_Neighborsx2)]=DB_f_in   
    # deal with the case of cells that disappear between this cell and the next one:
    disappearing_cells=DB_f[(DB_f.last_occ==FRAME)].cell_id.unique()
    disappearing_cells_division=DB_f[(DB_f.disappears_by=='Division')&(DB_f.last_occ==FRAME)].cell_id.unique()
    #this part treats with an error happening when the number of cells between two frames don't match
    list_cells_t=DB[(DB.frame==FRAME)& (~DB.cell_id.isin(disappearing_cells))].cell_id.unique()
    try : 
        DB.loc[(DB.frame==FRAME+1)&(DB.cell_id.isin(list_cells_t)),['u','diffusion_term','kernel_term','signaling_force','mcherry']]=DB_f[~DB_f.cell_id.isin(disappearing_cells)][['u','diffusion_term','kernel_term','signaling_force','mcherry']].values
    except ValueError :
        print('error frame'+ str(FRAME))
        list_next_time=DB.loc[(DB.frame==FRAME+1)&(DB.cell_id.isin(list_cells_t))].cell_id.unique()
        list_this_time=DB_f[~DB_f.cell_id.isin(disappearing_cells)].cell_id.unique()
        list_difference = [item for item in list_this_time if item not in list_next_time]
        DB_f=DB_f[~DB_f.cell_id.isin(list_difference)]
        DB.loc[(DB.frame==FRAME+1)&(DB.cell_id.isin(list_cells_t)),['u','diffusion_term','kernel_term','signaling_force','mcherry']]=DB_f[~DB_f.cell_id.isin(disappearing_cells)][['u','diffusion_term','kernel_term','signaling_force','mcherry']].values
    #this part treats cells appearing by division
    for CELL in disappearing_cells_division : 
            daughter1,daughter2=DB_f[DB_f.cell_id==CELL].left_daughter_cell_id.unique()[0],DB_f[DB_f.cell_id==CELL].right_daughter_cell_id.unique()[0]
            DB.loc[(DB.frame==FRAME+1)&(DB.cell_id==daughter1),['u','diffusion_term','kernel_term','signaling_force','mcherry']]=DB_f[DB_f.cell_id==CELL][['u','diffusion_term','kernel_term','signaling_force','mcherry']].values[0]
            DB.loc[(DB.frame==FRAME+1)&(DB.cell_id==daughter2),['u','diffusion_term','kernel_term','signaling_force','mcherry']]=DB_f[DB_f.cell_id==CELL][['u','diffusion_term','kernel_term','signaling_force','mcherry']].values[0]
    
    DB_f1=DB.loc[DB.frame==FRAME+1] #subset of the database for this frame
    Neighbors_f=Neighbors.loc[Neighbors.frame==FRAME+1]
    DB_f1['neighbors']=DB_f1['cell_id'].transform(lambda x: Neighbors_f[(Neighbors_f.cell_id==x)].neighbor_cell_id.unique())
    DB_f1['NeighborsCount']=DB_f1['neighbors'].apply(lambda x: len(x))
    #This next part selects the cells at the border of the tissue (within 3 cell layers) to set the boundary condition:
    Differences=[]
    frame_cellshapes=tml.subset_dataframe(movie.cellshapes, ['frame'], [FRAME+1])    
    for cell in DB_f1.cell_id.unique():
        Differences.append((-DB_f1[DB_f1.cell_id==cell].NeighborsCount.values+frame_cellshapes[frame_cellshapes.cell_id==cell].bond_order.max())[0])
    DB_f1['Border']=Differences
    Border_cells=DB_f1[DB_f1.Border!=0].cell_id.unique()
    Border_And_Neighbors=list(Neighbors_f[Neighbors_f.cell_id.isin(Border_cells)].neighbor_cell_id.unique())
    Border_And_Neighborsx2=list(Neighbors_f[Neighbors_f.cell_id.isin(Border_And_Neighbors)].neighbor_cell_id.unique())
    
    #update the value of M to calculate experimental value of u for border cells
    data=DB[DB.frame==FRAME].DSRF_Conc
    y,x,_=plt.hist(data,bins=50); 
    x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
    params, cov = curve_fit(bimodal, x, y, expected)
    M=params[2]
    DB.loc[(DB.frame==FRAME+1)&(DB.cell_id.isin(Border_And_Neighborsx2)),'u']=(1-(DB[(DB.frame==FRAME+1)&(DB.cell_id.isin(Border_And_Neighborsx2))].DSRF_Conc-m)/(M-m))
    DB_f1.loc[(DB.cell_id.isin(Border_And_Neighborsx2)),'u']=(1-(DB_f1.loc[(DB.cell_id.isin(Border_And_Neighborsx2))].DSRF_Conc-m)/(M-m))

#save the database
DB.to_csv('./'+Boundary+'/'+param+'/DB.csv')

