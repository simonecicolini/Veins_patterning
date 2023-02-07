import os
import time
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
import math
#------------------------------------------------------
class hexagon:
    def __init__(self, center_m, center_n,size=1,
                 color=(1.0,1.0,1.0,1.0),u=0,
                 line_thickness=1):

        self.__center_m = center_m
        self.__center_n = center_n
        self.__size = size
        self.__u = u
        self.__color = color
        self.__line_thickness = line_thickness
        
    def set_center_m(self, new_center):
        self.__center_m = new_center
    
    def get_center_m(self):
        return self.__center_m
    
    def set_center_n(self, new_center):
        self.__center_n = new_center
    
    def get_center_n(self):
        return self.__center_n

    def set_size(self, new_size):
        self.__size = new_size

    def get_size(self):
        return self.__size

    def set_u(self, u):
        self.__u = u

    def get_u(self):
        return self.__u

    def set_color(self, color):
        self.__color = color

    def get_color(self):
        return self.__color
#----------------------------------------------------------
#parameters (time scales are expressed in frames (1 frame=5 min=1/12 hours) and length scales in microns)
kl=2; tau=1.5; D=2.5/tau; u_I=0.52; alpha_I=0.075; r=0.1 ;rho=0.4
D_tau=D*tau
RF=2
#define initial condition for the simulation
condition='rough_stripe'

dx=4 #distance between cells (averaged in experimental movie)
a=dx/np.sqrt(3) # hexagon size
l0=kl*dx;
M=55;N=55; #lattice size
dt=1/10.*3*a**2/(5*D)
T=2000
Grid = [[hexagon(m,n) for n in range(-(m-1)//2,N-m//2)] for m in range(M)]
x_size=a*M*3./2
y_size=a*N*np.sqrt(3)
u=np.zeros((M,N))
s=np.zeros((M,N))
parameters = 'hexagons_'+condition+'_tau='+str(tau)+'_Dtau='+str(D_tau)+'_' + str(M)+'_'+str(N)+'/T=' + str(T)+'/l0=' + str(l0)+ '/u_I='+str(u_I) + '_/r=' +str(r) +'/rho=' + str(rho)+'tau=' + str(tau)  
if not os.path.exists('./'+parameters):
    os.makedirs('./'+parameters)
#compute distances for the Notch-inhibition term
if 'distances_a='+str(a)+'_'+str(M)+'_'+str(N)+'.npy'  in os.listdir('./'):
    d=np.load('distances_a='+str(a)+'_'+str(M)+'_'+str(N)+'.npy')
else :
    d=np.zeros((M,N,M,N))
    for k in range(M) :
        for l in range(N) :
            cell=Grid[k][l]
            x,y=cell.get_center_m()*a*np.sqrt(3)*np.cos(np.pi/6),cell.get_center_n()*a*np.sqrt(3)+cell.get_center_m()*a*np.sqrt(3)*np.sin(np.pi/6)
            for i in range(M) :
                for j in range(N) :
                    cell2=Grid[i][j]
                    x1,y1=cell2.get_center_m()*a*np.sqrt(3)*np.cos(np.pi/6),cell2.get_center_n()*a*np.sqrt(3)+cell2.get_center_m()*a*np.sqrt(3)*np.sin(np.pi/6)
                    Dx = x1-x; dy=y1-y;
                    if (Dx >   x_size*0.5):
                        Dx = Dx - x_size
                    if (Dx <= -x_size * 0.5):
                        Dx = Dx + x_size
                    if (dy >   y_size*0.5):
                        dy = dy - y_size
                    if (dy <= -y_size * 0.5):
                        dy = dy + y_size
                    d[k,l,i,j]=np.sqrt(Dx**2+dy**2)
    np.save('distances_a='+str(a)+'_'+str(M)+'_'+str(N)+'.npy',d)
    
#autonomous dynamics    
def f(x) :
    return (r - x)*(-1 + x)*x

if condition=='stripe' :
    for n in range(N):
        for m in range(2*M//5,3*M//5):
            u[m,n]=1
                             
if condition=='aitch' :
    for n in range(N):
        for m in range(M//6,2*M//6):
            u[m,n]=1                   
        for m in range(4*M//6,5*M//6):
            u[m,n]=1
    for m in range(M//6,5*M//6):
        for n in range(3*N//6,4*N//6):
            u[m,n]=1
            
if condition=='rough_aitch' :
    for n in range(N):
        for m in range(1*M//6-2,2*M//6-2):
            u[m,n]=1                  
        for m in range(4*M//6+2,5*M//6+2):
            u[m,n]=1
    for m in range(2*M//6-2,4*M//6+2):
        r1=int(2*np.random.rand(1))
        r2=int(2*np.random.rand(1))
        for n in range(3*N//6,4*N//6):
            u[m,n]=1
        for n in range(3*N//6-5,3*N//6):
            u[m,n]=r1
        for n in range(4*N//6,4*N//6+5):
            u[m,n]=r2

if condition=='rough_stripe' :
    print (condition)
    for n in np.arange(N):
        r1=int(2*np.random.rand(1))
        r2=int(2*np.random.rand(1))
        for m in range(M):
            u[m,n]=0
        for m in range(M//2-2+1,M//2+2+1):
            u[m,n]=1
        for m in range(M//2-2-RF+1,M//2-2+1):
            u[m,n]=r1
        for m in range(M//2+2+1,M//2+2+RF+1):
            u[m,n]=r2
                                 
if condition=='rough_large_stripe':
    for n in range(N):
        r1=int(2*np.random.rand(1))
        r2=int(2*np.random.rand(1))
        for m in range(M):
            u[m,n]=0
        for m in range(17,37):
            u[m,n]=1
        for m in range(12,17):
            u[m,n]=r1
        for m in range(37,42):
            u[m,n]=r2
                
#find neighboring cells 
neighbors=np.zeros((M,N,6,2))
for k in range(M) :
    for l in range(N) :
        if k%2==1 :
            neighbors[k,l]=(np.array([k,l])+np.array([[0,1],[1,0],[-1,1],[0,-1],[-1,0],[1,1]]))%[M,N]
        else :
            neighbors[k,l]=(np.array([k,l])+np.array([[0,1],[1,0],[-1,-1],[0,-1],[-1,0],[1,-1]]))%[M,N]
#start the simulation           
for i in tqdm(range(int(T/dt)-1)) :
    #plot hexagons colored according to the value of u in each hexagons
    if i%100==0 :
        #--------------------------------------------------------------
        minValue=-0.1;
        maxValue=1.1;
        cmap = mpl.cm.get_cmap('plasma')
        fig,ax=plt.subplots()
        ax1 = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        norm = mpl.colors.Normalize(vmin=np.min(u), vmax=np.max(u))
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                        norm=norm)
        polygons = [];
        for k in range(M):
            for l in range(N):
                cell=Grid[k][l];
                x,y=cell.get_center_m()*a*np.sqrt(3)*np.cos(np.pi/6),cell.get_center_n()*a*np.sqrt(3)+cell.get_center_m()*a*np.sqrt(3)*np.sin(np.pi/6)
                pgon=np.array([[-a,0],[-a/2.,-np.sqrt(3)*a/2],[a/2.,-np.sqrt(3)*a/2],[a,0],[a/2.,np.sqrt(3)*a/2],[-a/2.,np.sqrt(3)*a/2]])
                pgon+=[x,y]
                pgon=patches.Polygon(pgon,edgecolor='white',fill=True,facecolor=cmap((u[k,l]-minValue)*2/(maxValue - minValue)-1))
                polygons.append(pgon)
        fig,ax=plt.subplots()
        ax.add_collection(PatchCollection(polygons,match_original=True))
        ax.axis('equal')
        plt.axis('off')
        plt.savefig('./'+parameters + '/u_'+str(i)+'.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=150)
        plt.close(fig)
        #--------------------------------------------------------------
    #loop through each hexagonal cell    
    for l in range(N) :
        for k in range(M) :
            #calculate Notch inhibition term
            s[k,l]=(np.sum(np.exp(-d[k,l,:,:]/l0)*u[:,:]))/(np.sum(np.exp(-d[k,l,:,:]/l0)))*(1-math.tanh((u[k,l]-u_I)/alpha_I))/2 
            #calculate nearest neighbor activation term
            diff=0;
            for ind in neighbors[k,l,:,:] :
                diff+=u[int(ind[0]),int(ind[1])]  
            #update the value of u in the hexagon labelled by k,l    
            u[k,l]=u[k,l]+dt/tau*(D*tau/dx**2*(diff-6*u[k,l])+rho*(-s[k,l])+f(u[k,l]))+np.random.normal(0,np.sqrt(0.0026*dt))
        
