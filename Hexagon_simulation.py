#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
#from Hexagon import hexagon
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
#--------------------------------------------------
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
Writer = animation.writers['ffmpeg']
writer=Writer(codec='mjpeg')
#parameters (time scales are expressed in frames (1 frame=5 min=1/12 hours) and length scales in microns)
kl=2; tau=1.5; D=2.5/tau; u_I=0.52; alpha_I=0.075; r=0.1 ;rho=0.35
D_tau=D*tau
#--------------------------------------
# print(sys.argv)
# r,rho,tau,D,kl,tresh=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])
#------------------------------------
RF=2
Boundary='_'
for condition in ['aitch']:

        print("r,rho=",r,rho)
        dx=4 #distance between cells (averaged in experimental movie)
        a=dx/np.sqrt(3) # hexagon size
        l0=kl*dx;
        M=55;N=55;
        dt=1/10.*3*a**2/(5*D)
        T=10000
        Grid = [[hexagon(m,n) for n in range(-(m-1)//2,N-m//2)] for m in range(M)]
        x_size=a*M*3./2
        y_size=a*N*np.sqrt(3)
        u=np.zeros((int(T/dt),M,N))
        s=np.zeros((int(T/dt),M,N))
        parameters = 'S7snaps_tau='+str(tau)+'_Dtau='+str(D_tau)+'_' + condition + Boundary + str(M)+'_'+str(N)+'/T=' + str(T)+'/l0=' + str(l0)+ '/u_I='+str(u_I) + '_/r=' +str(r) +'/rho=' + str(rho)+'tau=' + str(tau)  
        #parameters = 'repet_'+ str(repet) 
        if not os.path.exists('./'+parameters):
            os.makedirs('./'+parameters)
        #compute distances for the Notch-inhibition term
        if 'distances_a='+str(a)+'_'+str(M)+'_'+str(N)+'.npy'  in os.listdir('./'):
            d=np.load('distances_a='+str(a)+'_'+str(M)+'_'+str(N)+'.npy')
        else :
            d=np.zeros((M,N,M,N))
            # d=np.zeros((M*N,M*N))
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
        def F(u) :
            return 1./4*u**2*(u-1)**2+(r-1./2)*(u**2/2-u**3/3-1./12)
        def f(x) :
            return (r - x)*(-1 + x)*x

        DF=F(1)-F(0)
        print ('Delta F = ' , DF)
        p=[]

        if condition=='stripe' :
            for n in range(N):
                for m in range(2*M//5,3*M//5):
                    u[0,m,n]=1
                    
        if condition=='horizontal_stripe' :
            for n in range(N):
                for m in range(2*M//5,3*M//5):
                    u[0,n,m]=1                   
            for m in range(2*M//5):
                r1=int(2*np.random.rand(1))
                r2=int(2*np.random.rand(1))
                for n in range(18,32):
                    u[0,m,n]=1
                for n in range(15,18):
                    u[0,m,n]=r1
                for n in range(32,35):
                    u[0,m,n]=r2        
                 
        if condition=='aitch' :
            for n in range(N):
                for m in range(M//6,2*M//6):
                    u[0,m,n]=1                   
                for m in range(4*M//6,5*M//6):
                    u[0,m,n]=1
            for m in range(M//6,5*M//6):
                for n in range(3*N//6,4*N//6):
                    u[0,m,n]=1
                    
        if condition=='double_aitch':
            for n in range(N):
                for m in range(8,18):
                    u[0,m,n]=1                   
                for m in range(32,42):
                    u[0,m,n]=1
            for m in range(18,32):
                r1=int(2*np.random.rand(1))
                r2=int(2*np.random.rand(1))
                r3=int(2*np.random.rand(1))
                r4=int(2*np.random.rand(1))
                for n in range(14,24):
                    u[0,m,n]=1    
                for n in range(11,14):
                    u[0,m,n]=r1
                for n in range(24,27):
                    u[0,m,n]=r2                    
                for n in range(31,41):
                    u[0,m,n]=1      
                for n in range(28,31):
                    u[0,m,n]=r1
                for n in range(41,44):
                    u[0,m,n]=r2                                                                    
                    
        if condition=='hor_rough_aitch' :
            for n in range(N):
                for m in range(1*M//6-2,2*M//6-2):
                    u[0,n,m]=1                  
                for m in range(4*M//6+2,5*M//6+2):
                    u[0,n,m]=1
            for m in range(2*M//6-2,4*M//6+2):
                r1=int(2*np.random.rand(1))
                r2=int(2*np.random.rand(1))
                for n in range(3*N//6,4*N//6):
                    u[0,n,m]=1
                for n in range(3*N//6-5,3*N//6):
                    u[0,n,m]=r1
                for n in range(4*N//6,4*N//6+5):
                    u[0,n,m]=r2

        if condition=='rough_aitch' :
            for n in range(N):
                for m in range(1*M//6-2,2*M//6-2):
                    u[0,m,n]=1                  
                for m in range(4*M//6+2,5*M//6+2):
                    u[0,m,n]=1
            for m in range(2*M//6-2,4*M//6+2):
                r1=int(2*np.random.rand(1))
                r2=int(2*np.random.rand(1))
                for n in range(3*N//6,4*N//6):
                    u[0,m,n]=1
                for n in range(3*N//6-5,3*N//6):
                    u[0,m,n]=r1
                for n in range(4*N//6,4*N//6+5):
                    u[0,m,n]=r2

        if condition=='rough_stripe' :
            print (condition)
            for n in np.arange(N):
                r1=int(2*np.random.rand(1))
                r2=int(2*np.random.rand(1))
                for m in range(M):
                    u[0,m,n]=0
                for m in range(M//2-2+1,M//2+2+1):
                    u[0,m,n]=1
                for m in range(M//2-2-RF+1,M//2-2+1):
                    u[0,m,n]=r1
                for m in range(M//2+2+1,M//2+2+RF+1):
                    u[0,m,n]=r2
                    
        if condition=='hor_rough_stripe' :
            print (condition)
            for n in np.arange(N):
                r1=int(2*np.random.rand(1))
                r2=int(2*np.random.rand(1))
                for m in range(M):
                    u[0,n,m]=0
                for m in range(M//2-2+1,M//2+2+1):
                    u[0,n,m]=1
                for m in range(M//2-2-RF+1,M//2-2+1):
                    u[0,n,m]=r1
                for m in range(M//2+2+1,M//2+2+RF+1):
                    u[0,n,m]=r2
                    
        if condition=='rough_large_stripe':
            for n in range(N):
                r1=int(2*np.random.rand(1))
                r2=int(2*np.random.rand(1))
                for m in range(M):
                    u[0,m,n]=0
                for m in range(17,37):
                    u[0,m,n]=1
                for m in range(12,17):
                    u[0,m,n]=r1
                for m in range(37,42):
                    u[0,m,n]=r2
                    
        if condition=='hor_rough_large_stripe':
            for n in range(N):
                r1=int(2*np.random.rand(1))
                r2=int(2*np.random.rand(1))
                for m in range(M):
                    u[0,n,m]=0
                for m in range(17,37):
                    u[0,n,m]=1
                for m in range(12,17):
                    u[0,n,m]=r1
                for m in range(37,42):
                    u[0,n,m]=r2
    
        #find neighboring cells
        neighbors=np.zeros((M,N,6,2))
        for k in range(M) :
            for l in range(N) :
                if k%2==1 :
                    neighbors[k,l]=(np.array([k,l])+np.array([[0,1],[1,0],[-1,1],[0,-1],[-1,0],[1,1]]))%[M,N]
                else :
                    neighbors[k,l]=(np.array([k,l])+np.array([[0,1],[1,0],[-1,-1],[0,-1],[-1,0],[1,-1]]))%[M,N]
        #start the simulation           
        for i in tqdm(range(int(T/dt)-1))  :
            if i%250==0 :
                #--------------------------------------------------------------
                ##plot
                plt.figure()
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
                        pgon=patches.Polygon(pgon,edgecolor='white',fill=True,facecolor=cmap((u[i,k,l]-minValue)*2/(maxValue - minValue)-1))
                        polygons.append(pgon)
                fig,ax=plt.subplots()
                ax.add_collection(PatchCollection(polygons,match_original=True))
                ax.axis('equal')
                plt.axis('off')
                #plt.title('u_final_t')
                plt.savefig('./'+parameters + '/u_'+str(i)+'.png',format='png',bbox_inches='tight', pad_inches = 0,dpi=150)
                #plt.savefig(parameters + '/u_final.png',format='png')
                plt.close(fig)
                #--------------------------------------------------------------
            for l in range(N) :
                for k in range(M) :
                    #Notch inhibition term
                    s[i,k,l]=(np.sum(np.exp(-d[k,l,:,:]/l0)*u[i,:,:]))/(np.sum(np.exp(-d[k,l,:,:]/l0)))*(1-math.tanh((u[i,k,l]-u_I)/alpha_I))/2 
                    #diffusion term
                    diff=0;
                    for ind in neighbors[k,l,:,:] :
                        diff+=u[i,int(ind[0]),int(ind[1])]    
                    u[i+1,k,l]=u[i,k,l]+dt/tau*(D*tau/dx**2*(diff-6*u[i,k,l])+rho*(-s[i,k,l])+f(u[i,k,l]))+np.random.normal(0,np.sqrt(0.0026*dt))

        i=int(T/dt-1) ;
        polygons = [];
        for k in range(M):
            for l in range(N):
                cell=Grid[k][l];
                x,y=cell.get_center_m()*a*np.sqrt(3)*np.cos(np.pi/6),cell.get_center_n()*a*np.sqrt(3)+cell.get_center_m()*a*np.sqrt(3)*np.sin(np.pi/6)
                pgon=np.array([[-a,0],[-a/2.,-np.sqrt(3)*a/2],[a/2.,-np.sqrt(3)*a/2],[a,0],[a/2.,np.sqrt(3)*a/2],[-a/2.,np.sqrt(3)*a/2]])
                pgon+=[x,y]
                pgon=patches.Polygon(pgon,edgecolor='white',fill=True,facecolor=cmap((u[i,k,l]-minValue)*2/(maxValue - minValue)-1))
                polygons.append(pgon)
        fig,ax=plt.subplots()
        ax.add_collection(PatchCollection(polygons,match_original=True))
        ax.axis('equal')
        plt.axis('off')
        #plt.title('u_final_t')
        plt.savefig(parameters + '/u_final.png',format='png',bbox_inches='tight', pad_inches = 0, dpi=150)
        plt.close(fig)
        



