#Example plotting script to generate a vertical cross-section of graupel, snow, and rain mixing ratios between two x,y points. 
#Use WRF output, but can be easily modified to the gridded output of your choice.

import matplotlib.pyplot as plt
import numpy as np
import math
import netCDF4 as nc4

def xsect_km(x1,y1,x2,y2,dx,dy):
    #determine km along xsect line
     m = (y2-y1) / (x2-x1)
     totalkm = (((x2-x1)*dx)**2. + ((y2-y1)*dy)**2.)**0.5
     totalz = ((x2-x1)**2. + (y2-y1)**2.)**0.5
     num_gpts =  math.floor(totalz)
     km = (np.arange(num_gpts)+1)*(totalkm / num_gpts)
     km = np.concatenate((np.zeros(1),km))
     return km;

def xsect_2d(x1,y1,x2,y2,t,n):
     #x1,y1,x2,y2 endpoints; t dataset; n #gridpoints either side you want averaged
     m = np.float(y2-y1) / (x2-x1)
     totalz = ((x2-x1)**2. + (y2-y1)**2.)**0.5
     num_gpts =  np.floor(totalz)
     z = (np.arange(num_gpts)+1)*(totalz / num_gpts)
     z = np.concatenate((np.zeros(1),z))
     xn = z/((1+m**2.)**0.5) + x1
     yn = z/(((1/m**2.)+1)**0.5)  + y1
     if (y2-y1 < 0):
        yn = z/(((1/m**2.)+1)**0.5)  + y2
        yn = yn[::-1]
     #for each gridpoint, find n gridpoints on either side and average
     alpha = np.arctan(m)  #angle of slope from horizontal
     newt = np.zeros(z.shape[0])
     for i in range(xn.shape[0]):
           xi = xn[i]
           yi = yn[i]
            #array for point xi,yi plus n pts surrounding
           xipn = np.zeros((2*n+1)) 
           yipn = np.zeros((2*n+1))
           xipn[n] = xi  
           yipn[n] = yi
           for j in np.arange(n)+1:
                 npt = j
                 a = npt * np.sin(alpha)
                 b = npt * np.cos(alpha)
                 xipn[npt+n] = xipn[n] + a
                 yipn[npt+n] = yipn[n] - b
                 xipn[n-npt] =  xipn[n] - a
                 yipn[n-npt] = yipn[n] + b
           xipna = np.array(np.floor(xipn),dtype=int)
           yipna = np.array(np.floor(yipn),dtype=int)
           xipnb = xipna + 1
           yipnb = yipna + 1
           tempt = t[yipna,xipna]*( (xipnb-xipn)*(yipnb-yipn) )/( (xipnb-xipna)*(yipnb-yipna) ) + \
                         t[yipna,xipnb]*( (xipn-xipna)*(yipnb-yipn) )/( (xipnb-xipna)*(yipnb-yipna) ) + \
                         t[yipnb,xipna]*( (xipnb-xipn)*(yipn-yipna) )/( (xipnb-xipna)*(yipnb-yipna) ) + \
                         t[yipnb,xipnb]*( (xipn-xipna)*(yipn-yipna) )/( (xipnb-xipna)*(yipnb-yipna) )
           newt[i] = tempt.mean()
     return newt;

def xsect_3d(x1,y1,x2,y2,t,n):
     #x1,y1,x2,y2 endpoints; t dataset; n #gridpoints either side you want averaged
     m = np.float(y2-y1) / (x2-x1)
     totalz = ((x2-x1)**2. + (y2-y1)**2.)**0.5
     num_gpts =  np.floor(totalz)
     z = (np.arange(num_gpts)+1)*(totalz / num_gpts)
     z = np.concatenate((np.zeros(1),z))
     xn = z/((1+m**2.)**0.5) + x1
     yn = z/(((1/m**2.)+1)**0.5)  + y1
     if (y2-y1 < 0):
        yn = z/(((1/m**2.)+1)**0.5)  + y2
        yn = yn[::-1]
     #for each gridpoint, find n gridpoints on either side and average
     alpha = np.arctan(m)  #angle of slope from horizontal
     newt = np.zeros((t.shape[0],z.shape[0]))
     for i in range(xn.shape[0]):
           xi = xn[i]
           yi = yn[i]
           #array for point xi,yi plus n pts surrounding
           xipn = np.zeros((2*n+1),dtype='f') 
           yipn = np.zeros((2*n+1),dtype='f')
           xipn[n] = xi  
           yipn[n] = yi
           for j in np.arange(n)+1:
                 npt = j
                 a = npt * np.sin(alpha)
                 b = npt * np.cos(alpha)
                 xipn[npt+n] = xipn[n] + a
                 yipn[npt+n] = yipn[n] - b
                 xipn[n-npt] =  xipn[n] - a
                 yipn[n-npt] = yipn[n] + b
           xipna = np.array(np.floor(xipn),dtype=int)
           yipna = np.array(np.floor(yipn),dtype=int)
           xipnb = xipna + 1at 
           yipnb = yipna + 1
           tempt = t[:,yipna,xipna]*( (xipnb-xipn)*(yipnb-yipn) )/( (xipnb-xipna)*(yipnb-yipna) ) + \
                         t[:,yipna,xipnb]*( (xipn-xipna)*(yipnb-yipn) )/( (xipnb-xipna)*(yipnb-yipna) ) + \
                         t[:,yipnb,xipna]*( (xipnb-xipn)*(yipn-yipna) )/( (xipnb-xipna)*(yipnb-yipna) ) + \
                         t[:,yipnb,xipnb]*( (xipn-xipna)*(yipn-yipna) )/( (xipnb-xipna)*(yipnb-yipna) )
           newt[:,i] = tempt.mean(axis=1)
     return newt;

def xsect_3d_noavg(x1,y1,x2,y2,t):
     m = np.float(y2-y1) / (x2-x1)
     totalz = ((x2-x1)**2. + (y2-y1)**2.)**0.5
     num_gpts =  np.floor(totalz)
     z = (np.arange(num_gpts)+1)*(totalz / num_gpts)
     z = np.concatenate((np.zeros(1),z))
     xn = z/((1+m**2.)**0.5) + x1
     yn = z/(((1/m**2.)+1)**0.5)  + y1
     if (y2-y1 < 0):
        yn = z/(((1/m**2.)+1)**0.5)  + y2
        yn = yn[::-1]
     #set up the components for bilinear interpolation
     xa = np.array(np.floor(xn),dtype='int')
     xb = xa+1
     ya = np.array(np.floor(yn),dtype='int')
     yb = ya+1
     newt = t[:,ya,xa]*( (xb-xn)*(yb-yn) )/( (xb-xa)*(yb-ya) ) + \
               t[:,ya,xb]*( (xn-xa)*(yb-yn) )/( (xb-xa)*(yb-ya) ) + \
               t[:,yb,xa]*( (xb-xn)*(yn-ya) )/( (xb-xa)*(yb-ya) ) + \
               t[:,yb,xb]*( (xn-xa)*(yn-ya) )/( (xb-xa)*(yb-ya) )
     return newt;


#define a bunch of color names
white                =( 1.00 , 1.00  , 1.00 )
light_purple         =( 0.87 , 0.69  , 1.00 )
light_violet         =( 0.38 , 0.25  , 0.62 )
violet               =( 0.50 , 0.00  , 0.50 )
dark_violet          =( 0.25 , 0.00  , 0.25 )
blue                 =( 0.00 , 0.00  , 1.00 )
dark_blue            =( 0.00 , 0.00  , 0.50 )
light_blue           =( 0.50 , 0.50  , 1.00 )
dark_green           =( 0.00 , 0.50  , 0.00 )
green                =( 0.00 , 1.00  , 0.00 )
light_green          =( 0.50 , 1.00  , 0.50 )
yellow               =( 1.00 , 1.00  , 0.00 )
orange               =( 1.00 , 0.50  , 0.00 )
tan                  =( 0.80 , 0.40  , 0.00 )


#read in data from a WRF file
filename = 'wrfout_d01_2003-03-13_04:00:00'
namestring = filename[-19:-3] 
nc = nc4.Dataset(filename,'r')
qc = np.array(nc.variables['QCLOUD'][0,:,:,:])
qr = np.array(nc.variables['QRAIN'][0,:,:,:])
qi = np.array(nc.variables['QICE'][0,:,:,:])
qs = np.array(nc.variables['QSNOW'][0,:,:,:])
qg = np.array(nc.variables['QGRAUP'][0,:,:,:])
dx = nc.attributes['DX'][0] / 1000.
dy = nc.attributes['DY'][0] / 1000.
#calculate new variables
ph = np.array(nc.variables['PH'][0,:,:,:])
phb = np.array(nc.variables['PHB'][0,:,:,:])
nz = ph.shape[0]
height = ( ( (ph[:nz-1,:,:]+ph[1:nz,:,:])/2.) + ( (phb[:nz-1,:,:]+phb[1:nz,:,:])/2.)) / 9.81
theta = np.array(nc.variables['T'][0,:,:,:])+300.0
pres = (np.array(nc.variables['P'][0,:,:,:]) + np.array(nc.variables['PB'][0,:,:,:]))/100.0
temp = theta * (pres / 1000.0)**0.2857
rho = (pres * 100.0) / (287. * temp)
min_sstart = np.array(nc.variables['XTIME'])
nc.close()


#set your x,y start and endpoints for the cross-section
x1=341
y1=287
x2=388
y2=261



#calculate the cross-sections
n=5 #average 5 grid-points on either side of the cross-section
xkm = xsect_km(x1,y1,x2,y2,dx,dy)
xkm = np.resize(xkm,(34,xkm.shape[0]))  #34 vertical levels in the WRF run
xhgt = xsect_3d(x1,y1,x2,y2,height,n) / 1000.0
xqg = xsect_3d(x1,y1,x2,y2,qg,n)
xrn = xsect_3d(x1,y1,x2,y2,qr,n)
xsn = xsect_3d(x1,y1,x2,y2,qs,n)
xcl = xsect_3d(x1,y1,x2,y2,qc+qi,n)
xt = xsect_3d(x1,y1,x2,y2,temp,n)
xrho = xsect_3d(x1,y1,x2,y2,rho,n)
xth = xsect_3d(x1,y1,x2,y2,theta,n)



#make the plots
fig = plt.figure(num=None, figsize=(8,6))
fig.subplots_adjust(hspace=0.03,left=0.07,right=0.97,top=0.97,bottom=0.03)
ax = plt.subplot(111)
ax.axis([0,xkm[0,xkm.shape[1]-1],0,15])  #vertical cross section extends up to 15 km in height
cqg = ax.contourf(xkm, xhgt, xqg*1000., [0.01,0.5,1.,2.,2.5,3.,3.5,4.,4.5,5.,6.],  #customized graupel contour fill levels, see readme
                 colors=(white,light_violet,violet,blue,light_blue,dark_green,green,
                             light_green,yellow,tan,orange))
PB = plt.colorbar(cqg)
cqt = ax.contour(xkm, xhgt, xt, [273.15], colors='k', linewidth=4, linestyles='dashed')  #melting level
cth = ax.contour(xkm, xhgt, xth, [290.,292.,294.],colors='k',linewidths=2)  #cold pool contours
cqr = ax.contour(xkm,xhgt,xrn*1000.,np.arange(0.5,10,0.5),colors='g',linewidths=1.5)  
ax.clabel(cqr,fontsize=6,fmt='%2.1f')
cqs = ax.contour(xkm,xhgt,xsn*1000.,np.arange(0.1,10,0.5),colors='k',linewidths=1.5)
ax.clabel(cqs,fontsize=6,fmt='%2.1f')
cqc = ax.contour(xkm, xhgt, xcl*1000., np.arange(0.5,10,0.5),colors='r',linewidths=1.5)
ax.clabel(cqc,fontsize=6,fmt='%2.1f')
ax.axis((0,xkm[0,xkm.shape[1]-1],0,15))
print 'saving figure: xsect_qgsr_'+namestring+'.png'
plt.savefig('xsect_qgsr_'+namestring+'.png', format='png')
plt.close()   
