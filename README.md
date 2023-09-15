# diagonal_cross_sections
Script to plot diagonal cross-sections from WRF data.


This script features a number of functions that, given a pair of x1,y1 and x2,y2 grid points, will calculate a cross-section between those two gridpooints. These functions are:

## xsect_km(x1,y1,x2,y2,dx,dy)
Given input points `x1`, `y1` and `x2`, `y2`, and grid spacing in km `dx` and `dy`, calculate the number of km along a cross-section line. It returns an array of shape `(1, num_points)`, so that you can easily resize it for plotting a vertical cross-section.

## xsect_2d(x1,y1,x2,y2,t,n)
`t` is the 2 dimensional input dataset with indices `[y, x]`.
`n` is the number of gridpoints on either side you would like averaged.
The function outputs `newt`, a 1d array at grid spacing identical to that of `t`.

## xsect_3d(x1,y1,x2,y2,t,n)
`t` is the 3 dimensional input dataset with indices `[z, y, z]`. 
`n` is the number of gridpoints on either side you would like averaged.
The function outputs a 2D array of dimensions `[z, newxy]`. Newxy is at interval spacing identical to that of `x` and `y` (i.e., if `x` is at 1 km spacing, so is `newxy`.)

## xsect_3d_noavg(x1,y1,x2,y2,t)
Same as `xsect_3d`, but if you don't want to calculate an average.



## Sample figure 
Fig. 5 from Adams-Selin, R. D., S. C. van den Heever, and R. H. Johnson, 2013: Impact of graupel parameterization schemes on idealized bow echo simulations. Monthly Weather Review, 141, 1241–1262.):

![image](https://user-images.githubusercontent.com/51211535/234317462-738ba7c4-e7f2-4e26-b93a-dff9ad564615.png)
