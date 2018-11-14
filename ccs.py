import os
import math
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl    
    """    
    if not isinstance(hull,Delaunay):
        raise ValueError('hull is not of type `spatial.Delaunay`!')

    return hull.find_simplex(p)>=0

# https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y, center):
    x = x - center[0]
    y = y - center[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi,center):
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    x += center[0]
    y += center[1]
    return(x, y)
    
    
    
# copy of np.isin method
def isin(element, test_elements, assume_unique=False, invert=False):
    "..."
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)

def get_adjacency_matrix(img,max_radius_lim=None,epsilon=1e-5):
    # construct adjacency matrix later used to create a sparse graph per Chiu et al.
    # https://www.mathworks.com/matlabcentral/fileexchange/43518-graph-based-segmentation-of-retinal-layers-in-oct-images
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3342188/

    adjMAsub = np.array(np.arange(0,len(img.ravel())))
    adjMAx,adjMAy = np.unravel_index(adjMAsub,img.shape)

    adjMAsub = np.expand_dims(adjMAsub,axis=-1)
    adjMAx = np.expand_dims(adjMAx,axis=-1)
    adjMAy = np.expand_dims(adjMAy,axis=-1)
    
    neighborIterX = np.array([[1, 1, 1, 0, 0, -1, -1 ,-1]])
    neighborIterY = np.array([[1, 0, -1, 1, -1,  1,  0, -1]])
    neighborIterX = np.repeat(neighborIterX, adjMAsub.shape[0], axis=0)
    neighborIterY = np.repeat(neighborIterY, adjMAsub.shape[0], axis=0)
    
    adjMAsub = np.repeat(adjMAsub, 8, axis=1)
    adjMAx = np.repeat(adjMAx, 8, axis=1)
    adjMAy = np.repeat(adjMAy, 8, axis=1)
    
    adjMBx = adjMAx + neighborIterX
    adjMBy = adjMAy + neighborIterY
    
    adjMAsub = adjMAsub.ravel()
    adjMAx = adjMAx.ravel()
    adjMAy = adjMAy.ravel()
    
    adjMBx = adjMBx.ravel()
    adjMBy = adjMBy.ravel()

    criteria = np.array([
        [adjMBx > 0],
        [adjMBx < img.shape[0]],
        [adjMBy > 0],
        [adjMBy < img.shape[1]],
    ])
    
    keepInd = np.all(criteria,axis=0).squeeze()
        
    adjMAsub = adjMAsub[keepInd==1]
    adjMAx = adjMAx[keepInd==1]
    adjMAy = adjMAy[keepInd==1]
    
    adjMBx = adjMBx[keepInd==1]
    adjMBy = adjMBy[keepInd==1]
    adjMBsub = np.ravel_multi_index([adjMBx,adjMBy], img.shape) 
    
    adjMW = 2 - img.ravel()[adjMAsub] - img.ravel()[adjMBsub] + epsilon;
    
    # make side easy
    mask = np.zeros(img.shape)
    mask[0,:]=1
    mask[-1,:]=1
    side_x, side_y = np.where(mask==1)
    side_ind = np.ravel_multi_index([side_x, side_y],img.shape)
    isonsideA = isin(adjMAsub, side_ind)
    #isonsideB = isin(adjMBsub, side_ind) # prolly not needed
    adjMW[isonsideA==1]=epsilon #super low weight to force node to be selected as part of path
    #adjMW[isonsideB==1]=epsilon # prolly not needed
    
    
    # make bottom hard.
    if max_radius_lim is not None:
        nogo = np.zeros(img.shape)
        nogo[2:img.shape[0]-2,max_radius_lim:]=1
        nogo_x, nogo_y = np.where(nogo==1)
        nogo_ind = np.ravel_multi_index([nogo_x, nogo_y],img.shape)
        isnogoA = isin(adjMAsub, nogo_ind)        
        adjMW[isnogoA==1]=2 #high weight to prevent node to be select as part of path.
        
    edge_weights = np.array([adjMAsub,adjMBsub,adjMW])
    return edge_weights, adjMAsub,adjMBsub
    
    
def get_closed_contour(testimg,max_radius_lim=None,phi_length=None):

    # build an image where
    # rho is from 0 to const*estimated_radius
    # phi is from -1 to 1
    rho_length = int((testimg.shape[0]+testimg.shape[1])/2)

    if max_radius_lim is None:
        max_radius_lim = int(rho_length)

    if phi_length is None:
        phi_length = int(2 * np.pi *max_radius_lim/2)
    
    center = (int(testimg.shape[0]/2),int(testimg.shape[1]/2))
    
    # make coord in cartesian
    x = np.array(np.arange(0,testimg.shape[0],1))
    y = np.array(np.arange(0,testimg.shape[0],1))
    XX, YY = np.meshgrid(x, y)
    
    # make coord in polar 
    rho_list = np.arange(0,rho_length,1)
    phi_list = np.linspace(-np.pi,np.pi,num=phi_length,endpoint=True)
    r = np.array(rho_list)
    p = np.array(phi_list)
    RR,PP = np.meshgrid(r,p)
    
    # convert from polar to cartesian
    _xx,_yy=pol2cart(RR.ravel(), PP.ravel(),center)
    subscript = np.array([_xx,_yy]).astype(int)
    _ind = np.ravel_multi_index(subscript, testimg.shape, mode='clip')
    pimg = testimg.ravel()[_ind]
    pimg = np.reshape(pimg,RR.shape)

    # get gradient top to down (y axis)
    epsilon = 1e-5
    #from skimage import filters
    #blurred = filters.gaussian(pimg, sigma=(0.5,0)) # blur in the phi direction.
    _,gradx = np.gradient(pimg,1)
    gradx = (gradx-np.min(gradx))/(np.max(gradx)-np.min(gradx))

    # construct edge weights with gradient image
    edge_weights, adjMAsub,adjMBsub = get_adjacency_matrix(gradx,max_radius_lim)

    # build graph
    FG=nx.Graph()
    FG.add_weighted_edges_from(edge_weights.T)

    # declare start and end points
    start_coord = (0,0)
    end_coord = tuple(np.array(gradx.shape)-1)
    selected_inds = np.ravel_multi_index(np.array([start_coord,end_coord]).T,gradx.shape)
    start_ind,end_ind=tuple(selected_inds)

    # get shortest path between 2 points
    sp = nx.shortest_path(FG,source=start_ind,target=end_ind,weight='weight')
    sp_x, sp_y = np.unravel_index(np.array(sp).astype(int),gradx.shape)

    # trim the path ( dont care about path at both sides of image)
    keeps = np.logical_and(sp_x>1,sp_x<gradx.shape[0]-2)
    sp_x = sp_x[keeps]
    sp_y = sp_y[keeps]
    sp = np.array(sp).astype(int)[keeps]

    # get rho, phi coordinates of shortest path
    sp_rho = RR.ravel()[sp]
    sp_phi = PP.ravel()[sp]
    
    # convert path to cartesian coordinates
    cart_x, cart_y = pol2cart(sp_rho,sp_phi,center)
    cart_x = cart_x.astype(int)
    cart_y = cart_y.astype(int)

    # make mask from closed contour path (shortest path)
    try:
        hull = spatial.Delaunay(np.array([cart_x,cart_y]).T)
        a=np.array([XX.ravel(),YY.ravel()]).T
        b=in_hull(a,hull)
        mask=np.reshape(b,testimg.shape)
    except:
        mask=np.zeros(testimg.shape)
    return cart_x,cart_y,mask


def get_test_image():
    # create circle coord
    radius = 20
    t = np.linspace(0, 2 * np.pi, 20)
    xc, yc = 30,30
    a, b = radius, radius
    x = xc + a * np.cos(t)
    y = yc + b * np.sin(t)
    data = np.column_stack([x, y])
    np.random.seed(seed=1234)
    # get hull
    hull = spatial.Delaunay(data)

    # get image grid
    sz = int(xc/2+radius*2)
    x = np.array(np.arange(0,sz,1))
    y = np.array(np.arange(0,sz,1))
    XX, YY = np.meshgrid(x, y)

    # get image containing hull
    a=np.array([XX.ravel(),YY.ravel()]).T
    b=in_hull(a,hull)
    c=np.reshape(b,(sz,sz))

    # add noise to image
    testimg = -1*( c+np.random.normal(loc=0.0, scale=0.2, size=c.shape))+1
    
    return testimg