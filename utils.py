import argparse
import numpy as np
import networkx as nx
from skimage import filters
import imageio
import json

# copy of np.isin method
def isin(element, test_elements, assume_unique=False, invert=False):
    "..."
    element = np.asarray(element)
    return np.in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)

def get_adjacency_matrix(img,ulim=None,llim=None,epsilon=1e-5):
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
    
    # make both sides low weight
    mask = np.zeros(img.shape)
    mask[:,0]=1
    mask[:,-1]=1
    side_x, side_y = np.where(mask==1)
    side_ind = np.ravel_multi_index([side_x, side_y],img.shape)
    isonsideA = isin(adjMAsub, side_ind)
    adjMW[isonsideA==1]=epsilon #super low weight to force node to be selected as part of path
    

    # limit path to a slab based on llim and ulim
    if llim is not None:
        nogo = np.zeros(img.shape)
        nogo[llim:,2:img.shape[1]-2]=1
        nogo_x, nogo_y = np.where(nogo==1)
        nogo_ind = np.ravel_multi_index([nogo_x, nogo_y],img.shape)
        isnogoA = isin(adjMAsub, nogo_ind)        
        adjMW[isnogoA==1]=2 #high weight to prevent node to be select as part of path.
        
    if ulim is not None:
        nogo = np.zeros(img.shape)
        nogo[:ulim,2:img.shape[1]-2]=1
        nogo_x, nogo_y = np.where(nogo==1)
        nogo_ind = np.ravel_multi_index([nogo_x, nogo_y],img.shape)
        isnogoA = isin(adjMAsub, nogo_ind)        
        adjMW[isnogoA==1]=2 #high weight to prevent node to be select as part of path.
        
    edge_weights = np.array([adjMAsub,adjMBsub,adjMW])
    return edge_weights, adjMAsub,adjMBsub
    
def segment_layer(pimg,dark2bright=True,ulim=None,llim=None):
    
    pimg = pimg.astype(np.float)
    # blur - more smooth in horizontal direction
    pimg = filters.gaussian(pimg, sigma=(0.5,3))
    
    # get gradient image
    gradx,grady = np.gradient(pimg,1)

    if dark2bright:
        pass
    else:
        gradx*=-1
        
    gradx = (gradx-np.min(gradx))/(np.max(gradx)-np.min(gradx))

    # construct edge weights with gradient image
    edge_weights, adjMAsub,adjMBsub = get_adjacency_matrix(gradx,ulim=ulim,llim=llim)
    
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
    keeps = np.logical_and(sp_y>1,sp_y<gradx.shape[1]-1)
    sp_x = sp_x[keeps]
    sp_y = sp_y[keeps]
    sp = np.array(sp).astype(int)[keeps]
    
    return sp_x,sp_y,sp


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image_path',type=str)
    parser.add_argument('output_yaml_path',type=str)
    parser.add_argument('-d','--dark2bright',type=str,default='True',choices={'True','False'})
    parser.add_argument('-u','--ulim',type=int,default=None)
    parser.add_argument('-l','--llim',type=int,default=None)
    args = parser.parse_args()

    input_image_path = args.input_image_path
    output_yaml_path = args.output_yaml_path
    dark2bright = eval(args.dark2bright)
    ulim = args.ulim
    llim = args.llim

    input_image = imageio.imread(input_image_path)
    input_image = np.array(input_image)

    # grab one channel
    if len(input_image.shape) > 2:
        input_image = input_image[:,:,0]

    indx,indy,inds = segment_layer(input_image,dark2bright=dark2bright,ulim=ulim,llim=llim)

    mydict = {'x':indx.tolist(),'y':indy.tolist()}
    with open(output_yaml_path,'w') as f:
        f.write(json.dumps(mydict, sort_keys=True, indent=4))