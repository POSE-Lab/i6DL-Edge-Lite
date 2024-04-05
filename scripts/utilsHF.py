"""
Utilities
"""
import yaml
from absl import logging
import glob
import sys
sys.path.append("../")
import tqdm
import numpy as np
import cv2 as cv
import json
import time
from plyfile import PlyData
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
from matplotlib import image
from matplotlib import pyplot as plt
from PIL import Image as ImagePIL
import json
import os
from JsonEncoder import NoIndent, NoIndentEncoder

@dataclass
class point3D:
    x: float
    y: float
    z: float
@dataclass
class triangle3D:
    v1: point3D
    v2: point3D
    v3: point3D

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.

    Args:
        directory_path (str): The path of the directory to be created.

    Returns:
        str: Message indicating success or failure.
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    except Exception as e:
        return f"Error creating directory '{directory_path}': {e}"
    
def loadJson(filename):
    with open(filename,'r') as f:
        data = json.load(f)

    return data
def saveEPOSJSON(filename,poses):

    data = {}
    with open(filename,'w') as f:
        for p in tqdm.tqdm(poses):
            R = p['R'].flatten()
            T = p['t'].flatten()
            data[str(p['im_id'])] = NoIndent([{'cam_R_m2c': [R[0], R[1], R[2],
                                                          R[3], R[4], R[5],
                                                          R[6], R[7], R[8]],
                                            'cam_t_m2c':[T[0], T[1], T[2]],  # swap y,z axis by reversing the order of t[i][1]mt[i][2]\
                                          'obj_id':p['obj_id']}])
        f.write(json.dumps(data, cls=NoIndentEncoder, indent=2))

def load_config(filename):
    """Loads a configuration file.

    Args:
        filename (str): Path to configuration file

    Returns:
        dict: Loaded yaml dictionary
    """
    with open(filename,'r') as c: 
        data = yaml.load(c, Loader=yaml.SafeLoader)

    return data

def readImagesFromDir(dir):
    """Loads images from "dir" path to disc and
    returns both the image filenames and the loaded images.

    Args:
        dir (str): Path to images

    Returns:
        tuple: Images list (paths) , Images loaded to disk (arrays)
    """
    images = glob.glob(dir + "/*.png")
    logging.info("Detected %d images" , len(images))

    #limgs = [load_image(_) for _ in tqdm.tqdm(images,desc="Loading...")]
    logging.info("Done loading images   .")

    return images
def check_dataset_Image_size(image):
    return image.shape

def iskMultiInstance(prediction):
    return len(prediction) > 1

def pred2pose(prediction):
    """
    Extracts pose from prediction dictionary
    """
    
    return np.array(prediction['R']),np.array(prediction['t'])
def loadProjectionPoints(file):

    with open(file,'r') as f:
        data = json.load(f)
 
    return np.array([[data['3'][i][0]["x"],data['3'][i][0]["y"],data['3'][i][0]["z"]] for i in data["3"]])

def load3DModel(modelpath):
    data = PlyData.read(modelpath)

    x = np.array(data['vertex']['x'])
    y = np.array(data['vertex']['y'])
    z = np.array(data['vertex']['z'])
    points = np.column_stack((x, y, z))

    return points, loadModelFaces(data)

def boundingBoxPerInstance(rot,tr,obj_points,K,obj_id):
    """Computes the bounding box of each instance present in the image
    by projecting some known 3D points of the model to the image and finding 
    the Lower-Left and Upper-Right points of the boudning box by finding the 
    maximum and minimum of the projected points in image coordinates.

    Args:
        rot (ND ARRAY): Rotation matrix 3 x 3
        tr (ND ARRAY): Translation vector 1 x 3
        obj_points (ND ARRAY): The loaded known 3D points of the model
        K (ND ARRAY): The calibration matrix
        obj_id (int): The object's id

    Returns:
        tuple: Lower-Left,Upper-Right points of the computed bounding box
    """
    t = time.time()
    result,_= cv.projectPoints(obj_points,
                              rot,
                              tr,
                              cameraMatrix=K,
                              distCoeffs=None)
    #print(f"Project points took : {time.time() - t}")
    # calculate the lower-left and upper-right of the bounding bot
    # lower-left -> [minx,miny]
    LL = (result[:,...,0].min() , result[:,...,1].max())
    
    # upper-right -> [maxx,maxy]
    UR = (result[:,...,0].max() , result[:,...,1].min())

    return LL,UR

def projectModelFace(image,face,K,rot,tr):
    """Projectsa a model face (v1,v2,v3) -> vertices in 3D coordinates
    to their 2D projection on the image

    Args:
        image (np.array): Image to project the face
        face (triangle3D): Face
        K (np.array): Calibration matrix
        rot (np.array): Rotation matrix
        tr (np.array): Translation matrix

    Returns:
        np.array: Image with projected face
    """
   
    result,_= cv.projectPoints(face,
                            rot,
                            tr,
                            cameraMatrix=K,
                            distCoeffs=None)
   
    cv.fillPoly(image,[result.astype(np.int32)],color=255)
    return image
def inliers2BinaryMask():
    pass
def loadModelFaces(modelData):

    triangles3D = []
    for f in modelData['face']:
        vs = []
        for indice in f[0]:
            #print(indice)
            v = point3D(modelData["vertex"][indice][0], 
                            modelData["vertex"][indice][1], 
                            modelData["vertex"][indice][2])
            vs.append(v)
        tr = triangle3D(vs[0], 
                        vs[1], 
                        vs[2])       
        triangles3D.append(tr)

    return triangles3D

def inliers_per_2DBoundingBox():
    pass
def inliers_per_ObjectMask():
    pass
def filderCorrespFromBoundingBox(obj_corrs,LL,UR):

    # the x,y coordinates of the correspodense lie inside the 2D
    # bouding box if LL.x < x < UR.x and LL.y < y < LL.y
    condition = (obj_corrs['coord_2d'][:,0] < UR[0]) & \
                (obj_corrs['coord_2d'][:,0] > LL[0]) & \
                (obj_corrs['coord_2d'][:,1] > UR[1]) & \
                (obj_corrs['coord_2d'][:,1] < LL[1]) 
    
    return len(np.argwhere(condition))
    
    
def vis_object_confs(image,conf,savePath):
    """
    Helper function to plot data with associated colormap.
    """
    #colormaps = [ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])]


    # colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
    # my_cmap = [ListedColormap(colors, name="my_cmap")]

    viridis = mpl.colormaps['viridis'].resampled(256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248/256, 24/256, 148/256, 1])
    newcmp = [ListedColormap(newcolors)]

    
    fig, axs = plt.subplots(1, 1, figsize=(1 * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, newcmp):
        psm = ax.pcolormesh(image, cmap=cmap, rasterized=True, vmin=0, vmax=255)
        fig.colorbar(psm, ax=ax)
    plt.gca().invert_yaxis()
    bbox = dict(boxstyle ="round", fc ="1.0")
    plt.annotate("Avg conf: "+str(np.round(conf,3)),(190,170),bbox=bbox)
    plt.savefig(savePath)
    plt.close(fig)

def draw_2DBB(seg_pixels):
    # find most left and up pixel

    pass


def draw_point(x, y, marker, color):
    plt.plot(x, y, marker=marker, color=color, markersize=2)


def load_image(img_path):
    return image.imread(img_path)


def read_inliers_indices(file):
    temp2 =[]
    with open(file, 'r') as f:
        temp = f.readlines()[8:-1]
        #print(temp)print
        for t in temp:
            #print(t)
            temp2.append(t[0:-2].strip("\n").split(" "))
        #print(temp2)
    return np.array(temp2).astype(float)


def save_result(img,path,inliers,total):
    plt.title("Total corresp: {}, Inliers: {}, Ratio(I/O): {:.2f}".format(total,inliers,inliers/total))
    plt.imshow(img)
    plt.savefig(path)

def load_image_PIL(img_path):

    image = np.array(ImagePIL.open(img_path)).astype(np.float32)
    rgb_image_input = np.array(image).astype(np.float32)

    return rgb_image_input