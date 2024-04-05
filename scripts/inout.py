import numpy as np
import os
import imageio
from PIL import Image as ImagePIL
import pandas as pd
from profiler import Profiler
from utilsHF import vis_object_confs

def create_multiple_columns(dataframe,values, column_names, logic_function):
    for column_name in column_names:
        dataframe[column_name] = logic_function(values)
    return dataframe

def save_profiling(profiler: Profiler,savePath): #TODO store
    profiler.write_json_report(savePath)

def save_EPOS_pose(path,pos,orie,conf,pose_status,objid, K_mat):
    orie = np.array(orie).reshape((3,3))
    _t = np.vstack((orie,np.array([0.0,0.0,0.0]).reshape((1,3))))
    print(_t)
    _f = np.array(pos).reshape((3,1))
    _f = np.append(_f,[1.0])
    print(_f)
    _f = np.hstack((_t,_f.reshape((4,1))))

    filename = path+"/Obj_"+str(objid)+"/pose.txt"
    print("saving pose at ", filename)
    file = open(filename, 'w')
    file.write(pose_status+"\n")
    np.savetxt(file, _f, newline=',\n',delimiter=',',fmt='%f')
    np.savetxt(file, K_mat, newline=', ',delimiter=',',fmt='%f')
    file.write('\n'+str(conf))
    file.close()
    #f.write(str(orie))

def make_test_dirs(parent_path,timestamp,objID): #TODO store
# dynamically create test directorys based on the number of service calls
    current_working_dir = parent_path+"/Test_"+str(timestamp)
    if not os.path.exists(current_working_dir):
        try:
            os.makedirs(current_working_dir)
        except OSError as e:
            print(e)

    #also create dirs for ach object
    if not os.path.exists(current_working_dir+"/Obj_"+str(objID)):
        try:
            os.makedirs(current_working_dir+"/Obj_"+str(objID))
        except OSError as e:
            print(e)
    if not os.path.exists(current_working_dir+"/Obj_"+str(objID)+"/corr_"):
        try:
            os.makedirs(current_working_dir+"/Obj_"+str(objID)+"/corr_")
        except OSError as e:
            print(e)
    return current_working_dir
def save_Image(path,img,timestamp,objID): #TODO store

    print("Saving image at ", path+"/Test_"+str(timestamp)+"/Obj_"+str(objID)+"/image_raw.png")
    imageio.imwrite(path+"/Test_"+str(timestamp)+"/Obj_"+str(objID)+"/image_raw.png", img)
def load_image_PIL(img_path):

    image = np.array(ImagePIL.open(img_path)).astype(np.float3232)
    rgb_image_input = np.array(image).astype(np.float3232)

    return rgb_image_input
def extract_EPOS_confs(model_store,frag_coords,obj_confs,frag_confs,logits_scale,obj_conf_threshold,frag_rel_conf,objID):

    # Mask of pixels with high enough confidence for the current class.
    obj_conf = obj_confs[:, :, objID]
    obj_mask = obj_conf > obj_conf_threshold

    # Coordinates (y, x) of shape [num_mask_px, 2].
    obj_mask_yx_coords = np.stack(np.nonzero(obj_mask), axis=0).T

    # Convert to (x, y) image coordinates.
    im_coords = np.flip(obj_mask_yx_coords, axis=1)
    im_coords = (1/logits_scale) * (im_coords.astype(np.float32) + 0.5)

    # Fragment confidences of shape [num_obj_mask_px, num_frags].
    frag_conf_masked = frag_confs[obj_mask][:, objID - 1, :]

    # Select all fragments with a high enough confidence.
    frag_conf_max = np.max(frag_conf_masked, axis=1, keepdims=True)
    frag_mask = frag_conf_masked > (frag_conf_max * frag_rel_conf)
    
    # Indices (y, x) of positive frags of shape [num_frag_mask_px, 2].
    frag_inds = np.stack(np.nonzero(frag_mask), axis=0).T

    # Collect 2D-3D correspondences.
    corr_2d = im_coords[frag_inds[:, 0]]
    corr_3d = model_store.frag_centers[objID][frag_inds[:, 1]]
    

    # Add the predicted 3D fragment coordinates.
    frag_scales = np.expand_dims(
    model_store.frag_sizes[objID][frag_inds[:, 1]], 1)
    corr_3d_local = frag_coords[obj_mask][:, objID - 1, :, :][frag_mask]
    corr_3d_local *= frag_scales
    corr_3d += corr_3d_local

    # The confidence of the correspondence is given by:
    # P(fragment, object) = P(fragment | object) * P(object)
    corr_conf_obj = obj_conf[obj_mask][frag_inds[:, 0]]
    corr_conf_frag = frag_conf_masked[frag_mask]
    corr_conf = corr_conf_obj * corr_conf_frag

    t = np.array(np.nonzero(frag_mask))
    # print("im_coord: ",t)
    # print("Fragment mask: ",frag_conf_masked[frag_inds].shape)
    # print("FRAGMENTS SHAPE: ",frag_conf_masked.shape)
    # print("corr_2d SHAPE: ",corr_2d.shape)
    # print(corr_3d.shape)

    return frag_inds[:, 0],frag_inds[:, 1],corr_2d,corr_3d,corr_conf,corr_conf_obj,\
    corr_conf_frag,frag_conf_masked[frag_inds[:,0]]

def save_corr_csv(model_store,frag_coords,inlier_indices,obj_confs,frag_confs,logits_scale,obj_conf_threshold,frag_rel_conf,objID,path,imgID):
    

    confs = extract_EPOS_confs(model_store,frag_coords,obj_confs,frag_confs,logits_scale,obj_conf_threshold,frag_rel_conf,objID)
    #print(confs[-1].shape)
    fragment_columns = [confs[-1][:,i] for i in range(0,64)]
    
    
    df = pd.DataFrame(data=[inlier_indices,
            confs[2][:,0], confs[2][:,1],
            confs[3][:,0], confs[3][:,1], confs[3][:,2],
            confs[0], confs[1], confs[4], confs[5], confs[6],*fragment_columns]).transpose()
 
    # check
    # for i in range(0,confs[1].shape[0]):
    #     print(confs[6][i],confs[-1][i,:].max())

    
    corr_path = os.path.join(
        path, 'corr{}'.format('_'),str(imgID)+"_corr")
    df.to_csv(corr_path)

    #return fconfs

    
def save_correspondences(
        scene_id, im_id, timestamp, obj_id, image_path, K, obj_pred,sort_inds, pred_time,
        infer_name, obj_gt_poses, infer_dir,inliers_indices,inliers,total): 

    # Add meta information.
    txt = '# Corr format: u v x y z px_id frag_id conf conf_obj conf_frag\n'
    txt += '{}\n'.format(image_path)
    #print(inliers_indices.shape)
    #print("INLIERS",inliers)
    txt += 'Number of Inliers :{} ,Number of outliers: {}, Ratio: {}\n'.format(inliers,total- inliers,inliers/total)

    # Add intrinsics.
    for i in range(3):
        txt += '{} {} {}\n'.format(K[i, 0], K[i, 1], K[i, 2])

    # Add ground-truth poses.
    txt += '{}\n'.format(len(obj_gt_poses))
    for pose in obj_gt_poses:
        for i in range(3):
            txt += '{} {} {} {}\n'.format(
                pose['R'][i, 0], pose['R'][i, 1], pose['R'][i, 2], pose['t'][i, 0])

    # Sort the predicted correspondences by confidence.
    px_id = obj_pred['px_id']#[sort_inds]
    frag_id = obj_pred['frag_id']#[sort_inds]
    coord_2d = obj_pred['coord_2d']#[sort_inds]
    coord_3d = obj_pred['coord_3d']#[sort_inds]
    conf = obj_pred['conf']#[sort_inds]
    conf_obj = obj_pred['conf_obj']#[sort_inds]
    conf_frag = obj_pred['conf_frag']#[sort_inds]

    # Add the predicted correspondences.
    pred_corr_num = len(coord_2d)
    txt += '{}\n'.format(pred_corr_num)
    for i in range(pred_corr_num):
        txt += '{} {} {} {} {} {} {} {} {} {} {}\n'.format(inliers_indices[i],
            coord_2d[i, 0], coord_2d[i, 1],
            coord_3d[i, 0], coord_3d[i, 1], coord_3d[i, 2],
            px_id[i], frag_id[i], conf[i], conf_obj[i], conf_frag[i])

    # Save the correspondences into a file.
    corr_suffix = infer_name
    if corr_suffix is None:
        corr_suffix = ''
    else:
        corr_suffix = '_' + corr_suffix

    corr_path = os.path.join(
        infer_dir, 'corr{}'.format(corr_suffix),str(timestamp)+"_corr")
    #   print("TIMES CALLED :",im_id)
    with open(corr_path, 'w') as f:
        f.write(txt)
