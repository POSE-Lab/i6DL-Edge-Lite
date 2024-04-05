import numpy as np
from epos_lib import common
from epos_lib import corresp
import pyprogressivex
import cv2
import time
from utilsHF import (
    filderCorrespFromBoundingBox,
    boundingBoxPerInstance,
    pred2pose,
    loadModelFaces,projectModelFace)
from inout import save_correspondences,save_corr_csv,extract_EPOS_confs
from profiler import Profiler
import cv2 as cv




def process_image(self,K,predTime,image,predictions, im_id, scene_id, output_scale, model_store,
        renderer, task_type,corr_path,timestamp,profiler: Profiler,obj_3d_points,faces_3d_obj):
    """Estimates object poses from one image.

    Args:
      sess: TensorFlow session.
      samples: Dictionary with input data.
      predictions: Dictionary with predictions.
      im_ind: Index of the current image.
      crop_size: Image crop size (width, height).
      output_scale: Scale of the model output w.r.t. the input (output / input).
      model_store: Store for 3D object models of class ObjectModelStore.
      renderer: Renderer of class bop_renderer.Renderer().
      task_type: 6D object pose estimation task (common.LOCALIZATION or
        common.DETECTION).
      infer_name: Name of the current inference.
      infer_dir: Folder for inference results.
      vis_dir: Folder for visualizations.
    """
    # Dictionary for run times.
    run_times = {}
    run_times['prediction'] = predTime

    K = np.array(K).reshape((3,3)).astype(np.float64)

    gt_poses = None
    # Establish many-to-many 2D-3D correspondences.
    profiler.addTrackItem('establish_corres')
    profiler.start('establish_corres')
    
    
    corr = corresp.establish_many_to_many(
        obj_confs=predictions[common.PRED_OBJ_CONF][0],
        frag_confs=predictions[common.PRED_FRAG_CONF][0],
        frag_coords=predictions[common.PRED_FRAG_LOC][0],
        gt_obj_ids=[int(scene_id)],
        model_store=model_store,
        output_scale=output_scale,
        min_obj_conf=self.corr_min_obj_conf,
        min_frag_rel_conf=self.corr_min_frag_rel_conf,
        project_to_surface=self.project_to_surface,
        only_annotated_objs=(task_type == common.LOCALIZATION))
    
    run_times['establish_corr'] = profiler.stop('establish_corres')

    # PnP-RANSAC to estimate 6D object poses from the correspondences.
    profiler.addTrackItem("fitting")
    profiler.start("fitting")
    #print(len(corr.items()))
    poses = []
    pose_confidense = 0
    # TODO add exception if corr.items() is empty
    for obj_id, obj_corr in corr.items():
        
        # tf.compat.v1.logging.info(
        #   'Image path: {}, obj: {}'.format(samples[common.FLAGS.img_path][0], obj_id))

        # Number of established correspondences.
        num_corrs = obj_corr['coord_2d'].shape[0]

        # Skip the fitting if there are too few correspondences.
        min_required_corrs = 6
        if num_corrs < min_required_corrs:
            continue

        # The correspondences need to be sorted for PROSAC.
        if self.use_prosac:    
           
            sorted_inds = np.argsort(obj_corr['conf'])[::-1]
            for key in obj_corr.keys():
                obj_corr[key] = obj_corr[key][sorted_inds]
        
        #print(f"Max corrs: {self.max_correspondences}")
        # Select correspondences with the highest confidence.
        if self.max_correspondences is not None \
                and num_corrs > self.max_correspondences:
            # Sort the correspondences only if they have not been sorted for PROSAC.
            if self.use_prosac:
                keep_inds = np.arange(num_corrs)
            else:
                keep_inds = np.argsort(obj_corr['conf'])[::-1]
            keep_inds = keep_inds[:self.max_correspondences]
            for key in obj_corr.keys():
                obj_corr[key] = obj_corr[key][keep_inds]

        

        # Make sure the coordinates are saved continuously in memory.
        coord_2d = np.ascontiguousarray(
            obj_corr['coord_2d'].astype(np.float64))
        coord_3d = np.ascontiguousarray(
            obj_corr['coord_3d'].astype(np.float64))
        
        #print(len(coord_2d))
        #print(len(coord_3d))
        
        if self.fitting_method == common.PROGRESSIVE_X:
            # If num_instances == 1, then only GC-RANSAC is applied. If > 1, then
            # Progressive-X is applied and up to num_instances poses are returned.
            # If num_instances == -1, then Progressive-X is applied and all found
            # poses are returned.
            if task_type == common.LOCALIZATION:
                num_instances = 1
            else:
                num_instances = -1
            #print(self.fitting_method,self.max_instances_to_fit)
            if self.max_instances_to_fit is not None:
                num_instances = 1#min(num_instances, self.max_instances_to_fit)
            #print(len(coord_3d))
            start = time.time()
            #try:
            pose_ests, inlier_indices, pose_qualities = pyprogressivex.find6DPoses(
                x1y1=coord_2d,
                x2y2z2=coord_3d,
                K=K,
                threshold=self.inlier_thresh,
                neighborhood_ball_radius=self.neighbour_max_dist,
                spatial_coherence_weight=self.spatial_coherence_weight,
                scaling_from_millimeters=self.scaling_from_millimeters,
                max_tanimoto_similarity=self.max_tanimoto_similarity,
                max_iters=self.max_fitting_iterations,
                conf=self.required_progx_confidence,
                proposal_engine_conf=self.required_ransac_confidence,
                min_coverage=self.min_hypothesis_quality,
                min_triangle_area=self.min_triangle_area,
                min_point_number=6,
                max_model_number=1,
                max_model_number_for_optimization=self.max_model_number_for_pearl,
                use_prosac=self.use_prosac,
                log=False)
            # except RuntimeError:
            #     pose_ests, inlier_indices, pose_qualities = pyprogressivex.find6DPoses(
            #         x1y1=coord_2d,
            #         x2y2z2=coord_3d,
            #         K=K,
            #         threshold=self.inlier_thresh,
            #         neighborhood_ball_radius=self.neighbour_max_dist,
            #         spatial_coherence_weight=self.spatial_coherence_weight,
            #         scaling_from_millimeters=self.scaling_from_millimeters,
            #         max_tanimoto_similarity=self.max_tanimoto_similarity,
            #         max_iters=self.max_fitting_iterations,
            #         conf=self.required_progx_confidence,
            #         proposal_engine_conf=self.required_ransac_confidence,
            #         min_coverage=self.min_hypothesis_quality,
            #         min_triangle_area=self.min_triangle_area,
            #         min_point_number=6,
            #         max_model_number=1,
            #         max_model_number_for_optimization=self.max_model_number_for_pearl,
            #         use_prosac=self.use_prosac,
            #         log=False)
                #print(f"Progressive-x TWICE RUN: {time.time() - start}")
            #print(f"Progressive-x ONCE: {time.time() - start}")
            # Save the established correspondences (for analysis).
            
            
            obj_gt_poses = []
            if gt_poses is not None:
                obj_gt_poses = [x for x in gt_poses if x['obj_id'] == obj_id]
            pred_time = float(np.sum(list(run_times.values())))
            image_path = "test"

           

            # for the confidense calculation
            sort_inds = np.argsort(obj_corr['conf'])[::-1]
            coord_2d = obj_corr['coord_2d'][sort_inds]
            total = len(coord_2d)
            #print("ORIGINAL TOTAL",total)
            #inlier_image = np.zeros((720,1280,1),dtype=np.uint8)
            #print(inlier_indices)
            # cnt = 0
            # for i in range(len(inlier_indices)):
            #     #print(len(inlier_indices))
            #     if inlier_indices[i] == 1:
            #         xcorr,ycorr= obj_corr['coord_2d'][i]
            #         #print(xcorr,ycorr)
            #         #if xcorr < 1280 and ycorr< 720:
            #         inlier_image[int(ycorr),int(xcorr)] = 255
            #         cnt+=1
            inliers = np.count_nonzero(inlier_indices == 1)
            # inliers_inMask = np.count_nonzero(inlier_image.flatten())
            #print("INLIERS: ",inliers,total)
            #pose_confidense = (inliers / total) if total!=0 else 0
            
            save_corr_csv(model_store,predictions[common.PRED_FRAG_LOC][0],
                          inlier_indices,
                          predictions[common.PRED_OBJ_CONF][0],
                               predictions[common.PRED_FRAG_CONF][0],
                               output_scale, 
                               self.corr_min_obj_conf,
                                self.corr_min_frag_rel_conf,
                                obj_id,
                                path=corr_path,
                                imgID=im_id)
            #print(inlier_indices)
            # save_correspondences(
            #      scene_id, im_id, timestamp, obj_id, image_path, K, obj_corr,sort_inds, pred_time,
            #      "", obj_gt_poses, corr_path,inlier_indices,inliers,total)
            
            
            pose_est_success = pose_ests is not None
            if pose_est_success:
                for i in range(int(pose_ests.shape[0] / 3)):
                    j = i * 3
                    R_est = pose_ests[j:(j + 3), :3]
                    t_est = pose_ests[j:(j + 3), 3].reshape((3, 1))

                    LL,UR = boundingBoxPerInstance(R_est,t_est,obj_3d_points,K,obj_id)

                    # calculate object mask using the projection of the 
                    # decimated model faces to the image
                    # filter_mask = np.zeros((720, 1280, 1), dtype=np.uint8)
                    # # get faces
                    # for f in faces_3d_obj:
                    #     fc = np.array([[f.v1.x,f.v1.y,f.v1.z],
                    #                     [f.v2.x,f.v2.y,f.v2.z],
                    #                     [f.v3.x,f.v3.y,f.v3.z]])
                    #     projectModelFace(filter_mask,fc,K,R_est,t_est)  
                    # total_per_instance= filderCorrespFromBoundingBox(obj_corr,LL,UR)
                    # # now filter the inliers based on the binary mask
                    # inliersInMask = np.bitwise_and(filter_mask,inlier_image)
                    # corrs_in_object_mask = np.count_nonzero(inliersInMask)
                    #print(f"Inliers in mask : {np.count_nonzero(inlier_image==255)} , total: {total} ")
                    #total_per_instance= filderCorrespFromBoundingBox(obj_corr,LL,UR)
                    pose_confidense = (inliers / total) if total!=0 else 0
                    print("CONFIDENCE",pose_confidense)
                    if num_instances == 2:
                        #print(f"Looking for inliers {i}")
                        filter_mask = np.zeros((480, 640, 1), dtype=np.uint8)
                        # get faces
                        for f in faces_3d_obj:
                            fc = np.array([[f.v1.x,f.v1.y,f.v1.z],
                                            [f.v2.x,f.v2.y,f.v2.z],
                                            [f.v3.x,f.v3.y,f.v3.z]])
                            projectModelFace(filter_mask,fc,K,R_est,t_est)  

                        # get the total number of correspodences inside the object mask
                        # excludint the inliers of the other instance
                        inl_mask = 0
                        tc_in_mask = 0
                        #print(np.unique(inlier_indices))
                        for cr in range(len(obj_corr['coord_2d'])):
                            xc,yc = int(obj_corr['coord_2d'][cr][0]),int(obj_corr['coord_2d'][cr][1])
                            #print(xc,yc)
                            if yc < 480 and xc<640:
                                if filter_mask[yc,xc] == 255 \
                                    and inlier_indices[cr] ==2 or inlier_indices[cr] ==i:
                                        if inlier_indices[cr] == i:
                                            inl_mask+=1
                                        # then it is either a outlier or an inlier inside the
                                        # object mask
                                        tc_in_mask+=1

                        #print(f"Total in mask : {tc_in_mask}")
                        #cv.imshow("test",filter_mask)
                        #cv.waitKey(0)
                        #print("hjereee")
                        #total_per_instance= filderCorrespFromBoundingBox(obj_corr,LL,UR)
                        #print(total_per_instance)
                        inliers = np.count_nonzero(inlier_indices == i)
                        #print("INLIERS : ",inliers)
                        #pose_confidense = (inl_mask / tc_in_mask) if tc_in_mask!=0 else 0
                        #pose_confidense = min(pose_confidense,1.0)
                    #print(corrs_in_object_mask)
                    #cv.imshow("test",filter_mask)
                    #cv.imshow("test2",inlier_image)
                    #cv.waitKey(0)
                    #total_per_instance= filderCorrespFromBoundingBox(obj_corr,LL,UR)
                    #print(f"Total inliers per instance {total_per_instance}")
                    #inliers = np.count_nonzero(inlier_indices == i)
                    #print("INLIERS : ",inliers)
                    #pose_confidense = (inliers/total_per_instance ) if total_per_instance!=0 else 0
                    poses.append({
                        'scene_id': scene_id,
                        'im_id': im_id,
                        'obj_id': obj_id,
                        'R': R_est,
                        't': t_est,
                        'score': pose_confidense
                        #'score': pose_qualities[i],
                    })

            # for p in poses:
            #         rot,tr = pred2pose(p)
            #         LL,UR = boundingBoxPerInstance(rot,tr,obj_3d_points,K,obj_id)
            #         #print(f"LL : {LL}, UR: {UR}")
            #         filderCorrespFromBoundingBox(obj_corr,LL,UR)
            #print("POSESSS")        
            #print(poses)
        elif self.fitting_method == common.OPENCV_RANSAC:
            # This integration of OpenCV-RANSAC can estimate pose of only one object
            # instance. Note that in Table 3 of the EPOS CVPR'20 paper, the scores
            # for OpenCV-RANSAC were obtained with integrating cv2.solvePnPRansac
            # in the Progressive-X scheme (as the other methods in that table).
            pose_est_success, r_est, t_est, inliers = cv2.solvePnPRansac(
                objectPoints=coord_3d,
                imagePoints=coord_2d,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=self.max_fitting_iterations,
                reprojectionError=self.inlier_thresh,
                confidence=0.99,  # FLAGS.required_ransac_confidence
                flags=cv2.SOLVEPNP_EPNP)

            if pose_est_success:
                poses.append({
                    'scene_id': scene_id,
                    'im_id': im_id,
                    'obj_id': obj_id,
                    'R': cv2.Rodrigues(r_est)[0],
                    't': t_est,
                    'score': 0.0,  # TODO: Define the score.
                })

        else:
            raise ValueError(
                'Unknown pose fitting method ({}).'.format(self.fitting_method))

    run_times['fitting'] = profiler.stop("fitting")
    run_times['total'] = np.sum(list(run_times.values()))

    # Add the total time to each pose.
    for pose in poses:
        pose['time'] = run_times['fitting']

    return poses,pose_confidense,run_times