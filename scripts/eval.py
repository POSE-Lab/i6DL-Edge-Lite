from absl import flags,logging,app
from rich.console import Console
from metrics import ADD, build_percentiles
from utilsHF import *
from os.path import join as jn
import math 

FLAGS = flags.FLAGS

flags.DEFINE_string('config','./config.yml','Configuration file with parameters.')
flags.DEFINE_string('gtPoses','./scene_gt.json','Json with gt poses (EPOS format)')
flags.DEFINE_string('estPoses','./est_poses.json','Json with estimated poses (EPOS format)')

def calc_pts_diameter(pts):
    """Calculates the diameter of a set of 3D points (i.e. the maximum distance
    between any two points in the set).
    :param pts: nx3 ndarray with 3D points.
    :return: The calculated diameter.
    """
    diameter = -1.0
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def main(args):

    config = load_config(FLAGS.config)

    poses_gt = loadJson(FLAGS.gtPoses)
    est_gt = loadJson(FLAGS.estPoses)

    # load 3D model
    print('Loading 3D model from', config['model3D_path'])
    points,_ = load3DModel(config['model3D_path'])
    print("Calculating object diameter...")
    object_diameter_ = calc_pts_diameter(points)
    print("Diameter:", object_diameter_)
    adds = []
    outMetrics = {}
    
    for key,value in tqdm.tqdm(est_gt.items()):
        Rgt = np.array(poses_gt[key][0]['cam_R_m2c']).reshape(3,3)
        Tgt = np.array(poses_gt[key][0]['cam_t_m2c']).reshape(3,1)

        Rest = np.array(value[0]['cam_R_m2c']).reshape(3,3)
        Test = np.array(value[0]['cam_t_m2c']).reshape(3,1)

        add = round(ADD(points,Rgt,Rest,Tgt,Test),3)
        adds.append(add)
        outMetrics[key] = {"add": add}

    perc,pass_object_diam = build_percentiles(np.array(adds),
                             np.arange(2,30,2),
                             object_diameter=object_diameter_)
    # save metrics to file
    print(outMetrics)
    print(perc)
    print(np.array(adds).mean())
    print(f"Perc pass diameter: {pass_object_diam}")

    with open(jn(config["eval_path"],'metrics.json'),'w') as f:
        json.dump(outMetrics,f,indent=4)



if __name__ == "__main__":
    app.run(main)