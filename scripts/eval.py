from absl import flags,logging,app
from rich.console import Console
from metrics import ADD, build_percentiles
from utilsHF import *
from os.path import join as jn

FLAGS = flags.FLAGS

flags.DEFINE_string('config','./config.yml','Configuration file with parameters.')
flags.DEFINE_string('gtPoses','./scene_gt.json','Json with gt poses (EPOS format)')
flags.DEFINE_string('estPoses','./est_poses.json','Json with estimated poses (EPOS format)')

def main(args):

    config = load_config(FLAGS.config)

    poses_gt = loadJson(FLAGS.gtPoses)
    est_gt = loadJson(FLAGS.estPoses)

    # load 3D model
    points,_ = load3DModel(config['model3D_path'])

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
                             object_diameter=265.98)
    # save metrics to file
    print(outMetrics)
    print(perc)
    print(np.array(adds).mean())
    print(f"Perc pass diameter: {pass_object_diam}")

    with open(jn(config["eval_path"],'metrics.json'),'w') as f:
        json.dump(outMetrics,f,indent=4)



if __name__ == "__main__":
    app.run(main)