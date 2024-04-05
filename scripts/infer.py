from absl import flags, app, logging
import numpy as np
import os
from utilsHF import *
from initialize import EposModel
from profiler import Profiler
from rich.console import Console
from rich.progress import Progress
from os.path import join as jn

CONSOLE = Console()
FLAGS = flags.FLAGS

flags.DEFINE_string('imagePath','./images','Path to input images')
flags.DEFINE_string('config','./config.yml','Path to config file')
flags.DEFINE_integer('objID',1,'Object id to localize')

def main(args):

    # load configuration file
    config = load_config(FLAGS.config)
    create_directory(config["eval_path"])

    # initialize epos model
    epos_model = EposModel(config,profiler=Profiler())

    if epos_model.warm_up:
        CONSOLE.log("[bold yellow]Warmin up...")
        epos_model.warm_up(epos_model.input_tensor_name,10)
        CONSOLE.log(":white_check_mark: [bold green]Done warmup")

    CONSOLE.log("Starting inference...")

    images = glob.glob(FLAGS.imagePath + "/*.png")

    poses = []
    confs = []
    timings = []
    with Progress() as progress:
        infer_task = progress.add_task("[red]Running inference...", total=len(images))
        for img in images:
            pred = epos_model.predictPose(epos_model.K,FLAGS.objID,
                                    load_image_PIL(img),
                                    epos_model.corr_path,
                                    int(os.path.basename(img).strip(".png").strip("r")),
                                    os.path.basename(img).strip(".png")) 
            confs.append([int(os.path.basename(img).strip(".png").strip("r")),pred[1]])
            progress.update(infer_task, advance=1)
            poses+=pred[0]
            timings.append(pred[-1]['total'])

    CONSOLE.log(f"[:party:]Done inference. Mean time: {np.array(timings).mean()}")

    # save confs and estimated poses
    CONSOLE.log(f"Saving estimates to {config['eval_path']}")
    saveEPOSJSON(jn(config["eval_path"],"est_poses.json"),poses)

    # saveconfs to txt
    np.savetxt(jn(config["eval_path"],'confs.txt'),
                np.array(confs))

if __name__ == "__main__":
    app.run(main)