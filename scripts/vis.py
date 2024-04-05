from pose_vis.Renderer import Renderer
from pose_vis.camera import PerspectiveCamera
from pose_vis.utils import load_model
import cv2 as cv 
from os.path import join as jn
from OpenGL.GL import *
import numpy as np
from absl import flags,app
import glob
import sys
sys.path.append("..")
from utilsHF import *
import os

def renderPose(vertices,
               indices,
               renderer,
               objID,
               conf,
               resolution,
               RT,
               K,
               savePath,
               mesh_color= [1.0,0.5,0.31],
               rgb_image = None):
    

    camera = PerspectiveCamera(resolution[0],resolution[1])
    projection = camera.fromIntrinsics(
        fx = K[0,0],
        fy = K[1,1],
        cx = K[0,2],
        cy = K[1,2],
        nearP = 1,
        farP=5000
    )

    model_ = np.eye(4)

    # configure rendering params
    uniform_vars = {"objectColor": {"value":mesh_color,"type":'glUniform3fv'}, # 1.0, 0.5, 0.31
                    "lightColor":{"value": [1.0, 1.0, 1.0],"type":'glUniform3fv'},
                    "lightPos":{"value": [0.0, 0.0 , 0.0],"type":'glUniform3fv'},
                    "viewPos":{"value": [0.0, 0.0, 0.0],"type":'glUniform3fv'},
                    "model":{"value": model_,"type":'glUniformMatrix4fv'},
                    "view":{"value":RT,"type":'glUniformMatrix4fv'},
                    "projection":{"value": projection,"type":'glUniformMatrix4fv'},
                    }
    LL,UR = boundingBoxPerInstance(RT[:3,:3],RT[:3,-1],vertices,K.reshape(3,3),FLAGS.objID)
    UL,LR = (LL[0],UR[1]),(UR[0],LL[1])


    RT = renderer.cv2gl(RT)

    # adjust lighting position
    lightPos = np.dot(np.array([RT[0,-1],RT[1,-1],RT[2,-1],1.0]),
                      np.linalg.inv(RT))
    # update uniform variables
    uniform_vars["view"]["value"] = RT
    uniform_vars["lightPos"]["value"] = [lightPos[0],lightPos[1],lightPos[2]]
    uniform_vars["viewPos"]["value"] = [-RT[0,-1], -RT[1,-1], -RT[2,-1]]

    renderer.setUniformVariables(renderer.shader_programm,uniform_vars)
    glBindVertexArray(renderer.VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    renderer.ProjectFramebuffer(renderer.framebuffer,resolution)

    if not rgb_image:
        mask = renderer.CaptureFramebufferScene(jn(savePath),saveRendered=True)
    else:
        mask = renderer.CaptureFramebufferScene(jn(savePath,'test.png'),saveRendered=False)
        renderer.draw2DBoundingBox(cv.imread(rgb_image).astype(np.float32),
                                   mask,
                                   str(objID),
                                   conf=conf,
                                   savePath=savePath,
                                   bb=np.array([UL,LR]).astype(int),
                                   buildMask=False,
                                   maskFolder=None,
                                   opacity=0.6
                                   )
        
FLAGS = flags.FLAGS


flags.DEFINE_string('config','./config.yml','Configuration file with parameters.')
flags.DEFINE_string('confs','confs.txt','Condidence valus for visualization')
flags.DEFINE_integer('objID',1,'Object id to render')
flags.DEFINE_string('images','./images','Path to input images')
flags.DEFINE_string('poses','./estPoses.json','Json file with estimated/gt poses')

def main(args):

    config = load_config(FLAGS.config)
    create_directory(config["vis_savePath"])

    images = glob.glob(FLAGS.images + "/*.png")

    # get image resolution
    resolution = cv.imread(images[0]).shape
    poses = loadJson(FLAGS.poses)

    # load 3D model
    vertices, indices = load_model(config["model3D_path"])

    # load confs for visualization
    confs = np.loadtxt(FLAGS.confs)

    # initialize renderer
    renderer = Renderer(bufferSize=(resolution[1],resolution[0]))
    renderer.load_shaders(config["vis_vertex_shader"],
                          config["vis_fragment_shader"],
                          config["vis_geometry_shader"])
    renderer.create_data_buffers(vertices,indices,attrs=[2,3,4])
    renderer.CreateFramebuffer(GL_RGB32F,GL_RGBA,GL_np.float32)


    for img in images:
        renderer.glConfig()
        key = str(int(os.path.basename(img).split('.')[0]))
        R = np.array(poses[key][0]['cam_R_m2c']).reshape(3,3)
        T = np.array(poses[key][0]['cam_t_m2c']).reshape(3,)

        RT = np.eye(4)
        RT[:3,:3] = R
        RT[:3,-1] = T

        renderPose(vertices.reshape(-1,3),
                   indices,
                   renderer,
                   objID=FLAGS.objID,
                   conf=confs[confs[:,0].astype(int) == int(key)][0,1],
                   resolution=(resolution[1],resolution[0]),
                   RT= RT,
                   K = np.array(config["K"]).reshape(3,3),
                   savePath= jn(config["vis_savePath"],os.path.basename(img)),
                   rgb_image=img
                   )

if __name__ == "__main__":
    app.run(main)