import argparse
import logging
import time
import ast

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

REF_POSE_PATH = '/home/msaleem/Ahmer/ComputerVision/tf-pose-estimation-master/Knee_Joints/p5.png'

from ProcrustesAnalysis.triangles import *
from ProcrustesAnalysis.translate_to_origin import *
from ProcrustesAnalysis.procrustes_analysis import *
from ProcrustesAnalysis.procrustes_distance import *
def pose_comparison():
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    ref_image = common.read_imgfile(REF_POSE_PATH, None, None)
    ref_image = cv2.resize(ref_image, (640, 480))
    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    image = cv2.resize(image, (640, 480))
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    #t = time.time()

    ref_humans = e.inference(ref_image, scales=scales)
    humans = e.inference(image, scales=scales)

    #elapsed = time.time() - t

    #logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    _,ref_centers = TfPoseEstimator.draw_humans_mine(ref_image, ref_humans, imgcopy=False)
    _,centers = TfPoseEstimator.draw_humans_mine(image, humans, imgcopy=False)

    ref_centers = list(ref_centers.values())
    centers = list(centers.values())

    ref_centers = list(sum(ref_centers, ()))
    centers = list(sum(centers, ()))

    ref_centers = np.array(ref_centers,dtype=int)
    centers = np.array(centers,dtype=int)


    shapes = []
    shapes.append(ref_centers)
    shapes.append(centers)

    #create canvas on which the triangles will be visualized
    canvas = np.full([640,480], 255).astype('uint8')
    
    #convert to 3 channel RGB for fun colors!
    canvas = cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB)
    #im = draw_shapes(canvas,shapes)


    x,y = get_translation(shapes[0])
    new_shapes = []
    new_shapes.append(shapes[0])

    for i in range(1,len(shapes)):
        new_shape = procrustes_analysis(shapes[0], shapes[i])
        new_shape[::2] = new_shape[::2] + x
        new_shape[1::2] = new_shape[1::2] + y
        new_shape = new_shape.astype(int)
        new_shapes.append(new_shape)

    pts_list = []

    for lst in new_shapes:
        temp = lst.reshape(-1,1,2)
        pts = list(map(tuple, temp))
        pts_list.append(pts)


    for i in range(18):
        cv2.circle(ref_image, tuple(pts_list[0][i][0]), 3, (255,0,0), thickness=3, lineType=8, shift=0)  
        cv2.circle(ref_image, tuple(pts_list[1][i][0]), 3, (255,255,0), thickness=3, lineType=8, shift=0)  

    cv2.imshow('tf-pose-estimation result', ref_image)
    cv2.waitKey(0)

    variations = []
    for i in range(len(new_shapes)):
        dist = procrustes_distance(shapes[0], new_shapes[i])
        variations.append(dist)

    print (variations)
    #print (new_shapes)
    #print (shapes)
    #draw_shapes(canvas, new_shapes)

    #cv2.imwrite('/home/ahmer/test.jpg',im)


    #draw_shapes()


 

if __name__ == '__main__':

    pose_comparison()

    # parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # parser.add_argument('--image', type=str, default='./images/p1.jpg')
    # parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    # parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    # parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    # args = parser.parse_args()
    # scales = ast.literal_eval(args.scales)

    # w, h = model_wh(args.resolution)
    # e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # # estimate human poses from a single image !
    # image = common.read_imgfile(args.image, None, None)
    # # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # t = time.time()
    # humans = e.inference(image, scales=scales)
    # elapsed = time.time() - t

    # logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    # image,centers = TfPoseEstimator.draw_humans_mine(image, humans, imgcopy=False)

    # for i in range(18):
    #     cv2.circle(image, centers[i], 3, (255,0,0), thickness=3, lineType=8, shift=0)

    # # cv2.imshow('tf-pose-estimation result', image)
    # # cv2.waitKey()
    # #time.sleep(1000)
    # cv2.imwrite( 'results/' + args.image.split('/')[-1:][0], image)
    # cv2.namedWindow('Pose Estimation',cv2.WINDOW_NORMAL)
    # cv2.imshow('Pose Estimation',image)
    # cv2.waitKey(0)
    # print (centers)

    '''
    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    plt.show()

    #import sys
    #sys.exit(0)

    logger.info('3d lifting initialization.')
    poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

    image_h, image_w = image.shape[:2]
    standard_w = 640
    standard_h = 480

    pose_2d_mpiis = []
    visibilities = []
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)

    for i, single_3d in enumerate(pose_3d):
        plot_pose(single_3d)
    plt.show()

    pass
    '''
