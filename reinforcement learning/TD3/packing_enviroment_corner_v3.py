#!/usr/bin/env python

from sensor_msgs.msg import CompressedImage, JointState, ChannelFloat32
from std_msgs.msg import Bool, Float32MultiArray
import numpy as np
import time
from scipy import ndimage
import matplotlib.pyplot as plt
from visualization_msgs.msg import *
# from gripper import *
# from ik.helper import *
from wsg_32_common.srv import *
from robot_comm.srv import *
import rospy, math, cv2, os, pickle
# from ik.ik import fastik, fastfk, fastfk_python
from geometry_msgs.msg import PoseStamped
import json
from std_srvs.srv import *
import random
# import helpers.services as services
from scipy.spatial.transform import Rotation as R
import os
from slip_detector_both import slip_detector


class Robot_motion:
    def __init__(self):
        self.initialJointPosition = [
            -18.45, 30.26, 35.84, -21.02, -67.59, 7.2
        ]  #initial joint position
        self.cartesianOfCircle = [
            128.7, -148.3, 41.2, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfEllipse = [
            128.8, -146.7 - 158, 41.2, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfRectangle = [
            127.8, -146.7 - 251.2, 41.2, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfHexagon = [
            128.2, -146.7 - 75.405 - 1.3, 41.2, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        # self.cartesian_Vitamin = [
        #     # 127.8 + 46., -146.7 - 251.2, 41.2, 0.6998, 0.7143, 0.0051, 0.0061
        #     303.64,
        #     -213.24,
        #     42.2 + 24,
        #     0.6998,
        #     0.7143,
        #     0.0051,
        #     0.0061
        # ]
        self.cartesian_Vitamin = [
            127.8, -146.7 - 357.2, 41.2, 0.6998, 0.7143, 0.0051, 0.0061
        ]  # real one

        self.cartesianOfCircle_top = list(self.cartesianOfCircle)
        self.cartesianOfCircle_top[2] += 75
        self.jointOfCircle_top = [-19.02, 61.68, 12.58, -20.64, -75.3, 4.31]
        self.cartesianOfEllipse_top = list(self.cartesianOfEllipse)
        self.cartesianOfEllipse_top[2] += 75
        self.jointOfEllipse_top = [-17.84, 39.18, 12.83, -23.29, -54.41, 12.91]
        self.cartesianOfRectangle_top = list(self.cartesianOfRectangle)
        self.cartesianOfRectangle_top[2] += 75
        self.jointOfRectangle_top = [
            -17.94, 28.78, 8.87, -29.15, -41.54, 21.51
        ]
        self.cartesianOfHexagon_top = list(self.cartesianOfHexagon)
        self.cartesianOfHexagon_top[2] += 75
        self.jointOfHexagon_top = [-17.84, 51.08, 11.41, -20.93, -64.14, 8.32]
        self.cartesian_Vitamin_top = list(self.cartesian_Vitamin)
        self.cartesian_Vitamin_top[2] += 150
        self.joint_Vitamin_top = [-17.84, 39.18, 12.83, -23.29, -54.41, 12.91]
        self.objectCartesianDict = {'circle':[self.cartesianOfCircle,self.cartesianOfCircle_top,self.jointOfCircle_top],\
                                   'rectangle':[self.cartesianOfRectangle,self.cartesianOfRectangle_top, self.jointOfRectangle_top],\
                                   'hexagon':[self.cartesianOfHexagon,self.cartesianOfHexagon_top, self.jointOfHexagon_top],\
                                   'ellipse':[self.cartesianOfEllipse,self.cartesianOfEllipse_top, self.jointOfEllipse_top],\
                                   'vitamin':[self.cartesian_Vitamin, self.cartesian_Vitamin_top, self.joint_Vitamin_top]
                                   }

        ################################corner##########################################
        self.cartesianOfGap_Circle = [
            127.7 + 106.5, -148.3 - 66, 42.2 + 25, 0.6998, 0.7143, 0.0051,
            0.0061
        ]
        self.cartesianOfGap_Rectangle = [
            127.7 + 106.5, -148.5 - 71, 42.2 + 24.5, 0.6998, 0.7143, 0.0051,
            0.0061
        ]
        self.cartesianOfGap_Hexagon = [
            127.7 + 105 - 1.5, -148.3 - 64.8, 42.2 + 25, 0.6998, 0.7143,
            0.0051, 0.0061
        ]
        self.cartesianOfGap_Ellipse = [
            127.7 + 106.5, -148.3 - 72.2, 42.2 + 25, 0.6998, 0.7143, 0.0051,
            0.0061
        ]
        #############################parallel wall######################################
        self.cartesianOfGap_parallel_Circle = [
            127.7 + 105.3, -148.3 - 66 + 77.5, 42.2 + 24, 0.6998, 0.7143,
            0.0051, 0.0061
        ]
        self.cartesianOfGap_parallel_Rectangle = [
            127.7 + 106.35, -148.3 - 72 + 85, 42.2 + 23.5, 0.6998, 0.7143,
            0.0051, 0.0061
        ]
        self.cartesianOfGap_parallel_Hexagon = [
            127.7 + 105 - 1.5, -148.3 - 66 + 77.5, 42.2 + 24, 0.6998, 0.7143,
            0.0051, 0.0061
        ]
        self.cartesianOfGap_parallel_Ellipse = [
            127.7 + 105.3, -148.3 - 72.2 + 85, 42.2 + 24, 0.6998, 0.7143,
            0.0051, 0.0061
        ]
        #############################parallel wall rotate#################################
        self.cartesianOfGap_parallel_rotate_Circle = [
            127.7 + 105.3 + 93., -148.3 - 66 + 77.5 + 2.0, 42.2 + 22, 0.6998,
            0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_parallel_rotate_Rectangle = [
            127.7 + 106.5 + 94.5 + 60, -148.3 - 72 + 85 - 64.5 + 2.5,
            42.2 + 23.5, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_parallel_rotate_Hexagon = [
            127.7 + 105 - 1.5 + 96., -148.3 - 66 + 77.5 + 2.1, 42.2 + 24,
            0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_parallel_rotate_Ellipse = [
            127.7 + 105.3 + 94.5 + 60, -148.3 - 72.2 + 85 - 64.5 + 1.5,
            42.2 + 24, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        ################################U shape##########################################
        # self.cartesianOfGap_Ushape_Circle = [
        #     127.7 + 105.3, -148.3 - 66 - 84, 42.2 + 24, 0.6998, 0.7143, 0.0051,
        #     0.0061
        # ]
        # self.cartesianOfGap_Ushape_Rectangle = [
        #     127.7 + 106.2, -148.3 - 72 - 84.4 + 0.5, 42.2 + 23.5, 0.6998,
        #     0.7143, 0.0051, 0.0061
        # ]
        # self.cartesianOfGap_Ushape_Hexagon = [
        #     127.7 + 105 - 1.7, -147.7 - 66 - 84, 42.2 + 24, 0.6998, 0.7143,
        #     0.0051, 0.0061
        # ]
        # self.cartesianOfGap_Ushape_Ellipse = [
        #     127.7 + 105.3, -148.3 - 72.2 - 84, 42.2 + 24, 0.6998, 0.7143,
        #     0.0051, 0.0061
        # ]
        self.cartesianOfGap_Ushape_Circle = [
            127.7 + 105.3 + 155, -148.3 - 66 - 84 + 67, 42.2 + 25, 0.6998,
            0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_Ushape_Rectangle = [
            127.7 + 106.2 + 155, -148.3 - 72 - 84.4 + 0.5 + 67, 42.2 + 24.5,
            0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_Ushape_Hexagon = [
            127.7 + 105 - 2.0 + 155, -147.7 - 66 - 84 + 67, 42.2 + 25, 0.6998,
            0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_Ushape_Ellipse = [
            127.7 + 105.3 + 155, -148.3 - 72.2 - 84 + 67.6, 42.2 + 25, 0.6998,
            0.7143, 0.0051, 0.0061
        ]
        ################################U shape rotate##########################################
        # self.cartesianOfGap_Ushape_rotate_Circle = [
        #     127.7 + 105.3 + 156, -148.3 - 66 - 84 + 1.5, 42.2 + 24, 0.6998,
        #     0.7143, 0.0051, 0.0061
        # ]
        # self.cartesianOfGap_Ushape_rotate_Rectangle = [
        #     127.7 + 106.5 + 156 - 0.6, -148.3 - 72 - 84 + 15 + 1.6,
        #     42.2 + 23.5, 0.6998, 0.7143, 0.0051, 0.0061
        # ]
        # self.cartesianOfGap_Ushape_rotate_Hexagon = [
        #     127.7 + 105.7 + 158, -148.3 - 66 - 84 + 2., 42.2 + 24, 0.6998,
        #     0.7143, 0.0051, 0.0061
        # ]
        # self.cartesianOfGap_Ushape_rotate_Ellipse = [
        #     127.7 + 105.3 + 156, -148.3 - 72.2 - 84 + 15 + 1.5, 42.2 + 24,
        #     0.6998, 0.7143, 0.0051, 0.0061
        # ]
        self.cartesianOfGap_Ushape_rotate_Circle = [
            127.7 + 105.3 + 156 + 3, -148.3 - 66 - 84 + 1.5 - 10, 42.2 + 25,
            0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_Ushape_rotate_Rectangle = [
            127.7 + 106.5 + 156 - 1.1 + 3, -148.3 - 71 - 84 + 15 + 1.6 - 10,
            42.2 + 24.5, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_Ushape_rotate_Hexagon = [
            127.7 + 105.7 + 158 + 2.4, -148.3 - 66 - 84 + 2. - 10, 42.2 + 25,
            0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_Ushape_rotate_Ellipse = [
            127.7 + 105.3 + 156 + 3, -148.3 - 72.2 - 83 + 15 + 1.5 - 10,
            42.2 + 25, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        ################################hole##########################################
        # self.cartesianOfGap_hole_Circle = [
        #     127.7 + 105.3 + 136.8, -148.3 - 66 - 84 - 95.5, 42.2 + 26, 0.6998,
        #     0.7143, 0.0051, 0.0061
        # ]
        # self.cartesianOfGap_hole_Rectangle = [
        #     127.7 + 105.3 + 134.5 + 59.8, -148.3 - 66 - 84 - 93.3, 42.2 + 23,
        #     0.6998, 0.7143, 0.0051, 0.0061
        # ]
        # self.cartesianOfGap_hole_Hexagon = [
        #     127.7 + 105.3 + 132.7 + 1.5, -147.7 - 66 - 84 - 95., 42.2 + 26,
        #     0.6998, 0.7143, 0.0051, 0.0061
        # ]
        # self.cartesianOfGap_hole_Ellipse = [
        #     127.7 + 105.3 + 134.5 + 59.8, -148.3 - 66 - 84 - 93.5, 42.2 + 26.,
        #     0.6998, 0.7143, 0.0051, 0.0061
        # ]
        self.cartesianOfGap_hole_Circle = [
            127.7 + 105.3 + 155 + 136.3, -148.3 - 66 - 84 + 67 - 77.3,
            42.2 + 24, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_hole_Rectangle = [
            127.7 + 106.2 + 155 + 77., -148.3 - 72 - 84.4 + 0.5 + 67 - 42,
            42.2 + 21.5, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_hole_Hexagon = [
            127.7 + 105 - 2.0 + 155 + 138.5, -147.7 - 66 - 84 + 67 - 22,
            42.2 + 24, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_hole_Ellipse = [
            127.7 + 105.3 + 155 + 78.2, -148.3 - 72.2 - 84 + 67.6 - 43,
            42.2 + 24, 0.6998, 0.7143, 0.0051, 0.0061
        ]
        # self.cartesianOfGap_hole_Vitamin = [
        #     40.56, -437.12 + 71, 51.34, 0.6998, 0.7143, 0.0051, 0.0061
        # ]
        self.cartesianOfGap_hole_Vitamin = [
            40.56 - 3, -437.12 - 3.5, 53.34 + 16., 0.6998, 0.7143, 0.0051,
            0.0061
        ]  # the right one
        # self.cartesianOfGap_hole_Vitamin = [
        #     127.7 + 105.3 + 170 + 85, -148.3 - 52.2 - 200 - 1,
        #     42.2 + 24.5 + 28, 0.6998, 0.7143, 0.0051, 0.0061
        # ]

        ################################single wall##########################################
        self.cartesianOfGap_onewall_Circle = [
            127.7 + 105.3 + 170, -148.3 - 66 - 247, 42.2 + 26, 0.6998, 0.7143,
            0.0051, 0.0061
        ]
        self.cartesianOfGap_onewall_Rectangle = [
            127.7 + 106.5 + 170, -148.3 - 72 - 246, 42.2 + 24, 0.6998, 0.7143,
            0.0051, 0.0061
        ]
        self.cartesianOfGap_onewall_Hexagon = [
            127.7 + 105 - 1.5 + 170, -148.3 - 66 - 246, 42.2 + 26, 0.6998,
            0.7143, 0.0051, 0.0061
        ]
        self.cartesianOfGap_onewall_Ellipse = [
            127.7 + 105.3 + 170, -148.3 - 72.2 - 246, 42.2 + 24.5, 0.6998,
            0.7143, 0.0051, 0.0061
        ]


        self.cartesianOfGapDict = {'circle': self.cartesianOfGap_Circle,\
                                    'rectangle': self.cartesianOfGap_Rectangle,\
                                    'hexagon': self.cartesianOfGap_Hexagon,\
                                    'ellipse': self.cartesianOfGap_Ellipse}

        self.cartesianOfGap_onewall_Dict = {'circle': self.cartesianOfGap_onewall_Circle,\
                                    'rectangle': self.cartesianOfGap_onewall_Rectangle,\
                                    'hexagon': self.cartesianOfGap_onewall_Hexagon,\
                                    'ellipse': self.cartesianOfGap_onewall_Ellipse}

        self.cartesianOfGap_parallel_Dict = {'circle': self.cartesianOfGap_parallel_Circle,\
                                    'rectangle': self.cartesianOfGap_parallel_Rectangle,\
                                    'hexagon': self.cartesianOfGap_parallel_Hexagon,\
                                    'ellipse': self.cartesianOfGap_parallel_Ellipse}

        self.cartesianOfGap_parallel_rotate_Dict = {'circle': self.cartesianOfGap_parallel_rotate_Circle,\
                                    'rectangle': self.cartesianOfGap_parallel_rotate_Rectangle,\
                                    'hexagon': self.cartesianOfGap_parallel_rotate_Hexagon,\
                                    'ellipse': self.cartesianOfGap_parallel_rotate_Ellipse}

        self.cartesianOfGap_Ushape_Dict = {'circle': self.cartesianOfGap_Ushape_Circle,\
                                    'rectangle': self.cartesianOfGap_Ushape_Rectangle,\
                                    'hexagon': self.cartesianOfGap_Ushape_Hexagon,\
                                    'ellipse': self.cartesianOfGap_Ushape_Ellipse}

        self.cartesianOfGap_Ushape_rotate_Dict = {'circle': self.cartesianOfGap_Ushape_rotate_Circle,\
                                    'rectangle': self.cartesianOfGap_Ushape_rotate_Rectangle,\
                                    'hexagon': self.cartesianOfGap_Ushape_rotate_Hexagon,\
                                    'ellipse': self.cartesianOfGap_Ushape_rotate_Ellipse}

        self.cartesianOfGap_hole_Dict = {'circle': self.cartesianOfGap_hole_Circle,\
                                    'rectangle': self.cartesianOfGap_hole_Rectangle,\
                                    'hexagon': self.cartesianOfGap_hole_Hexagon,\
                                    'ellipse': self.cartesianOfGap_hole_Ellipse,\
                                    'vitamin': self.cartesianOfGap_hole_Vitamin}

        self.object_width = {'circle': 40.,\
                                    'rectangle': 40.,\
                                    'hexagon': 34.,\
                                    'ellipse': 40.,
                                    'vitamin': 55.}
        self.ob_cmp = {'circle': [-3.3,-1.,-1.2,3.1],\
                                    'rectangle': [7,4.0,4.,8.3],\
                                    'hexagon': [0.6,0.4,0.8,4.0],\
                                    'ellipse': [5.5,5.5,4.8,9.2]}

        self.jointAngleOfGap = [0.51, 39.24, 6.46, 0.0, 44.3, 90.51]

        self.second_corner_dis = {
            'circle': [0., 0.],
            'rectangle': [0., 15.8],
            'hexagon': [5.0, -0.2],
            'ellipse': [0.7, 15.]
        }
        self.Start_EGM = rospy.ServiceProxy('/robot2_ActivateEGM',
                                            robot_ActivateEGM)
        self.Stop_EGM = rospy.ServiceProxy('/robot2_StopEGM', robot_StopEGM)
        self.setSpeed = rospy.ServiceProxy('/robot2_SetSpeed', robot_SetSpeed)
        self.command_pose_pub = rospy.Publisher('/robot2_EGM/SetCartesian',
                                                PoseStamped,
                                                queue_size=100,
                                                latch=True)
        self.mode = 0

    def move_cart_mm(self, position):
        setCartRos = rospy.ServiceProxy('/robot2_SetCartesian',
                                        robot_SetCartesian)
        setCartRos(position[0], position[1], position[2], position[6],
                   position[3], position[4], position[5])

    def move_cart_add(self, dx=0, dy=0, dz=0):
        #Define ros services
        getCartRos = rospy.ServiceProxy('/robot2_GetCartesian',
                                        robot_GetCartesian)
        setCartRos = rospy.ServiceProxy('/robot2_SetCartesian',
                                        robot_SetCartesian)
        #read current robot pose
        c = getCartRos()
        # print([c.x, c.y, c.z, c.q0, c.qx, c.qy, c.qz])
        #move robot to new pose
        setCartRos(c.x + dx, c.y + dy, c.z + dz, c.q0, c.qx, c.qy, c.qz)

    def get_cart(self):
        getCartRos = rospy.ServiceProxy('/robot2_GetCartesian',
                                        robot_GetCartesian)
        c = getCartRos()
        return [c.x, c.y, c.z, c.qx, c.qy, c.qz, c.q0]

    def close_gripper_f(self, grasp_speed=50, grasp_force=10, width=40.):
        grasp = rospy.ServiceProxy('/wsg_32_driver/grasp', Move)
        self.ack()
        self.set_grip_force(grasp_force)
        time.sleep(0.1)
        error = grasp(width=width, speed=grasp_speed)
        time.sleep(0.5)

    def home_gripper(self):
        self.ack()
        home = rospy.ServiceProxy('/wsg_32_driver/homing', Empty)
        try:
            error = home()
        except:
            pass
        time.sleep(0.5)
        # print('error', error)

    def open_gripper(self):
        self.ack()
        release = rospy.ServiceProxy('/wsg_32_driver/move', Move)
        release(68.0, 100)
        time.sleep(0.5)

    def set_grip_force(self, val=5):
        set_force = rospy.ServiceProxy('/wsg_32_driver/set_force', Conf)
        error = set_force(val)
        time.sleep(0.2)

    def ack(self):
        srv = rospy.ServiceProxy('/wsg_32_driver/ack', Empty)
        error = srv()
        time.sleep(0.5)

    def get_jointangle(self):
        getJoint = rospy.ServiceProxy('/robot2_GetJoints', robot_GetJoints)
        angle = getJoint()
        return [angle.j1, angle.j2, angle.j3, angle.j4, angle.j5, angle.j6]

    def set_jointangle(self, angle):
        setJoint = rospy.ServiceProxy('/robot2_SetJoints', robot_SetJoints)
        setJoint(angle[0], angle[1], angle[2], angle[3], angle[4], angle[5])
        #image processing

    def object_regrasp(self, objectCartesian, objectCartesian_top, graspForce,
                       target_object, random_pose):

        # print('random_pose regrasp', random_pose)

        # print('regrasp', 'rand y', self.randomy, 'rand z', self.randomz)
        objectCartesian = np.array(objectCartesian)
        objectCartesian_top = np.array(objectCartesian_top)

        cart_positon = self.get_cart()

        self.setSpeed(600, 200)
        self.move_cart_add(0., 0., 100.)
        time.sleep(0.5)
        # print "step6"
        # raw_input("Press Enter to continue...")
        #move to the top of the obejct
        objectCartesian_top[1] += random_pose[1]
        self.move_cart_mm(objectCartesian_top)
        time.sleep(0.5)
        # print "step3"

        self.setSpeed(100, 50)

        # raw_input("Press Enter to continue...")
        objectCartesian[:3] += np.array([0, random_pose[1], random_pose[2]])
        self.move_cart_mm(objectCartesian)
        time.sleep(0.5)
        # print "step6"

        # raw_input("Press Enter to continue...")
        self.open_gripper()
        # time.sleep(1)

        self.close_gripper_f(100, graspForce, self.object_width[target_object])
        # time.sleep(1)
        self.setSpeed(600, 200)
        self.move_cart_add(0., 0., 75.)
        time.sleep(0.5)

        # if target_object == 'rectangle':
        # raw_input("Press Enter to continue...")
        cart_positon_top = list(cart_positon)
        cart_positon_top[2] = cart_positon_top[2] + 25.

        # self.setSpeed(400, 100)

        self.move_cart_mm(cart_positon_top)
        time.sleep(0.5)

        # self.setSpeed(400, 100)
        self.move_cart_mm(cart_positon)
        time.sleep(0.5)

    def return_object(self, objectCartesian, objectCartesian_top,
                      objectJoint_top, random_pose):

        # print('random_pose return', random_pose)
        objectCartesian = np.array(objectCartesian)
        objectCartesian_top = np.array(objectCartesian_top)

        self.setSpeed(600, 200)
        self.move_cart_add(0., 0., 100.)
        time.sleep(0.5)
        # print('return', 'rand y', self.randomy, 'rand z', self.randomz)
        # self.set_jointangle(objectJoint_top)
        objectCartesian_top[1] += random_pose[1]
        self.move_cart_mm(objectCartesian_top)
        time.sleep(0.5)
        # current_joint_angle = self.get_jointangle()
        # print('current_joint_angle', current_joint_angle)
        # raw_input("Press Enter to continue...")

        self.setSpeed(100, 50)

        objectCartesian[:3] += np.array([0, random_pose[1], random_pose[2]])
        self.move_cart_mm(objectCartesian)
        time.sleep(0.5)

        self.open_gripper()
        time.sleep(0.2)
        # self.setSpeed(400, 100)
        # self.move_cart_add(0., 0., 50.)

    def movedown_EGM(self, slip_monitor):
        index = 1
        rate = rospy.Rate(248)
        while ((((not slip_monitor.slip_indicator1
                  and not slip_monitor.slip_indicator2)) or index < 248 * 1.0)
               and index < int(248 * 3.5)):
            # t = time.time()
            pose = PoseStamped()
            # pose.header.stamp = now
            # pose.header.frame_id = "map"
            # Position in mm or velocity in mm/s
            pose.pose.position.x = 0.
            pose.pose.position.y = 0.
            pose.pose.position.z = -2.
            # Orientation or angular velocity in xyzw
            pose.pose.orientation.x = 0.
            pose.pose.orientation.y = 0.
            pose.pose.orientation.z = 0.
            pose.pose.orientation.w = 0.
            self.command_pose_pub.publish(pose)
            index += 1
            rate.sleep()
        # print('index', index / 248.)
        if index == int(248 * 3.5):
            return True
        else:
            return False

    def get_cartesianOfGap_corner(self, random_num_rotation, cartesianOfGap,
                                  target_object):

        if self.mode == 0:
            cartesianOfGap_rotate = list(cartesianOfGap)
            self.mode = 0

        elif self.mode == 1:
            cartesianOfGap_rotate = self.tran_rotate_robot(
                list(cartesianOfGap), self.ob_cmp[target_object][0],
                self.ob_cmp[target_object][1], 90.)

        elif self.mode == 2:
            cartesianOfGap_rotate = self.tran_rotate_robot(
                list(cartesianOfGap), self.ob_cmp[target_object][2],
                self.ob_cmp[target_object][3], -90.)

        elif self.mode == 3:
            cartesianOfGap_rotate = list(cartesianOfGap)
            cartesianOfGap_rotate[
                0] += 77. + self.second_corner_dis[target_object][0]
            cartesianOfGap_rotate[
                1] += -28. + self.second_corner_dis[target_object][1]

        return cartesianOfGap_rotate

    def get_cartesianOfGap_singlewall(self, random_num_rotation,
                                      cartesianOfGap, target_object):

        self.ob_cmp_singlewall = {'circle': [-1.8,-1.5,0.,3.],\
                       'rectangle': [7,4.,4.,8.3],\
                       'hexagon': [0.5,0.7,1.5,4.0],\
                       'ellipse': [7,4.3,5.5,8.]}

        self.second_corner_dis_singlewall = {
            'circle': 0.,
            'rectangle': 14.5,
            'hexagon': 0.,
            'ellipse': 14.5
        }

        if self.mode == 9:
            cartesianOfGap_rotate = list(cartesianOfGap)
            self.mode = 0

        elif self.mode == 10:
            cartesianOfGap_rotate = self.tran_rotate_robot(
                list(cartesianOfGap), self.ob_cmp_singlewall[target_object][0],
                self.ob_cmp_singlewall[target_object][1], 90.)

        elif self.mode == 11:
            cartesianOfGap_rotate = self.tran_rotate_robot(
                list(cartesianOfGap), self.ob_cmp_singlewall[target_object][2],
                self.ob_cmp_singlewall[target_object][3], -90.)

        elif self.mode == 12:
            cartesianOfGap_rotate = list(cartesianOfGap)
            # cartesianOfGap_rotate[
            #     0] +=  self.second_corner_dis_singlewall[target_object][0]
            cartesianOfGap_rotate[
                1] += 133.5 + self.second_corner_dis_singlewall[target_object]

        return cartesianOfGap_rotate

    def pick_up_object(self, target_object, graspForce, inposition, mode,
                       random_pose):
        # print('random_pose pick up', random_pose)
        self.mode = mode
        object_cart_info = list(self.objectCartesianDict[target_object])

        # self.randomz = random.random() * 15 - 10
        # self.randomy = (random.random() - 0.5) * 12
        # print('rand y', self.randomy, 'rand z', self.randomz)
        objectCartesian = np.array(object_cart_info[0]).copy()
        objectCartesian[:3] += np.array([0, random_pose[1], random_pose[2]])
        objectCartesian_top = np.array(object_cart_info[1]).copy()
        objectCartesian_top[:3] += np.array([0, random_pose[1], 0])

        self.setSpeed(600, 200)
        if not inposition:
            self.move_cart_mm(objectCartesian_top)
            time.sleep(0.5)
            # print "go to the top of the object"
        self.move_cart_mm(objectCartesian)
        time.sleep(0.5)
        # print "go to the object"
        # rand_num = random.random() * -2
        # self.move_cart_add(0., 0., rand_num)
        # time.sleep(0.2)
        # raw_input("Press Enter to continue...")
        self.close_gripper_f(100, graspForce, self.object_width[target_object])

        # time.sleep(1000000000)
        # print "grasp the object"
        # raw_input("Press Enter to continue...")
        self.move_cart_mm(objectCartesian_top)
        time.sleep(0.5)
        # print "go up"
        # raw_input("Press Enter to continue...")

        # self.setSpeed(400, 100)

        random_num_env = np.random.rand()
        random_num_rotation = np.random.rand()
        # random_num_env = 0.6
        # random_num_rotation = 0.3

        if self.mode < 4:
            cartesianOfGap = list(self.cartesianOfGapDict[target_object])
            cartesianOfGap_rotate = self.get_cartesianOfGap_corner(
                random_num_rotation, cartesianOfGap, target_object)
        elif self.mode == 4:
            cartesianOfGap = self.cartesianOfGap_parallel_Dict[target_object]
            cartesianOfGap_rotate = list(cartesianOfGap)
        elif self.mode == 5:
            cartesianOfGap = self.cartesianOfGap_parallel_rotate_Dict[
                target_object]
            cartesianOfGap_rotate = list(cartesianOfGap)
        elif self.mode == 6:
            cartesianOfGap = list(
                self.cartesianOfGap_Ushape_Dict[target_object])
            cartesianOfGap_rotate = list(cartesianOfGap)
        elif self.mode == 7:
            cartesianOfGap = list(
                self.cartesianOfGap_Ushape_rotate_Dict[target_object])
            cartesianOfGap_rotate = list(cartesianOfGap)
        elif self.mode == 8:
            cartesianOfGap = list(self.cartesianOfGap_hole_Dict[target_object])
            cartesianOfGap_rotate = list(cartesianOfGap)
        elif 9 <= self.mode <= 12:
            cartesianOfGap = list(
                self.cartesianOfGap_onewall_Dict[target_object])
            cartesianOfGap_rotate = self.get_cartesianOfGap_singlewall(
                random_num_rotation, cartesianOfGap, target_object)

        # if target_object == 'rectangle':
        cart_positon_top = list(cartesianOfGap_rotate)
        cart_positon_top[1] += random_pose[1]
        cart_positon_top[2] += random_pose[2] + 25.
        # cart_positon_top[2] = cart_positon_top[2] + 25.
        self.move_cart_mm(cart_positon_top)
        time.sleep(0.5)

        cartesianOfGap_rotate[:3] += np.array(
            [0, random_pose[1], random_pose[2]])
        self.move_cart_mm(cartesianOfGap_rotate)

        time.sleep(0.5)
        # print('robot mode', self.mode)
        # raw_input("Press Enter to continue...")

    def robot_reset(self):
        print('go to the initial position')
        self.set_jointangle(self.initialJointPosition)

    def tran_rotate_robot(self, targetCartesian, x, y, theta):
        # targetCartesian = np.array(curren_tart)  #current cart
        relativeVector = np.array([0., 0., 0.])  # 0 12 380
        ratationMatrix1 = (R.from_quat(targetCartesian[3:])).as_dcm()
        ratationMatrix2 = (R.from_euler('z', -theta,
                                        degrees=True)).as_dcm()  #rotate theta
        targetQuaternion = R.from_dcm(
            ratationMatrix2.dot(ratationMatrix1)).as_quat()
        targetCartesian[:3] = np.array(
            targetCartesian[:3]) + ratationMatrix1.dot(
                relativeVector) - ratationMatrix2.dot(ratationMatrix1).dot(
                    relativeVector)
        targetCartesian[3:] = targetQuaternion
        targetCartesian[0] = targetCartesian[0] + x  # add
        targetCartesian[1] = targetCartesian[1] + y  # add
        return targetCartesian

    def error_converter(self, error_x, error_y):
        if self.mode == 0:
            error_x_new = error_x
            error_y_new = error_y
        if self.mode == 1:
            error_x_new = -error_y
            error_y_new = error_x
        elif self.mode == 2:
            error_x_new = error_y
            error_y_new = -error_x
        elif self.mode == 3:
            error_x_new = -error_x
            error_y_new = -error_y
        elif self.mode == 4:  # parallel wall
            error_x_new = error_x
            error_y_new = error_y
        elif self.mode == 5:  # parallel wall rotate
            error_x_new = error_x
            error_y_new = error_y
        elif self.mode == 6:  # U shape
            error_x_new = error_x
            error_y_new = error_y
        elif self.mode == 7:  # U shape rotate
            error_x_new = -error_x
            error_y_new = -error_y
        elif self.mode == 8:  # hole
            error_x_new = error_x
            error_y_new = error_y
        if self.mode == 9:
            error_x_new = error_x
            error_y_new = error_y
        if self.mode == 10:
            error_x_new = -error_y
            error_y_new = error_x
        elif self.mode == 11:
            error_x_new = error_y
            error_y_new = -error_x
        elif self.mode == 12:
            error_x_new = -error_x
            error_y_new = -error_y
        return error_x_new, error_y_new


class Packing_env:
    def __init__(self, num_frame):
        self.slip_monitor = slip_detector()
        self.robot = Robot_motion()
        self.done = False
        self.object_name_list = ['circle', 'hexagon', 'ellipse', 'rectangle']
        # self.object_name_list = ['circle', 'hexagon', 'ellipse']
        # self.object_name_list = ['hexagon', 'circle']
        # self.object_name_list = ['vitamin']
        self.target_object = self.object_name_list[self.select_object()]
        self.x_error = 0
        self.y_error = 0
        self.theta_error = 0
        self.error_generator()
        self.max_x_error = 14  #mm
        self.max_y_error = 14  #mm
        self.max_theta_error = 20  #degree
        self.reward = 0
        self.state = None
        self.rgrasp_counter = 0
        self.num_frame = num_frame
        self.save_data = True

    def select_object(self):
        rnum = random.random()
        ob_index = 0
        if rnum <= 0.2:
            ob_index = 0
        elif 0.2 < rnum <= 0.4:
            ob_index = 1
        elif 0.4 < rnum <= 0.6:
            ob_index = 2
        else:
            ob_index = 3
        return ob_index

    def select_image(self, success):

        image_g1, image_g2, time_g1, time_g2, motion_g1, motion_g2 = [], [], [], [], [], []
        self.data1 = list(self.data1)
        self.data2 = list(self.data2)
        # np.save('data_gelsight1.npy', self.data1)
        # np.save('data_gelsight2.npy', self.data2)

        for i in range(len(self.data1)):
            image_g1.append(self.data1[i][0])
            image_g2.append(self.data2[i][0])
            time_g1.append(self.data1[i][1])
            time_g2.append(self.data2[i][1])
            motion_g1.append(self.data1[i][2])
            motion_g2.append(self.data2[i][2])

        image_g1 = np.array(image_g1)
        image_g2 = np.array(image_g2)
        time_g1 = np.array(time_g1)
        time_g2 = np.array(time_g2)
        motion_g1 = np.array(motion_g1)
        motion_g2 = np.array(motion_g2)
        num_of_frame = 12
        motion_thre = 0.7
        start_num = -75
        kernel = np.ones((4, )) / 4.
        motion_g1_smooth = np.convolve(kernel,
                                       motion_g1[start_num:],
                                       mode='same')
        motion_g2_smooth = np.convolve(kernel,
                                       motion_g2[start_num:],
                                       mode='same')
        motion_diff = np.abs(motion_g1_smooth - motion_thre).tolist()
        start_index1 = motion_diff.index(min(motion_diff))
        motion_diff = np.abs(motion_g2_smooth - motion_thre).tolist()
        start_index2 = motion_diff.index(min(motion_diff))

        if time_g1[start_num + start_index1] > time_g2[start_num +
                                                       start_index2]:
            start_frame2 = max(-75, start_num + start_index2 - 5)
            time_diff = np.abs(time_g1[start_num:] -
                               time_g2[start_frame2]).tolist()
            index = time_diff.index(min(time_diff))
            start_frame1 = start_num + index
            # print('gelsight 2 is early')
        else:
            start_frame1 = max(-75, start_num + start_index1 - 5)
            time_diff = np.abs(time_g2[start_num:] -
                               time_g1[start_frame1]).tolist()
            index = time_diff.index(min(time_diff))
            start_frame2 = start_num + index
            # print('gelsight 1 is early')

        # image2save_g1 = image_g1[start_frame1:start_frame1 + 8]
        # image2save_g2 = image_g2[start_frame2:start_frame2 + 8]
        imageseq1 = np.array(image_g1[start_frame1:min(start_frame1 + 20, -1)])
        imageseq2 = np.array(image_g2[start_frame2:min(start_frame2 + 20, -1)])
        # flow_g1 = motion_g1[start_frame1:]
        # flow_g2 = motion_g2[start_frame2:]

        num_frame = min(imageseq1.shape[0], imageseq2.shape[0])
        sample_index = np.linspace(0, num_frame - 1,
                                   num=num_of_frame).astype(np.uint8)

        image2save_g1 = imageseq1[sample_index, :, :, :]
        image2save_g2 = imageseq2[sample_index, :, :, :]
        # flow2save_g1 = flow_g1[sample_index]
        # flow2save_g2 = flow_g2[sample_index]

        # print('start_index', start_index1, start_index2)
        # print('start_frame', start_frame1, start_frame2)
        # print('image shape', image2save_g1.shape, image2save_g2.shape)
        # np.save('image2save_g1.npy', image2save_g1)
        # np.save('image2save_g2.npy', image2save_g2)
        return image2save_g1, image2save_g2

    def step(self, action, check_bound):
        action_origin = np.array(action)
        action, r_matrix = self.action_convertor(
            action, check_bound)  # action in gripper frame
        # print("converted action", action)
        if check_bound:
            Fail_sign, reward, state_full = self.check_boundary(action)
        else:
            Fail_sign = False
            reward = 0
            state_full = np.array(action)

        if not Fail_sign:
            # print('data number', folder_num)
            current_cart = self.robot.get_cart()
            self.robot.setSpeed(100, 50)
            if self.mode in [3, 7, 12]:
                targetCartesian = self.robot.tran_rotate_robot(
                    np.array(current_cart), -action[0], -action[1], action[2])
            else:
                targetCartesian = self.robot.tran_rotate_robot(
                    np.array(current_cart), action[0], action[1], action[2])

            self.robot.move_cart_mm(targetCartesian)
            self.slip_monitor.restart1 = True
            self.slip_monitor.restart2 = True
            time.sleep(0.5)
            # joint6 = -(self.robot.get_jointangle()[-1] - 1.0) / 180. * np.pi
            if self.mode in [0, 6, 8, 9]:
                joint6 = (self.theta_error) / 180. * np.pi
            elif self.mode in [1, 10]:
                joint6 = (self.theta_error - 90.) / 180. * np.pi
            elif self.mode in [2, 11]:
                joint6 = (self.theta_error + 90.) / 180. * np.pi
            elif self.mode in [3, 7, 12]:
                joint6 = (self.theta_error + 180.) / 180. * np.pi

                # if self.mode != 3 and self.mode != 7 and self.mode != 12:
            r_matrix_next = np.array([[np.cos(joint6), -np.sin(joint6), 0],\
                                 [np.sin(joint6), np.cos(joint6), 0], \
                                 [0, 0, 1]])
            # else:
            #     r_matrix_next = np.array([[np.cos(joint6+180), -np.sin(joint6+180), 0],\
            #                      [np.sin(joint6+180), np.cos(joint6+180), 0], \
            #                      [0, 0, 1]])
            # raw_input("Press Enter to continue...")
            ret = self.robot.Start_EGM(True, 86400)
            success_sign = self.robot.movedown_EGM(self.slip_monitor)
            ret = self.robot.Stop_EGM()
            time.sleep(0.3)
            self.data1 = self.slip_monitor.data1
            self.data2 = self.slip_monitor.data2
            self.robot.move_cart_add(0., 0., 3.)
            time.sleep(0.2)
            # if not os.path.isfile(
            #         '/home/mcube/siyuan/Packing_RL/utils/gelsight1_data.npy'):
            #     return [], [], True, [0, 0, 0], [], [], []

            if success_sign:
                print(
                    'object inserted!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                )
                if np.abs(state_full[2]) < 4.:
                    self.reward = reward + 80
                else:
                    self.reward = reward + 40
                # self.state = np.array(
                #     self.slip_monitor.image2save_g1[-self.num_frame * 3:])
                image2save_g1, image2save_g2 = self.select_image(True)
                self.state = np.concatenate((np.array(image2save_g1),\
                        np.array(image2save_g2)),axis = 0)
                self.done = True
            else:
                # self.state = np.array(
                #     self.slip_monitor.image2save_g1[-self.num_frame * 3:])
                image2save_g1, image2save_g2 = self.select_image(False)
                self.state = np.concatenate((np.array(image2save_g1),\
                        np.array(image2save_g2)),axis = 0)
                self.reward = reward - 5
                self.done = False
            # self.robot.setSpeed(200,10)
            # self.slip_monitor.restartDetector1()
            # self.slip_monitor.restartDetector2()

        else:
            self.reward = reward - 40
            self.done = True
            image2save_g1, image2save_g2 = self.select_image(True)
            # self.state = np.array(
            #     self.slip_monitor.image2save_g1[-self.num_frame * 3:])
            # self.state = np.concatenate((np.array(self.slip_monitor.image2save_g1[-45:]),\
            #             np.array(self.slip_monitor.image2save_g2[-45:])),axis = 0)
            self.state = np.concatenate((np.array(image2save_g1),\
                        np.array(image2save_g2)),axis = 0)
            r_matrix_next = np.array(r_matrix)
            print('Failed?????????????????????????????????????????????')

        return self.state, self.reward, self.done, state_full, action, r_matrix, r_matrix_next

    def action_convertor(self, action, notinitial):
        # print 'action'
        time.sleep(0.5)
        # joint6 = -(self.robot.get_jointangle()[-1] + 24.36) / 180. * np.pi
        if self.mode in [0, 6, 8, 9]:
            joint6 = (self.theta_error * notinitial) / 180. * np.pi
        elif self.mode in [1, 10]:
            joint6 = (self.theta_error * notinitial - 90.) / 180. * np.pi
        elif self.mode in [2, 11]:
            joint6 = (self.theta_error * notinitial + 90.) / 180. * np.pi
        elif self.mode in [3, 7, 12]:
            joint6 = (self.theta_error * notinitial + 180.) / 180. * np.pi

        # if notinitial:
        #     joint6 = (self.theta_error * notinitial) / 180. * np.pi
        # else:
        #     joint6 = 0.
        # print("joint6", joint6)
        r_matrix = np.array([[np.cos(joint6), -np.sin(joint6), 0],\
                             [np.sin(joint6), np.cos(joint6), 0], \
                             [0, 0, 1]])
        # print 'r_matrix', r_matrix
        action_world = r_matrix.dot(np.array(action))
        # print 'action_world', action_world
        # action_world = action_world.tolist()
        # action_world.append(action[2])
        # print 'action_world', action_world
        # if self.mode == 3 or self.mode == 7 or self.mode == 12:
        #     action_world[0] *= -1
        #     action_world[1] *= -1
        # r_matrix = np.array([[np.cos(joint6+180), -np.sin(joint6+180), 0],\
        #                  [np.sin(joint6+180), np.cos(joint6+180), 0], \
        #                  [0, 0, 1]])
        return action_world, r_matrix

    def check_rgrasp(self, x_error, theta_error, x_error_previous,
                     theta_error_previous):
        if x_error * x_error_previous > 0 and abs(theta_error) < 8.0 and abs(
                theta_error_previous) < 8.0:
            self.rgrasp_counter += 1
        else:
            self.rgrasp_counter = 0

    def reward_function(self, error_before, error_after):
        error_before *= -1
        error_after *= -1
        if error_before > 0:
            error_before = 0
        if error_after > 0:
            error_after = 0

        reward = error_after - error_before

        # if reward > 0:
        #     reward *= 2
        return reward

    def reward_function_2(self, error_before, error_after):
        if error_after < 0:
            reward = 0
        else:
            reward = -error_after**2

        return reward

    def reward_function_3(self, error_before, error_after):
        margin = 4.
        if -margin <= error_before <= 0. and -margin <= error_after <= 0.:  # still in the safe zoom ok
            reward = 0
        elif error_before >= 0. and error_after >= 0.:  # still in the collision zoom, depends on the action
            reward = (abs(error_before) - abs(error_after))
        elif error_before < -margin and error_after < -margin:  # still in the safe zoom, but the gap too big, depends on the action
            reward = abs(error_before) - abs(error_after)
        elif -margin <= error_before <= 0. and error_after >= 0.:  # collied with the enviroment $very bad$
            reward = -error_after
        elif error_before < -margin and error_after >= 0.:  # collied with the enviroment $very bad$
            reward = -error_after + error_before + margin
        elif -margin <= error_before <= 0. and error_after <= -margin:  # increased the gap $bad$
            reward = error_after + margin
        elif error_before >= 0 and -margin <= error_after <= 0.:  # correct the error $very good$
            reward = error_before
        elif error_before >= 0 and error_after <= -margin:  # correct the error good but overshoot
            reward = error_before + error_after + margin
        elif error_before <= -margin and -margin <= error_after <= 0.:  # make the gap smaller good
            reward = -margin - error_before

        return reward

    def check_boundary(self, action):
        theta_error_acc = self.theta_error + action[2]

        # if self.mode != 3 and self.mode != 7 and self.mode != 12:
        x_error_acc = self.x_error + action[0]
        y_error_acc = self.y_error + action[1]

        reward_x = self.reward_function_3(-self.x_error, -x_error_acc) * 3
        reward_y = self.reward_function_3(self.y_error, y_error_acc) * 3

        reward_theta = (abs(self.theta_error) - abs(theta_error_acc)) * 1
        # reward_theta = -(theta_error_acc)**2
        # print 'accumulated error: ', x_error_acc, theta_error_acc
        # print('accumulated error', [x_error_acc, y_error_acc, theta_error_acc])
        x_offset = -2
        y_offset = 2
        # x_offset = 0
        # y_offset = 0
        if abs(x_error_acc + x_offset) > self.max_x_error or abs(
                y_error_acc + y_offset) > self.max_y_error or abs(
                    theta_error_acc) > self.max_theta_error:
            # print("cross the max error limitation, please restart")
            self.error_generator()
            Fail_sign = True
            # reward = -200
        else:
            self.x_error = x_error_acc
            self.y_error = y_error_acc
            self.theta_error = theta_error_acc
            Fail_sign = False
            # reward = (reward_x > 0)*reward_x*2 + (reward_x <= 0)*reward_x*3 + \
            # (reward_y > 0)*reward_y*2 + (reward_y <= 0)*reward_y*3 + \
            # (reward_theta > 0)*reward_theta*2 + (reward_theta <= 0)*reward_theta*3
        reward = reward_x + reward_y + reward_theta
        state_full = np.array([x_error_acc, y_error_acc, theta_error_acc])
        # self.check_rgrasp(x_error_acc, theta_error_acc, self.x_error, self.theta_error)

        return Fail_sign, reward, state_full

    def reset(self, random_pose):
        object_cart_info = list(
            self.robot.objectCartesianDict[self.target_object])
        self.robot.return_object(object_cart_info[0], object_cart_info[1],
                                 object_cart_info[2], random_pose)
        self.target_object = self.object_name_list[self.select_object()]
        self.error_generator()

    def mode_generator(self):
        # self.mode = random.randint(6, 7)
        self.mode = 8
        # self.mode = random.randint(9, 12)
        # self.mode = 11
        # self.mode = random.randint(0, 3)
        # self.mode = 9
        # self.mode = 2
        # self.mode = 6
        # if self.mode == 6:
        #     self.mode = 4
        # elif self.mode == 7:
        #     self.mode = 5

    def U_shape_error_generator(self):
        rand_error = random.random()
        if self.target_object != 'hexagon':
            add_on = 5
        else:
            add_on = 7

        add_on_x = 2
        add_on_y = 2

        if rand_error <= 0.1:
            self.x_error = 0
            self.y_error = random.random() * 6. + 1.
        elif 0.1 < rand_error <= 0.2:
            self.x_error = random.random() * 6. + add_on
            self.y_error = 0
        elif 0.2 < rand_error <= 0.3:
            self.x_error = random.random() * -6. - 1.
            self.y_error = 0
        elif 0.3 < rand_error <= 0.65:
            self.x_error = random.random() * 6. + add_on
            self.y_error = random.random() * 6. + 1.
        else:
            self.x_error = random.random() * -6. - 1.
            self.y_error = random.random() * 6. + 1.

        self.x_error += add_on_x
        self.y_error -= add_on_y

    def hole_error_generator(self):
        # if self.target_object == 'hexagon':
        #     add_on_x = 2.5
        #     add_on_y = 1
        # else:
        add_on_x = 2
        add_on_y = 2
        mag = 6.

        rand_error_x1 = random.random() * mag + 1
        rand_error_x2 = random.random() * -mag - 1

        rand_error_y1 = random.random() * mag + 1
        rand_error_y2 = random.random() * -mag - 1

        if random.random() <= 0.5:
            self.x_error = rand_error_x1 + add_on_x
        else:
            self.x_error = rand_error_x2 + add_on_x

        if random.random() <= 0.5:
            self.y_error = rand_error_y1 - add_on_y
        else:
            self.y_error = rand_error_y2 - add_on_y

    def error_generator(self):
        # self.pose_y_error = 0.  #(random.random() - 0.5) * 10
        # self.pose_z_error =
        # self.pose_z_error = 0.
        # self.random_pose = [0., self.pose_y_error, self.pose_z_error]
        self.mode_generator()
        self.x_error = random.random() * -6. - 1.
        self.y_error = random.random() * 6. + 1.
        self.theta_error = (random.random() - 0.5) * 20
        if random.random() < 0.2:
            self.theta_error = 0

        if self.mode < 4:
            rand_error = random.random()
            if rand_error <= 0.2:
                self.x_error = 0.
            elif 0.2 < rand_error <= 0.4:
                self.y_error = 0.

        elif self.mode == 4:
            self.y_error = 0.
            if self.target_object != 'hexagon':
                self.x_error = (random.random() - 0.5) * 12 + 1
            else:
                self.x_error = (random.random() - 0.5) * 14 + 2
                if 0 < self.x_error <= 4.:
                    self.x_error = random.random() * 5 + 4.

        elif self.mode == 5:
            self.x_error = 0.
            self.y_error = (random.random() - 0.5) * 12 - 1

        elif self.mode == 6 or self.mode == 7:
            self.U_shape_error_generator()

        elif self.mode == 8:
            self.hole_error_generator()

        elif self.mode in [9, 10, 11, 12]:
            # self.x_error = random.random() * 7. + 1.
            self.x_error = 0
            self.y_error = random.random() * 6. + 1.
            # self.y_error = 3

        # elif self.mode == 10:
        #     self.x_error = random.random() * -7. - 1.
        #     self.y_error = 0

        # elif self.mode == 11:
        #     self.x_error = random.random() * 7. + 1.
        #     self.y_error = 0
        print('initial error', 'x', self.x_error, 'y', self.y_error, 'theta',
              self.theta_error, 'mode', self.mode)

    def regrasp(self, graspForce, random_pose):
        object_cart_info = list(
            self.robot.objectCartesianDict[self.target_object])
        self.robot.object_regrasp(object_cart_info[0], object_cart_info[1],
                                  graspForce, self.target_object, random_pose)
