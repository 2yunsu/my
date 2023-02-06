from __future__ import print_function
import rospy
import math
from sensor_msgs.msg import Image, PointCloud2, JointState
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
from geometry_msgs.msg import WrenchStamped
import sys
import cv2
import ros_numpy
from hsrb_interface import Robot
import trajectory_msgs.msg
from hsrb_interface import geometry
import controller_manager_msgs.srv
import numpy as np
from std_msgs.msg import Int16
from tmc_manipulation_msgs.srv import (
    SafeJointChange,
    SafeJointChangeRequest
)

_CONNECTION_TIMEOUT = 10.0

class VisionController(object):
    def __init__(self):
        self.rgb_img = None
        self.depth_img = None
        self.box_info = None
        self.bridge = CvBridge()

        rgb_topic = '/hsrb/head_rgbd_sensor/rgb/image_rect_color' # '/camera/color/image_raw'      #head_camera '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        depth_topic = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw' # '/camera/depth/image_raw'    #'/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'
        cloud_topic = 'hsrb/head_rgbd_sensor/depth_registered/rectified_points'
        box_info_topic = 'box_info'

        self._rgb_sub = rospy.Subscriber(rgb_topic, Image, self._rgb_callback)
        self._depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_callback)
        self._pointcloud_sub = rospy.Subscriber(cloud_topic, PointCloud2, self._pc_callback)
        self.box_info_sub = rospy.Subscriber(box_info_topic, Int32MultiArray, self._box_info_callback)

    def _rgb_callback(self, img_msg):
        self.rgb_img = self.bridge.imgmsg_to_cv2(img_msg, 'passthrough')

    def _depth_callback(self, data):
        self.depth_img = self.bridge.imgmsg_to_cv2(data, "32FC1")


    def _pc_callback(self, point_msg):
        self.pc = ros_numpy.numpify(point_msg)

    def _box_info_callback(self, box_info):
        self.box_info = box_info.data
        self.box_info = np.array(self.box_info).reshape((-1, 4))

    def get_object_pc(self):
        #if len(self.box_info) == 0:  # end project condition
         #   return None, None, None
        min_z_obj_dist = 100
        min_object_pc = None
        min_box_info = None

        for box in self.box_info:
            center_rgb_x = box[0] + box[2] // 2
            center_rgb_y = box[1] + box[3] // 2
            object_pc_z = self.pc[center_rgb_y][center_rgb_x][2]
            object_real_w = self.pc[box[1] + box[3]][box[0] + box[2]][0] - self.pc[box[1]][box[0]][0]
            object_real_h = self.pc[box[1] + box[3]][box[0] + box[2]][1] - self.pc[box[1]][box[0]][1]
            print(object_real_w, object_real_h)
            if object_real_w > 0.3 or object_real_h > 0.3:
                continue
            if min_z_obj_dist > object_pc_z:
                min_z_obj_dist = object_pc_z
                min_object_pc = self.pc[center_rgb_y][center_rgb_x]
                min_box_info = box
        return min_object_pc, min_box_info

class ClassifyController(object):
    def __init__(self):
        self.class_label = -1
        self.class_label_list = []
        grasping_topic = '/unseen_cnn_learning/classify'
        rospy.Subscriber(grasping_topic, Int16, self._classify_callback)

    def _classify_callback(self, x):
        if x.data == 0:  # can
            self.class_label = 0
        elif x.data == 1: # plastic
            self.class_label = 1
        elif x.data == 2: # plastic
            self.class_label = 2
        else: # error
            self.class_label = 3
        self.class_label_list.append(self.class_label)

    def reset_class_label_list(self):
        self.class_label_list = []

    def get_result(self):
        _res = [self.class_label_list.count(0),
                self.class_label_list.count(1),
                self.class_label_list.count(2)]
        return _res.index(max(_res))






class JointController(object):
    """Control arm and gripper"""

    def __init__(self):
        joint_control_service = '/safe_pose_changer/change_joint'
        self._joint_control_client = rospy.ServiceProxy(
            joint_control_service, SafeJointChange)
        self.gripper_pub = rospy.Publisher('/hsrb/gripper_controller/command', trajectory_msgs.msg.JointTrajectory,
                                           queue_size=10)

        # Wait for connection
        try:
            self._joint_control_client.wait_for_service(
                timeout=_CONNECTION_TIMEOUT)

        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

    def move_to_joint_positions(self, goal_pose):
        """Joint position control"""
        if goal_pose == 'pick_position_for_vertical':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(
                ['wrist_roll_joint', 'wrist_flex_joint', 'arm_roll_joint', 'arm_lift_joint', 'arm_flex_joint'])
            goal_joint_states.position.extend([-1.57, -1.57, 0, 0.3, -1.57])
        elif goal_pose == 'pick_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['wrist_flex_joint', 'arm_roll_joint', 'arm_lift_joint', 'arm_flex_joint'])
            goal_joint_states.position.extend([-1.57, 0, 0.3, -1.57])
        elif goal_pose == 'search_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['head_tilt_joint'])
            # goal_joint_states.position.extend([-0.59])
            goal_joint_states.position.extend([-0.87])
        elif goal_pose == 'go_to_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'])
            goal_joint_states.position.extend([0, 0, -1.57, 0])
        elif goal_pose == 'place_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['arm_flex_joint', 'wrist_flex_joint'])
            goal_joint_states.position.extend([-1.2, -0.4])
        elif goal_pose == 'start_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['arm_flex_joint', 'wrist_flex_joint', 'arm_roll_joint','arm_lift_joint'])
            goal_joint_states.position.extend([0, -1.57, -1.57, 0])
        elif goal_pose == 'scan_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['head_tilt_joint', 'arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint'])
            goal_joint_states.position.extend([-0.8, 0, -1, 0, 1.2])
        elif goal_pose == 'scan_x1':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(
                ['wrist_roll_joint'])
            goal_joint_states.position.extend([-1.919])
        elif goal_pose == 'scan_x2':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(
                ['wrist_roll_joint'])
            goal_joint_states.position.extend([3.665])
        elif goal_pose == 'scan_y1':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(
                ['wrist_flex_joint'])
            goal_joint_states.position.extend([-0.2])
        elif goal_pose == 'scan_y2':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(
                ['wrist_flex_joint'])
            goal_joint_states.position.extend([1.2])
        elif goal_pose == 'scan_z1':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(
                ['arm_roll_joint'])
            goal_joint_states.position.extend([-1.5])
        elif goal_pose == 'scan_z2':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(
                ['arm_roll_joint'])
            goal_joint_states.position.extend([1.8])
        else:
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['arm_flex_joint', 'wrist_flex_joint'])
            goal_joint_states.position.extend([-1.2, -0.4])
        try:
            req = SafeJointChangeRequest(goal_joint_states)
            res = self._joint_control_client(req)
        except rospy.ServiceException as e:
            rospy.logerr(e)
            return False
        return res.success

    def grasp(self, position):
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = ["hand_motor_joint"]
        p = trajectory_msgs.msg.JointTrajectoryPoint()
        p.positions = [position]
        p.velocities = [0]
        p.effort = [0.1]
        p.time_from_start = rospy.Time(3)
        traj.points = [p]

        self.gripper_pub.publish(traj)

    def move_to_learn_object(self, whole_body):
        # scan position1
        whole_body.move_to_joint_positions({'head_tilt_joint': -0.8})
        whole_body.move_to_joint_positions({'arm_lift_joint': 0})
        whole_body.move_to_joint_positions({'arm_flex_joint': -0.7})
        whole_body.move_to_joint_positions({'arm_roll_joint': 0.3})
        whole_body.move_to_joint_positions({'wrist_flex_joint': 1.1})

        whole_body.move_to_joint_positions({'wrist_roll_joint': -1.919})  # spin
        whole_body.move_to_joint_positions({'wrist_roll_joint': 3.665})

        # scan_pposition2
        whole_body.move_to_joint_positions({'head_tilt_joint': -0.6})
        whole_body.move_to_joint_positions({'arm_lift_joint': 0})
        whole_body.move_to_joint_positions({'arm_roll_joint': -0.5})
        whole_body.move_to_joint_positions({'wrist_flex_joint': -0.6})
        whole_body.move_to_joint_positions({'arm_flex_joint': 0})

        whole_body.move_to_joint_positions({'wrist_roll_joint': -1.919})


def get_config():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--queue_size', type=int, default=100)
    parser.add_argument('--default_x_offset', type=float, default=-0.05, help='number of features')
    parser.add_argument('--default_z_offset', type=float, default=0.29)


    args = parser.parse_args()

    return args
# living room

start_position = [0, 0, 0]
box_position = [-0.7047239603180638, -0.4771467885889781, -1.5046124674942505]


if __name__ == '__main__':
    # start setup
    args = get_config()
    rospy.init_node('hsr_classify_with_cnn')
    print(1, end=' ')
    vision_controller = VisionController()
    print(2, end=' ')
    classify_controller = ClassifyController()
    print(3, end=' ')
    joint_controller = JointController()
    print(4, end=' ')
    # hsr python api
    robot = Robot()
    tts = robot.try_get('default_tts')
    tts.language = tts.ENGLISH
    print(5, end=' ')
    omni_base = robot.try_get('omni_base')
    print(6, end=' ')
    whole_body = robot.try_get('whole_body')
    print(7, end=' ')
    gripper = robot.try_get('gripper')
    print('[END] Load HSR API')

    # train setup
    x_offset = args.default_x_offset
    z_offset = args.default_z_offset

    while True:
        # 0. initial joint
        joint_controller.move_to_joint_positions('start_position')
        joint_controller.grasp(1.0)

        # 1. start position
        print('1. go to start position')
        omni_base.go_abs(start_position[0], start_position[1], start_position[2], 100)
        # hand_down
        joint_controller.move_to_joint_positions('search_position')
        rospy.sleep(10)

        # detect object and decide object to grasp
        print('2. detect & decide object to grasp')
        pc, _box_info = vision_controller.get_object_pc()    # calc x, y, distance to object
        pc = pc.copy()

        if pc is None:  # any more object in scene
            joint_controller.move_to_joint_positions('start_position')
            tts.say('No more Object. Bye Bye')
            sys.exit()

        pc[0], pc[1], pc[2] = round(pc[0], 3), round(pc[1], 3), round(pc[2], 3)
        head_tilt_joint = 0.87

        front_dist = (pc[2] * math.sin((math.pi / 2) - head_tilt_joint)) - (
                    pc[1] * math.cos((math.pi / 2) - head_tilt_joint)) - z_offset
        print('3. Go to : ', 'y', -pc[0] + x_offset, 'x', front_dist)  # x, z

        omni_base.go_rel(0, round(-pc[0] + x_offset,4), 0, 100)

        joint_controller.move_to_joint_positions('go_to_position')
        # go front
        whole_body.linear_weight = 0.1
        whole_body.move_end_effector_pose(geometry.pose(z=front_dist), 'hand_palm_link')

        # next level pose
        if _box_info[2] > _box_info[3]:  # width is larger
            joint_controller.move_to_joint_positions('pick_position_for_vertical')
        else:
            joint_controller.move_to_joint_positions('pick_position')
        joint_controller.grasp(1.2)

        # go down
        whole_body.move_end_effector_pose(geometry.pose(z=0.40), 'hand_palm_link')

        print('4. Pick the Object')
        # pick
        gripper.apply_force(0.5)

        # whole_body.move_end_effector_pose(geometry.pose(z=-0.2), 'hand_palm_link')
        joint_controller.move_to_joint_positions('go_to_position')
        rospy.sleep(2)

        print('5. Return to train position')

        omni_base.go_abs(0,0,-1.57,100)
        classify_controller.reset_class_label_list()

        # scan

        joint_controller.move_to_learn_object(whole_body)

        object_label = classify_controller.get_result()
        object_list = ['can', 'plastic', 'paper']
        print('This object is '+object_list[object_label])
        tts.say('This object is '+object_list[object_label])

        print('5. Go to the place box')
        omni_base.go_abs(box_position[0], box_position[1], box_position[2], 100)
        joint_controller.move_to_joint_positions('place_position')
        joint_controller.move_to_joint_positions('place_position')

        
        # place
        rospy.sleep(3)
        print('6. place the object')
        joint_controller.grasp(1.0)
        rospy.sleep(2)




