import math
import sys

from geometry_msgs.msg import WrenchStamped
import rospy

import time

from std_msgs.msg import Int16
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

from collections import deque

import torch
import cv2


import matplotlib.pyplot as plt
import numpy as np

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')


_CONNECTION_TIMEOUT = 10.0




class VisionController(object):
    def __init__(self, now, maxlen):
        self.rgb_img = None
        self.depth_img = None
        self.hand_img = None

        self.bridge = CvBridge()
        self.starttime = now

        self.hand_queue = deque(maxlen=maxlen)
        self.depth_queue = deque(maxlen=maxlen)

        rgb_topic = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        head_depth_topic = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'

        # self._rgb_sub = rospy.Subscriber(rgb_topic, Image, self._rgb_callback)
        self._head_sub = rospy.Subscriber(rgb_topic, Image, self._head_callback)
        self._depth_sub = rospy.Subscriber(head_depth_topic, Image, self._depth_callback)
        try:
            # rospy.wait_for_message(rgb_topic, CompressedImage, timeout=_CONNECTION_TIMEOUT)
            rospy.wait_for_message(rgb_topic, Image, timeout=_CONNECTION_TIMEOUT)
            rospy.wait_for_message(head_depth_topic, Image, timeout=_CONNECTION_TIMEOUT)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)


    def _head_callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv_image = cv2.resize(cv_image, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)

        self.hand_queue.append(cv_image)

    def _depth_callback(self, data):
        depth_img = self.bridge.imgmsg_to_cv2(data,"32FC1")
        depth_img = cv2.resize(depth_img, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
        self.depth_queue.append(depth_img)



def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1

def norm_vec(v, range_in=None, range_out=None):
    if range_out is None:
        range_out = [-1, 1]
    if range_in is None:
        range_in = [torch.min(v), torch.max(v)]

    r_out = range_out[1] - range_out[0]
    r_in = range_in[1] - range_in[0]
    v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
    return v


def get_config():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--device_id', type=int, default=0, help='device id(default : 0)')
    parser.add_argument('--n_features', type=int, default=1664, help='number of features')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--save_file_path', type=str, default="cnn_unseen_learning.pt")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    from CNN import Net
    args = get_config()

    rospy.init_node('cnn_realtime_tester')
    grasp_pub = rospy.Publisher('/unseen_cnn_learning/classify', Int16, queue_size=10)
    now = time.time()

    maxlen = args.batch_size

    # for data stream
    vision_controller = VisionController(now, maxlen)

    # LOAD MODEL
    model = Net()
    model = model.to(args.device_id)
    print(model)

    model.load_state_dict(torch.load(args.save_file_path))
    model.eval()
    rospy.sleep(3)


    size = 300
    x_vec = np.linspace(0, 1, size + 1)[0:-1]
    y_vec = np.zeros(len(x_vec))
    line1 = []

    with torch.no_grad():
        while not rospy.is_shutdown():
            hand_q = vision_controller.hand_queue
            r = torch.FloatTensor(hand_q).view(-1, 1, 3, 128, 128).cuda(args.device_id)

            test_output = model(r)
            pred = test_output.max(1, keepdim=True)[1]
            pred = pred.cpu()
            pred = np.squeeze(pred, axis=1)
            new_val = np.array(pred)
            y_vec[-args.batch_size:] = new_val
            line1 = live_plotter(x_vec,y_vec,line1)
            y_vec = np.append(y_vec[args.batch_size:], [0.0 for i in range(args.batch_size)])
            grasp_pub.publish(int(new_val))

