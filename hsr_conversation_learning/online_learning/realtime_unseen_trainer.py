import sys
import rospy
import time
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Int16
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from collections import deque, Counter
import random
import cv2

import torch
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')


_CONNECTION_TIMEOUT = 10.0


class Trainer(object):
    def __init__(self, now, maxlen, args):
        self.args = args
        self.rgb_img = None
        self.depth_img = None
        self.hand_img = None
        self.bridge = CvBridge()

        self._force_data_x = 0.0
        self._force_data_y = 0.0
        self._force_data_z = 0.0
        self.init_force = None


        self.starttime = now
        self.model = None
        self.train_mode = False
        self.first_train = True
        self.item_size = 0

        self.label = torch.Tensor([0]).to(args.device_id)
        self.rgb_queue = deque(maxlen=maxlen)

        self.memory = deque(maxlen=args.q_size)
        self.train_idx = 0
        self.train_time = 0
        self.train_df = pd.DataFrame([{'id': self.train_idx, 'time': now, 'abs_time': 0, 'train_time': 0,'train_loss' : 0, 'queue_size':0,  'queue_byte':0}])

        self.eval_memory = deque(maxlen=args.q_size)
        self.eval_idx = 0
        self.eval_time = 0
        self.eval_df = pd.DataFrame([{'id': self.eval_idx, 'time': now, 'abs_time': 0, 'train_time': 0,
                                       'train_loss': 0, 'queue_size': 0, 'queue_byte': 0}])

        rgb_topic = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'

        self._head_sub = rospy.Subscriber(rgb_topic, Image, self._head_callback)

        self.train_mode_sub = rospy.Subscriber('/unseen_cnn_learning/start', Int16, self.train_mode_callback)
        self.train_label_sub = rospy.Subscriber('/unseen_cnn_learning/label', Int16, self.train_label_callback)



    def _head_callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv_image = cv2.resize(cv_image, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
        self.rgb_queue.append(cv_image)

    def _depth_callback(self, data):
        depth_img = self.bridge.imgmsg_to_cv2(data,"32FC1")
        depth_img = cv2.resize(depth_img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
        self.depth_queue.append(depth_img)



    def set_model(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.00005, betas=[0.9, 0.999], eps=1e-08,
                               weight_decay=0, amsgrad=False)

    def train_mode_callback(self, data):
        if data.data == 1:
            print('go train mode')
            self.train_mode = True
        else: #
            print('go eval mode')
            self.train_mode = False
            self.model.eval()
            self.train_df.to_csv('cnn_train_log.csv')
            self.eval_df.to_csv('cnn_eval_log.csv')
            torch.save(self.model.state_dict(), self.args.save_file_path)
            if data.data == 2:
                print('End train')


    def train_label_callback(self, data):
        if data.data == 0:
            print('Label : 0')
            self.label = torch.LongTensor([0]).cuda(self.args.device_id)
        elif data.data == 1:
            print('Label : 1')
            self.label = torch.LongTensor([1]).cuda(self.args.device_id)
        else:
            print('Label : 2')
            self.label = torch.LongTensor([2]).cuda(self.args.device_id)

    def train(self):
        start_train = time.time()
        rgb_q = self.rgb_queue
        _r = torch.FloatTensor(rgb_q).view(-1, 1, 3, 128, 128).cuda(args.device_id)
        label = self.label
        # memorize
        self.memory.append((_r, label))

        # sample the batch
        batch_size = self.args.batch_size
        if len(self.memory) < self.args.batch_size:
            batch_size = len(self.memory)
        batch = random.sample(self.memory, batch_size)
        r, l = zip(*batch)

        r = torch.cat(r)
        l = torch.cat(l)
        if len(Counter(l.cpu().numpy()).items()) == 1:
            return 0


        model.train()
        self.optimizer.zero_grad()

        output = model(r)
        loss = F.cross_entropy(output, l)
        loss.backward()

        self.optimizer.step()
        new_val = np.array(loss.item())
        self.train_idx += 1
        self.train_time += time.time() - start_train
        if self.first_train:
            self.first_train = False

            self.item_size = sys.getsizeof(_r) + sys.getsizeof(label)
        print(loss.item())
        tempdf = pd.DataFrame([{'id': self.train_idx, 'time': time.time()-self.starttime, 'abs_time': time.time(), 'train_time': self.train_time,
                                       'train_loss': loss.item(), 'queue_size': len(self.memory), 'queue_byte': self.item_size * len(self.memory) / (1024.0)}])
        self.train_df = self.train_df.append(tempdf, ignore_index=True)

        return new_val

    def eval_for_train(self):
        rgb_q = self.rgb_queue
        _r = torch.FloatTensor(rgb_q).view(-1, 1, 3, 128, 128).cuda(args.device_id)
        label = self.label
        # memorize
        self.eval_memory.append((_r, label))

        return 0

    def evaluate(self):
        start_eval = time.time()
        # sample the batch
        batch_size = self.args.batch_size
        if len(self.eval_memory) < self.args.batch_size:
            batch_size = len(self.eval_memory)
            return
        batch = random.sample(self.eval_memory, batch_size)
        r, l = zip(*batch)
        r = torch.cat(r)
        l = torch.cat(l)

        model.eval()
        output = model(r)
        loss = F.cross_entropy(output, l)

        new_val = np.array(loss.item())
        self.eval_idx += 1
        self.eval_time += time.time() - start_eval

        tempdf = pd.DataFrame([{'id': self.eval_idx, 'time': time.time() - self.starttime, 'abs_time': time.time(),
                                'eval_time': self.eval_time,
                                'eval_loss': loss.item(), 'queue_size': len(self.eval_memory),
                                'queue_byte': self.item_size * len(self.eval_memory) / (1024.0)}])
        self.eval_df = self.eval_df.append(tempdf, ignore_index=True)

        return new_val

    def test(self):
        rgb_q = self.rgb_queue
        r = torch.FloatTensor(rgb_q).view(-1, 1, 3, 128, 128).cuda(args.device_id)
        test_output = model(r)
        pred = test_output.max(1, keepdim=True)[1]
        pred = pred.cpu()
        pred = np.squeeze(pred, axis=1)
        new_val = np.array(pred)
        return new_val

    def HsrDataset(self, args, hand_q):
        r = torch.FloatTensor(hand_q).view(-1, 1, 3, 128, 128)
        return r.cuda(args.device_id)




def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(6, 3))
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



def get_config():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--device_id', type=int, default=0, help='device id(default : 0)')
    parser.add_argument('--n_features', type=int, default=1664, help='number of features')
    parser.add_argument('--maxlen', type=int, default=1)
    parser.add_argument('--q_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--save_file_path', type=str, default="cnn_unseen_learning.pt")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    from CNN import Net
    args = get_config()

    rospy.init_node('hsr_realtime_trainer')
    now = time.time()
    maxlen = args.maxlen

    # for data stream
    train_controller = Trainer(now, maxlen, args)

    # LOAD MODEL
    model = Net()
    model = model.to(args.device_id)
    train_controller.set_model(model)
    print(model)


    size = 300
    x_vec = np.linspace(0, 1, size + 1)[0:-1]
    y_vec = np.zeros(len(x_vec))
    line1 = []

    x_vec_train = np.linspace(0, 1, size + 1)[0:-1]
    y_vec_train = np.zeros(len(x_vec_train))
    line2 = []

    rospy.sleep(5)
    print('start')
    y_vec_train[-args.maxlen:] = 0
    line2 = live_plotter(x_vec_train, y_vec_train, line2, identifier='train_loss')
    y_vec_train = np.append(y_vec_train[args.maxlen:], [0.0 for i in range(args.maxlen)])


    while not rospy.is_shutdown():
        if train_controller.train_mode == True:
            new_val = train_controller.train()
            # train_controller.evaluate()
            y_vec_train[-args.maxlen:] = new_val
            line2 = live_plotter(x_vec_train, y_vec_train, line2, identifier='train_loss')
            y_vec_train = np.append(y_vec_train[args.maxlen:], [0.0 for i in range(args.maxlen)])
        else:
            new_val = train_controller.test()
            y_vec[-args.maxlen:] = new_val
            line1 = live_plotter(x_vec, y_vec, line1, identifier='grasp detection label')
            y_vec = np.append(y_vec[args.maxlen:], [0.0 for i in range(args.maxlen)])


