import urx
import time
from basis import bs, bsr
from Dependencies.urx_custom.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
robot = urx.Robot('192.168.1.66')
gripper = Robotiq_Two_Finger_Gripper(robot)

def pp(x1,y1,x2,y2):#relative=False, pick(x1,y1), place(x2,y2)
    depth=-0.08
    acc=0.1
    vel=0.1
    gripper.open_gripper()
    robot.movel(bs(0,y1,0,0,0,0), acc=acc, vel=vel, relative=False)  # move y1
    robot.movel(bs(x1, y1, 0, 0, 0, 0), acc=acc, vel=vel, relative=False)  # move x1
    robot.movel(bs(x1, y1,depth, 0, 0, 0), acc=acc, vel=vel, relative=False)  # down
    gripper.close_gripper()
    time.sleep(0.5)
    robot.movel(bs(x1, y1, 0, 0, 0, 0), acc=acc, vel=vel, relative=False)  # up
    robot.movel(bs(x1, y2, 0, 0, 0, 0), acc=acc, vel=vel, relative=False)  # move y2
    robot.movel(bs(x2, y2, 0, 0, 0, 0), acc=acc, vel=vel, relative=False)  # move x2
    robot.movel(bs(x2, y2,depth, 0, 0, 0), acc=acc, vel=vel, relative=False)  # down
    gripper.open_gripper()
    robot.movel(bs(x2, y2, 0, 0, 0, 0), acc=acc, vel=vel, relative=False)  # up
    robot.movel(bs(x2,0,0,0,0,0), acc=acc, vel=vel, relative=False) #move (x2,0,0)
    robot.movel(bs(0,0,0,0,0,0), acc=acc, vel=vel, relative=False) #move (0,0,0)
    gripper.close_gripper()

def ppr(x1,y1,x2,y2):#relative=True, pick(+x1,+y1), place(+x1+x2, +y1+y2)
    depth = -0.08
    acc = 0.1
    vel = 0.1
    gripper.open_gripper()
    robot.movel(bsr(0,y1,0,0,0,0), acc=acc, vel=vel, relative=True)  # move y1
    robot.movel(bsr(x1, 0, 0, 0, 0, 0), acc=acc, vel=vel, relative=True)  # move x1
    robot.movel(bsr(0, 0,depth, 0, 0, 0), acc=acc, vel=vel, relative=True)  # down
    gripper.close_gripper()
    time.sleep(0.5)
    robot.movel(bsr(0, 0,-depth, 0, 0, 0), acc=acc, vel=vel, relative=True)  # up
    robot.movel(bsr(0, y2, 0, 0, 0, 0), acc=acc, vel=vel, relative=True)  # move y2
    robot.movel(bsr(x2, 0, 0, 0, 0, 0), acc=acc, vel=vel, relative=True)  # move x2
    robot.movel(bsr(0, 0,depth, 0, 0, 0), acc=acc, vel=vel, relative=True)  # down
    gripper.open_gripper()
    robot.movel(bsr(0, 0,-depth, 0, 0, 0), acc=acc, vel=vel, relative=True)  # up
    gripper.close_gripper()
