import urx
import time
from Dependencies.urx_custom.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
robot = urx.Robot('192.168.1.66')
gripper = Robotiq_Two_Finger_Gripper(robot)

time.sleep(3)
gripper.open_gripper()
robot.movel([0.4,0,0.5,2,1.5,4],acc=0.1,vel=0.1,relative=False)#default
robot.movel([0.2,0,0,0,0,0],acc=0.1,vel=0.1,relative=True)#front
robot.movel([0,0.2,-0.2,0,0,0],acc=0.1,vel=0.1,relative=True)#right
robot.movel([0,0.08,0.08,0,0,0],acc=0.1,vel=0.1,relative=True)#down
gripper.close_gripper()
robot.movel([0,-0.08,-0.08,0,0,0],acc=0.1,vel=0.1,relative=True)#up
robot.movel([-0.2,0,0,0,0,0],acc=0.1,vel=0.1,relative=True)#back
robot.movel([0,0.1,-0.1,0,0,0],acc=0.1,vel=0.1,relative=True)#right
robot.movel([0,0.07,0.07,0,0,0],acc=0.1,vel=0.1,relative=True)#down
gripper.open_gripper()
robot.movel([0,-0.08,-0.08,0,0,0],acc=0.1,vel=0.1,relative=True)#up
robot.movel([0.4,0,0.5,2,1.5,4],acc=0.1,vel=0.1,relative=False)


#robot.movel([x,y,z,rx,ry,rz],acc,vel,relative= True)
#robot.movej([3.4582676887512207, -0.5313866895488282, 1.344257656727926, -3.3320442638792933, -2.9878888765918177, 3.3432490825653076]
#,0,1,vel,relative= False)'


#gripper.close_gripper()

#gripper.open_gripper()

#robot.close()

#robot.movej([3.4582676887512207, -0.5313866895488282, 1.344257656727926, -3.3320442638792933, -2.9878888765918177, 3.3432490825653076])
#obs -> robot_position->data->init_position->

