# -*- encoding: UTF-8 -*-

import qi
import argparse
import sys
import math
import subprocess
import time

import touch
from naoqi import ALProxy
from threading import Thread

sys.path.insert(0, './motion')
import entertain
# import saju

FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
DEFAULT_VOLUME = 60

# jesnk touch
TOUCH_LIST = {}
TOUCH_LIST['RIGHT_SIDE'] = {"x": [FRAME_WIDTH / 2, FRAME_WIDTH], "y": [0, FRAME_HEIGHT], 'name': "RIGHT_SIDE"}
TOUCH_LIST['LEFT_SIDE'] = {"x": [0, FRAME_WIDTH], "y": [0, FRAME_HEIGHT], 'name': "LEFT_SIDE"}

TOUCH_LIST['JESNK_SIDE'] = {"x": [0, 200], "y": [0, 200], 'name': "JESNK_SIDE"}

TOUCH_LIST['BUTTON_LEFT'] = {"x": [75, 600], "y": [233, 593], 'name': "BUTTON_LEFT"}
TOUCH_LIST['BUTTON_RIGHT'] = {"x": [669, 1192], "y": [227, 598], 'name': "BUTTON_RIGHT"}
TOUCH_LIST['BUTTON_MIDDLE_DOWN'] = {"x": [485, 800], "y": [632, 705], 'name': "BUTTON_MIDDLE_DOWN"}
TOUCH_LIST['BUTTON_RIGHT_DOWN'] = {"x": [930, 1156], "y": [641, 707], 'name': "BUTTON_RIGHT_DOWN"}
TOUCH_LIST['BUTTON_LEFT_DOWN'] = {"x": [150, 390], "y": [621, 707], 'name': "BUTTON_LEFT_DOWN"}

scene_data = {}
scene_data['init'] = ['init', ['RIGHT_SIDE', 'LEFT_SIDE'], ['bye', 'next', 'first']]
scene_data['1'] = ['1', ['RIGHT_SIDE', 'LEFT_SIDE'], ['bye', 'next', 'first']]
scene_data['exit'] = ['exit', [], []]

scene_data['home'] = ['home', ['BUTTON_MIDDLE_DOWN', 'JESNK_SIDE'], ['start', 'next', 'pepper']]

scene_data['first_menu'] = ['first_menu', \
                            ['JESNK_SIDE', 'BUTTON_RIGHT', 'BUTTON_LEFT', \
                             'BUTTON_MIDDLE_DOWN', 'BUTTON_RIGHT_DOWN'], ['bye', 'next', 'first']]

scene_data['tour'] = ['tour', \
                      ['JESNK_SIDE', 'BUTTON_RIGHT', 'BUTTON_LEFT', \
                       'BUTTON_LEFT_DOWN', 'BUTTON_MIDDLE_DOWN', 'BUTTON_RIGHT_DOWN'], \
                      ['bye', 'next', 'first']]

scene_data['entertain'] = ['entertain', \
                           ['JESNK_SIDE', 'BUTTON_RIGHT', 'BUTTON_LEFT', \
                            'BUTTON_LEFT_DOWN', 'BUTTON_MIDDLE_DOWN', 'BUTTON_RIGHT_DOWN'], \
                           ['bye', 'next', 'first']]

scene_data['entertain2'] = ['entertain2', \
                            ['JESNK_SIDE', 'BUTTON_RIGHT', 'BUTTON_LEFT', \
                             'BUTTON_LEFT_DOWN', 'BUTTON_MIDDLE_DOWN', 'BUTTON_RIGHT_DOWN'], \
                            ['bye', 'next', 'first']]

scene_data['tour_hsr1'] = ['tour_hsr1', \
                           ['JESNK_SIDE', 'BUTTON_RIGHT', 'BUTTON_LEFT', \
                            'BUTTON_LEFT_DOWN', 'BUTTON_MIDDLE_DOWN', 'BUTTON_RIGHT_DOWN'], \
                           ['bye', 'next', 'first']]

scene_data['tour_hsr2'] = ['tour_hsr2', \
                           ['JESNK_SIDE', 'BUTTON_RIGHT', 'BUTTON_LEFT', \
                            'BUTTON_LEFT_DOWN', 'BUTTON_MIDDLE_DOWN', 'BUTTON_RIGHT_DOWN'], \
                           ['bye', 'next', 'first']]

signalID = 0


def touch_callback(x, y):
    print(" coordinate x : ", x, " y : ", y)
    print(signalID)


class Monitor_input:
    def __init__(self, srv, touch_list=[], word_list=[]):
        self.target_touch_list = touch_list
        self.target_word_list = word_list
        self.srv = srv
        self.tabletService = srv['tablet']
        self.signalID = srv['tablet'].onTouchDown.connect(self.touch_callback)
        self.touched_position = None
        self.exit_flag = False
        self.ret = {}
        self.memory = srv['memory']
        self.asr = srv['asr']
        self.asr.pause(True)
        #self.asr.setLanguage("Korean")
        self.asr.setLanguage("English")

        self.debug_mode = False
        self.debug_touch_count = 0
        self.debug_touch_coordinate = []
        try:
            self.asr.unsubscribe("asr")
        except:
            pass
        self.asr.pause(True)

    def check_valid_touch(self):
        for i in self.target_touch_list:
            if self.touch_x > TOUCH_LIST[i]['x'][0] and self.touch_x < TOUCH_LIST[i]['x'][1]:
                if self.touch_y > TOUCH_LIST[i]['y'][0] and self.touch_y < TOUCH_LIST[i]['y'][1]:
                    self.ret['touch_position'] = i
                    return True
        return False

    def touch_callback(self, x, y):
        print(self.debug_mode)
        if self.debug_mode:
            self.debug_touch_count += 1
            self.debug_touch_coordinate.append([x, y])
            print("x : ", x, " y : ", y)
            if self.debug_touch_count == 4:
                self.debug_mode = False
                self.debug_touch_count = 0
                print("test")
                xs = [x[0] for x in self.debug_touch_coordinate]
                xs.sort()
                ys = [x[1] for x in self.debug_touch_coordinate]
                ys.sort()
                print("X range : ", xs[0], "-", xs[-1])
                print("Y range : ", ys[0], "-", ys[-1])
                print("Touch_debug_mode Finished")
                self.debug_touch_coordinate = []
                return
            return

        self.touch_x = x
        self.touch_y = y
        if (self.check_valid_touch()):
            self.ret['type'] = 'touch'
            self.ret['x'] = x
            self.ret['y'] = y
            self.exit_flag = True

        print("class_ x ", x, " y ", y)

    def asr_callback(self, msg):
        # Threshold
        print(msg[0], ' is recognized. ', msg[1])
        if msg[1] > 0.5:
            print(msg[0], msg[1], " is returned")
            self.ret['type'] = 'speech'
            self.ret['word'] = msg[0]
            self.exit_flag = True

    def wait_for_get_input(self):
        self.asr.setVocabulary(self.target_word_list, False)
        print("Staring wait")
        self.srv['audio_device'].setOutputVolume(3)
        self.asr.subscribe('asr')
        asr_mem_sub = self.memory.subscriber("WordRecognized")
        asr_mem_sub.signal.connect(self.asr_callback)
        while not self.exit_flag:
            time.sleep(0.01)

        self.asr.unsubscribe('asr')
        self.srv['audio_device'].setOutputVolume(100)
        self.exit_flag = False
        return self.ret

    def set_target_touch_list(self, touch_list):
        self.target_touch_list = touch_list

    def set_target_word_list(self, word_list):
        self.target_word_list = word_list

    def __del__(self):
        self.tabletService.onTouchDown.disconnect(self.touch_callback)
        self.asr.unsubscribe("ASR")


def get_html_address(file_name):
    name = file_name
    if len(name) > 5 and name[-5:] == '.html':
        name = name[:-5]
    return "http://198.18.0.1/apps/bi-html/" + name + '.html'


def transition(srv, scene, input_ret):
    global monitor_input
    # return value : scene name, available touch, avail word
    print("Trainsition mode")
    print(scene, input_ret)

    if scene == 'home':
        if input_ret['type'] == 'touch':
            if input_ret['touch_position'] == 'BUTTON_MIDDLE_DOWN':
                next_scene = 'first_menu'

                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("next")

                return scene_data[next_scene]

            # jesnk : test

            if input_ret['touch_position'] == 'JESNK_SIDE':
                file_path = "/opt/aldebaran/www/apps/bi-sound/background.mp3"
                # srv['tts'].post.say('yes')
                player = ALProxy("ALAudioPlayer")
                player.post.playFileFromPosition(file_path, 120)

                # file_id = srv['audio_player'].loadFile("/opt/aldebaran/www/apps/bi-sound/background.mp3")
                # srv['audio_player'].playFileFromPosition(file_path,120)

                # srv['audio_player'].setVolume(file_id,0.3)


        elif input_ret['type'] == 'speech':
            if input_ret['word'] == 'start':
                next_scene = 'first_menu'
                srv['aas'].say("Hello, Nice to meet you!", aas_configuration)
                srv['tablet'].showWebview(get_html_address(next_scene))

                return scene_data[next_scene]
            if input_ret['word'] == 'Hello':
                next_scene = 'home'
                srv['aas'].say("Hello Sir!", aas_configuration)
                return scene_data[next_scene]

            if input_ret['word'] == 'pepper':
                next_scene = 'home'
                srv['aas'].say("Yep! Hello?!", aas_configuration)
                return scene_data[next_scene]

    if scene == 'first_menu':
        if input_ret['type'] == 'touch':
            if input_ret['touch_position'] == 'JESNK_SIDE':
                next_scene = 'first_menu'

                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("Debug")
                monitor_input.debug_mode = True

                while monitor_input.debug_mode:
                    time.sleep(0.01)
                srv['tts'].say("Debug finished")

                return scene_data[next_scene]

            if input_ret['touch_position'] == 'BUTTON_LEFT':
                next_scene = 'tour'

                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("next")

                return scene_data[next_scene]

            if input_ret['touch_position'] == 'BUTTON_RIGHT':
                next_scene = 'entertain'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("next")
                return scene_data[next_scene]

            if input_ret['touch_position'] == 'BUTTON_MIDDLE_DOWN':
                next_scene = 'home'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say("next")
                return scene_data[next_scene]
            if input_ret['touch_position'] == 'BUTTON_RIGHT_DOWN':
                next_scene = scene
                srv['tts'].setParameter("defaultVoiceSpeed", 100)
                srv['aas'].say(
                    "Are you curious about me? I am Pepper. It is a humanoid robot made by Softbank, and can use artificial intelligence. It is characterized by a cute appearance, and is introduced in various fields such as finance, bookstore, medical care, and distribution fields in Korea")
                srv['tts'].setParameter("defaultVoiceSpeed", 70)
                return scene_data[next_scene]
        elif input_ret['type'] == 'speech':
            if input_ret['word'] == 'bye':
                return scene_data['exit']

            if input_ret['word'] == 'first':
                next_scene = 'home'
                srv['tablet'].showWebview(get_html_address(next_scene))
                return scene_data[next_scene]

            if input_ret['word'] == 'who':
                next_scene = 'first_menu'
                srv['tablet'].showWebview(get_html_address(next_scene))

                return scene_data[next_scene]

    if scene == 'tour':
        if input_ret['type'] == 'touch':
            if input_ret['touch_position'] == 'BUTTON_RIGHT':
                next_scene = 'tour_hsr1'
                srv['tts'].setParameter("defaultVoiceSpeed", 100)
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say("Let me explain the robots in our lab. First, HSR, a human helper robot, is a mobile operation robot.", aas_configuration)

                next_scene = 'tour_hsr2'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say("It is about 1 meter tall and is a versatile robot that can recognize objects through various cameras and pick them up with a gripper. But is it ugly than me?",
                               aas_configuration)

                next_scene = 'tour_blitz'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say(
                    "The next robot, Blitz. It is a robot made by combining a base robot, which is specialized in moving objects, and a UR5 robot that picks up objects. In addition, it is a mobile operation robot that is equipped with sound and camera sensors, capable of recognizing objects and gripping them with a gripper.",
                    aas_configuration)

                next_scene = 'tour_pepper1'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say(
                    "The last robot to be introduced is me, Pepper. I am a humanoid robot made by Softbank, and I can use artificial intelligence.",aas_configuration)

                next_scene = 'tour_pepper2'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say(
                    "I have a cute appearance, and has been introduced in various fields such as finance, bookstores, medical care, and distribution fields in Korea. In addition, it is used as a standard robot in S, S, P, L, among the world robot competitions, Robo Cup League.",
                    aas_configuration)

                srv['tts'].setParameter("defaultVoiceSpeed", 70)
                next_scene = 'tour'
                srv['tablet'].showWebview(get_html_address(next_scene))
                return scene_data[next_scene]
            if input_ret['touch_position'] == 'BUTTON_LEFT':
                next_scene = 'tour_ourlab1'
                srv['tts'].setParameter("defaultVoiceSpeed", 100)
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say(
                    "Let me introduce our lab. Our bio-intelligence lab is conducting the following studies. First, we are conducting interdisciplinary research in various fields such as artificial intelligence, psychology, and cognitive science to develop human-level artificial intelligence such as Baby Mind and VTT. We are also actively conducting research on robots on various platforms, such as home robots that work with humans and Robocup, a world robot competition.",
                    aas_configuration)

                next_scene = 'tour_ourlab2'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say("If you have any other questions or inquiries, please refer to the following website or contact us.", aas_configuration)

                srv['tts'].setParameter("defaultVoiceSpeed", 70)
                next_scene = 'tour'
                srv['tablet'].showWebview(get_html_address(next_scene))
                return scene_data[next_scene]
            if input_ret['touch_position'] == 'BUTTON_MIDDLE_DOWN':
                next_scene = 'home'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['aas'].say("To the inital screen", aas_configuration)
                return scene_data[next_scene]
            if input_ret['touch_position'] == 'BUTTON_LEFT_DOWN':
                next_scene = 'first_menu'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("previous")
                return scene_data[next_scene]
            if input_ret['touch_position'] == 'BUTTON_RIGHT_DOWN':
                next_scene = scene
                srv['tts'].setParameter("defaultVoiceSpeed", 110)
                srv['aas'].say(
                    "Are you curious about me? I am Pepper. It is a humanoid robot made by Softbank, and can use artificial intelligence. It is characterized by a cute appearance, and is introduced in various fields such as finance, bookstore, medical care, and distribution fields in Korea",
                    aas_configuration)
                srv['tts'].setParameter("defaultVoiceSpeed", 70)
                return scene_data[next_scene]

    if scene == 'entertain':
        if input_ret['type'] == 'touch':
            if input_ret['touch_position'] == 'BUTTON_MIDDLE_DOWN':
                next_scene = 'home'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("To the inital screen")
                return scene_data[next_scene]
            if input_ret['touch_position'] == 'BUTTON_LEFT_DOWN':
                next_scene = 'first_menu'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("previous")
                return scene_data[next_scene]

            if input_ret['touch_position'] == 'BUTTON_LEFT':
                file_path = "/opt/aldebaran/www/apps/bi-sound/elephant.ogg"
                # srv['tts'].post.say('yes')
                player = ALProxy("ALAudioPlayer", PEPPER_IP, 9559)
                player.post.playFileFromPosition(file_path, 0)
                entertain.elephant(srv)
                player.post.stopAll()
                pass
            if input_ret['touch_position'] == 'BUTTON_RIGHT':
                file_path = "/opt/aldebaran/www/apps/bi-sound/UrbanStreet.mp3"
                player = ALProxy("ALAudioPlayer", PEPPER_IP, 9559)
                player.post.playFileFromPosition(file_path, 0)
                entertain.disco(srv)
                player.post.stopAll()
                pass
            if input_ret['touch_position'] == 'BUTTON_RIGHT_DOWN':
                next_scene = 'entertain2'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("next")
                return scene_data[next_scene]

    if scene == 'entertain2':
        if input_ret['type'] == 'touch':
            if input_ret['touch_position'] == 'BUTTON_MIDDLE_DOWN':
                next_scene = 'home'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("To the inital screen")
                return scene_data[next_scene]
            if input_ret['touch_position'] == 'BUTTON_LEFT_DOWN':
                next_scene = 'first_menu'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("previous")
                return scene_data[next_scene]

            if input_ret['touch_position'] == 'BUTTON_LEFT':
                srv['aas'].say("Please tell me your date of birth in English.", aas_configuration)
                text = saju.main(srv)
                srv['ass'].say(text, aas_configuration)
            if input_ret['touch_position'] == 'BUTTON_RIGHT':
                pass

            if input_ret['touch_position'] == 'BUTTON_RIGHT_DOWN':
                next_scene = 'entertain'
                srv['tablet'].showWebview(get_html_address(next_scene))
                srv['tts'].say("next")
                return scene_data[next_scene]


# jesnk 1


monitor_input = None

aas_configuration = {"bodyLanguageMode": "contextual"}


def main(session):
    # jesnk main
    print("Hello")
    srv = {}
    srv['tablet'] = session.service("ALTabletService")
    srv['memory'] = session.service("ALMemory")
    srv['motion'] = session.service("ALMotion")
    srv['asr'] = session.service("ALSpeechRecognition")
    srv['tts'] = session.service("ALTextToSpeech")
    srv['aas'] = session.service("ALAnimatedSpeech")
    srv['audio_device'] = session.service("ALAudioDevice")

    srv['tts'].setVolume(0.1)
    srv['tts'].setParameter("defaultVoiceSpeed", 70)
    srv['audio_player'] = session.service("ALAudioPlayer")

    # Present Inital Page
    srv['tablet'].enableWifi()
    srv['tablet'].setOnTouchWebviewScaleFactor(1)
    srv['tablet'].showWebview('http://198.18.0.1/apps/bi-html/home.html')
    # Valid Input condition setting
    global monitor_input
    monitor_input = Monitor_input(srv)

    init_scene = 'home'
    scene_name, valid_touch_list, valid_word_list = \
        scene_data[init_scene][0], scene_data[init_scene][1], scene_data[init_scene][2]
    print(scene_name, valid_touch_list, valid_word_list)
    monitor_input.set_target_touch_list(valid_touch_list)
    monitor_input.set_target_word_list(valid_word_list)

    while (True):

        input_ret = monitor_input.wait_for_get_input()
        ret = transition(srv, scene_name, input_ret)
        if ret == None:
            continue
        print(ret)
        scene_name, valid_touch_list, valid_word_list = ret[0], ret[1], ret[2]
        monitor_input.set_target_touch_list(valid_touch_list)
        monitor_input.set_target_word_list(valid_word_list)
        if scene_name == 'exit':
            break

    print("passed 2")
    # global signalID
    # signalID = tabletService.onTouchDown.connect(touch_callback)

    srv['tablet'].hideWebview()
    print("Finished")


PEPPER_IP = '192.168.1.212'
if __name__ == "__main__":

    print("Hello")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default=PEPPER_IP,
                        help="Robot IP address. On robot or Local Naoqi: use '192.168.1.188'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    print("Hello")

    args = parser.parse_args()
    session = qi.Session()
    print("Hello")
    try:
        session.connect("tcp://" + PEPPER_IP + ":" + str(args.port))
	print("connection complete")
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
                                                                                              "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    main(session)
