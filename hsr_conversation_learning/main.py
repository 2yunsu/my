# -*- encoding: UTF-8 -*-
import qi
import argparse
import sys
import time
from naoqi import ALProxy
import numpy as np
from PIL import Image

aas_configuration = {"bodyLanguageMode": "contextual"}


def get_rgb(video, rgb_top, save_path=None):
    msg = video.getImageRemote(rgb_top)
    w = msg[0]
    h = msg[1]
    data = msg[6]
    ba = str(bytearray(data))

    im = Image.frombytes("RGB", (w, h), ba)

    return im.save("image/rgb.png", "PNG")

def main(session):
    tts=session.service("ALTextToSpeech")
    tts.say("3,2,1")
    video = session.service("ALVideoDevice")
    # photoCaptureProxy = session.service("ALPhotoCapture")
    rgb_top = video.subscribe('rgb_t', 2, 11, 20)
    get_rgb(video, rgb_top, save_path='image')
    video.unsubscribe(rgb_top)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.212",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)