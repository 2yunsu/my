import qi
import argparse
import sys
import time


def asr_callback(msg):
    # Threshold
    print(msg[0], ' is recognized. ', msg[1])
    if msg[1] > 0.5:
        print(msg[0], msg[1], " is returned")
        _type = 'speech'
        _word = msg[0]

def main(session):
    """
    This example uses the ALSpeechRecognition module.
    """
    audio_device = session.service("ALAudioDevice")
    while True:
        audio_device.setOutputVolume(3)
        # Get the service ALSpeechRecognition.

        asr_service = session.service("ALSpeechRecognition")
        memory = session.service("ALMemory")
        asr_service.pause(True)
        asr_service.setLanguage("English")

        # Example: Adds "yes", "no" and "please" to the vocabulary (without wordspotting)
        vocabulary = ["yes", "no", "please", 'bye']
        asr_service.setVocabulary(vocabulary, False)

        # Start the speech recognition engine with user Test_ASR
        asr_service.subscribe("asr")
        print 'Speech recognition engine started'

        asr_mem_sub = memory.subscriber("WordRecognized")
        asr_mem_sub.signal.connect(asr_callback)

        asr_service.unsubscribe("asr")
        audio_device.setOutputVolume(100)


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