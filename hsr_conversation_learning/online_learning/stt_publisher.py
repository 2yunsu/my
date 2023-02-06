
import rospy
from std_msgs.msg import Int16
# silero imports
import os
from os.path import exists
import torch
import random
import time
from glob import glob
from omegaconf import OmegaConf
from silero_stt.src.silero.utils import (init_jit_model,
                       split_into_batches,
                       read_batch,
                       prepare_model_input)
import pyaudio
import wave

# stt model setup
models = OmegaConf.load('silero_stt/models.yml')  # all available models are listed in the yml file
print(list(models.stt_models.keys()),
      list(models.stt_models.en.keys()),
      list(models.stt_models.en.latest.keys()),
      models.stt_models.en.latest.jit)
device = torch.device('cpu')  # you can use any pytorch device
model, decoder = init_jit_model(models.stt_models.en.latest.jit, device=device)

# voice record setup
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "audio/file.wav"
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

class ListenController(object):
    def __init__(self):
        self.start_num = -1
        self.label_list = ['can', 'plastic', 'paper', 'yes', 'no', ]
        self._stt_sub = rospy.Subscriber('stt_start', Int16, self._listen_callback)

    def _listen_callback(self, data):
        self.start_num = data.data


if __name__ == '__main__':
    try:
        rospy.init_node('stt_publisher')
        pub = rospy.Publisher('stt_hsr', Int16, queue_size=10)
        listen_controller = ListenController()
        print('ready')
        while not rospy.is_shutdown():
            if listen_controller.start_num != 1:
                continue
            print("recording...")
            
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
                if i % 50 == 0:
                    print(i//50)
            print()
            
            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            
            test_files = glob('audio/*.wav')  # replace with your data
            batches = split_into_batches(test_files, batch_size=10)
    
            # transcribe a set of files
            input = prepare_model_input(read_batch(random.sample(batches, k=1)[0]),
                                        device=device)
            output = model(input)
            for example in output:
                print(decoder(example.cpu()))
                sentence = decoder(example.cpu())
            print(sentence)
            if("can" in sentence.split(" ")):
                print('can', 0)
                pub.publish(0)
            elif("plastic" in sentence.split(" ")):
                print('plastic', 1)
                pub.publish(1)
            elif("paper" in sentence.split(" ")):
                print('paper', 2)
                pub.publish(2)
            elif ("yes" in sentence.split(" ")):
                print('yes', 3)
                pub.publish(3)
            elif ("no" in sentence.split(" ")):
                print('no', 4)
                pub.publish(4)
            else:
                pass
                # pub.publish(5)
                
    finally:
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
