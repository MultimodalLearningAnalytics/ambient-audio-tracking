
#!/usr/bin/python

""" PlayAudio: A python script that plays sounds and reports the time of play back to PSI

***************************************************************************************
Developer: Jeffrey Pronk
Github: jeffthestig
Email: j.s.pronk@student.tudelft.nl

Year: 2021

Developed at the request of Yoon Lee as part of the CSE BSc Research Project (CSE3000).
***************************************************************************************

A pythons script meant to play distracting audio during the experiment. When a sound is played, the sound and time of play
will be reported to PSI so in the analysis phase it is clear when what sound is played.
"""

import json
import keyboard
import vlc
import time

from datetime import datetime
from sys import exit
from threading import Thread
from utils.psi_connection import PsiConnection


class PlayAudio:
    """ PlayAudio: Class responsible for playing audio and resporting this to PSI
    """

    def __init__(self):
        # collect setup
        self.collect_settings()
        self.playing = False
        self.sound_id = 0
        self.id = 0

        # init vlc media player
        self.player = vlc.MediaPlayer()
        self.tks = True
        self.pos = 0
        self.t = Thread(target=self.pos_check)
        self.t.start()

        # init socket 
        if(self.online):
            self.conn = PsiConnection(pub_ip=self.PUB_IP, sub_ip=self.SUB_IP, sync=self.SYNC_TIME)


    def collect_settings(self):
        try:
            f = open("./config/player_setup.json", "r")
            js = json.loads(f.read())

            self.BLOCK = js["audio"]["block"]

            self.SUB_IP = js["socket"]["sub_ip"]
            self.PUB_IP = js["socket"]["pub_ip"]
            self.SOUND_PLAYED_TOPIC = js["socket"]["sound_played_topic"]
            self.SYNC_TIME = js["socket"]["sync_time"]

            self.online = js["usage"]["online"]
            self.ids = js["usage"]["ids"]
            self.sounds = js["usage"]["sounds"]
        except Exception as e:
            print("EXCEPTION: " + str(e))
            print("Please make sure the file ./play_setup.json is created and that the file is correct.\r\n" + 
            "The following JSON string should be present in the json file: \r\n{\r\n\t\"audio\": {\r\n\t\t\"block\": <block size>\r\n\t},\r\n\t\"socket\": {\r\n\t\t\"sub_ip\": \"<subscribe ip, connect ip>\",\r\n\t\t\"pub_ip\": \"<publish ip, bind ip>\",\r\n\t\t\"sound_played_topic\": \"<name of sound played report topic>\",\r\n\t\t\"sync_topic\": \"<true or false, if audio should be synced>\"\r\n\t},\r\n\t\"usage\": {\r\n\t\t\"online\": <true if socket should be enabled (should be default), false if socket is disabled: no comms with PSI!>\r\n\t}\r\n}")

            exit()
    
    def console_loop(self):
        try:
            self.id_s = input(">>> Please provide the test identifier: ")
            self.id = int(self.id_s)

            if not self.id_s in self.ids:
                print("ERROR: incorrect id, please check player_setup.json for available ids. ID: {%s}"%id)
                self.close()

            print("\r\nUsing id %s"%self.id_s)
            print("Press 1-9 for the respective sound. Press p for pause/play and press q to exit.\r\n")
            self.updateStatus()

            while True:
                key = keyboard.read_key()
                if key == "p":
                    self.playing = not self.playing
                    self.player.pause()
                    self.post_update()
                    self.updateStatus()
                elif key == "1" or key == "2" or key == "3" or key == "4" or key == "5" or key == "6" or key == "7" or key == "8" or key == "9":
                    self.sound_id = int(key)
                    fileN = self.sounds[self.id][self.sound_id]
                    self.load_sound("./audio/%s.mp3"%fileN)
                # elif key == "2":
                #     self.sound_id = 2
                #     self.load_sound("./audio/id_%s_2.mp3"%self.id)
                # elif key == "3":
                #     self.sound_id = 3
                #     self.load_sound("./audio/id_%s_3.mp3"%self.id)
                elif key == "q":
                    print("S")
                    self.close()
                else:
                    self.updateStatus("Unknown key: %s"%key)

                time.sleep(1)
        except KeyboardInterrupt:
            print("PlayAudio was interrupted. Shutting down python script!")
        except Exception as e:
            print("LOOP EXCEPTION: " + str(e))
    

    def updateStatus(self, warn = ""):
        playing = "Paused"
        if self.playing:
            playing = "Playing"

        prog = int(self.pos * 100 / 5)
        perc = int(self.pos * 100)
        progressbar = "[%s%s]"%(("■"*prog), (" "*(20-prog)))

        fileN = self.sounds[self.id][self.sound_id]
        print("Status: %s | Playing sound: %s | Position: %s - %d %% | Warning: %s"%(playing, fileN, progressbar, perc, warn), end="\r")

    def load_sound(self, file):
        media = vlc.Media(file)
        self.player.set_media(media)
        self.player.play()
        self.playing = True
        self.post_update()

        self.updateStatus()

    def post_update(self):
        ts = datetime.utcnow()
        fileN = self.sounds[self.id][self.sound_id]
        self.conn.publish(self.SOUND_PLAYED_TOPIC, ts, "Sound: %s, Playing: %s"%(fileN, self.playing))

    def pos_check(self):
        while self.tks:
            if self.playing:
                self.pos = self.player.get_position()

                if self.pos >= 0.995:
                    self.playing = False
                    self.post_update()

                self.updateStatus()
            time.sleep(1)


    def close(self):
        self.tks = False
        self.player.stop()

        time.sleep(2)

        print("Closed down audio playing script!")
        exit()

if __name__ == "__main__":
    audio_processor = PlayAudio()
    audio_processor.console_loop()
    audio_processor.close()

