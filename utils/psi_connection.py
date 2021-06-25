#!/usr/bin/python

""" PsiConnection: A python script that sets up a socket to the C# psi instance. Creates an outgoing socket and provides possibilities to send messages and sync timestamps.

***************************************************************************************
Developer: Jeffrey Pronk
Github: jeffthestig
Email: j.s.pronk@student.tudelft.nl

Year: 2021

Developed at the request of Yoon Lee as part of the CSE BSc Research Project (CSE3000).
***************************************************************************************

A python script that sets up sockets to communcate with the C# psi instance. Can be instansiated by importing this script and create an instance of PsiConnection().
The init script will by default binds to tcp://127.0.0.1:12345 and connects to tcp://127.0.0.1:25565. If you want to connect to a different ip, please specify this while creating an instance by adding it as a first argument and second arg. In this case only the ip and port are necessary. Will be always use tcp.
PSI provides the subscribe socket, python connects to it! PSI connects to the publish socket, python provides it!
Can send a sync message to set the timestamp difference between two computers. By default set to False. Please don't use if both PSI and the python script are running on the same system.


Dependencies
----------
    Pre provided by Python
        - datetime
        - time
        - json

    Need to be installed
        - ntplib
        - zmq
"""

from datetime import datetime, timedelta
import json
import ntplib
import time
import zmq

""" PsiConnection: A python script that sets up a socket to the C# psi instance. Creates an outgoing socket and provides possibilities to send messages and sync timestamps.
"""
class PsiConnection:

    """ Initiate the PsiConnection instance

        @type self: PsiConnect
        @param self: Instance of self.
        @type pub_ip: string
        @param pub_ip: The ip adres the publish socket should bind to (own ip). By default set to 127.0.0.1:12345
        @type sub_ip: string
        @param sub_ip: The ip adres the subscribe socket should connect to. By default set to 127.0.0.1:25565
        @type sync: string
        @param sync: The topic name for the time sync between the python and c# script. By default set to None, in this case will ignore the time sync. If necessary, define topic name.
    """
    def __init__(self, pub_ip="127.0.0.1:12345", sub_ip="127.0.0.1:25565", sync = False):
        try:
                print("initiating PsiConnect!")

                self.pub_ip = pub_ip
                self.sub_ip = sub_ip
                self.sync = sync
                self.diff = 0
                self.zmq_cont = zmq.Context()

                self.pub_sock = self.zmq_cont.socket(zmq.PUB)
                self.pub_sock.bind("tcp://%s"%pub_ip)

                self.sub_sock = self.zmq_cont.socket(zmq.SUB)
                self.sub_sock.connect("tcp://%s"%sub_ip)
                self.sub_sock.setsockopt(zmq.SUBSCRIBE, ''.encode())
                
                if(sync):
                    self.diff = self.__req_time_sync()

        except Exception as e:
            print("SOCKET INIT ERROR: " + str(e))
        except:
            print("SOCKET INIT ERROR: unkown error!")
    
    """ Creates a payload with a given timestamp and messages and sends it over the pub_socket on a given channel.

        @type self: PsiConnect
        @param self: Instance of self.
        @type topic: string
        @param topic: Name of the topic to publish the payload/message on.
        @type ts: datetime
        @param ts: The timestamp at which the message was created. Create by running datetime.datetime.utcnow() and providing it into this method.
        @type mess: Any type that can be dumped into a json.
        @param mess: The data that has to be send.
    """
    def publish(self, topic, ts: datetime, mess):
        try:
            td = timedelta(0)
            if (self.sync):
                td = timedelta(milliseconds=(self.diff * 1000))

            pl = {}
            pl['originatingTime'] = (ts + td).isoformat()
            pl['message'] = mess
            # print("%s >> sending: (%s, %s) to %s" % (ts, ts+td, mess, topic))
            dump = json.dumps(pl).encode('utf-8')
            
            json.loads(dump.decode('utf-8'))

            print("%s >> sending %d bytes to %s" % (ts, len(dump), topic))
            self.pub_sock.send_multipart([topic.encode(), json.dumps(pl).encode('utf-8')])
        except Exception as e:
            print("PUBLISH ERROR: %s \r\n\tTopic: %s | TimeStamp: %s | Synced TimeStamp: %s | Message: %s" % (str(e), topic, ts, (ts + td), "mess"))
        except:
            print("PUBLISH ERROR: unkown error! | Topic: %s | TimeStamp: %s | Message: %s" % (topic, ts, "mess"))

    def publish(self, topic, ts: str, mess):
        try:
            
            pl = {}
            pl['originatingTime'] = ts
            pl['message'] = mess
            # print("%s >> sending: (%s, %s) to %s" % (ts, ts+td, mess, topic))
            dump = json.dumps(pl).encode('utf-8')
            
            json.loads(dump.decode('utf-8'))

            print("%s >> sending %d bytes to %s" % (ts, len(dump), topic))
            self.pub_sock.send_multipart([topic.encode(), json.dumps(pl).encode('utf-8')])
        except Exception as e:
            print("PUBLISH ERROR: %s \r\n\tTopic: %s | TimeStamp: %s | Synced TimeStamp: %s | Message: %s" % (str(e), topic, ts, (ts + td), "mess"))
        except:
            print("PUBLISH ERROR: unkown error! | Topic: %s | TimeStamp: %s | Message: %s" % (topic, ts, "mess"))


    """ Makes a request to nl.pool.ntp.org and extracts the time difference. Returns this difference.

        @type self: PsiConnect
        @param self: Instance of self.
    """
    def __req_time_sync(self):
        try:
            c = ntplib.NTPClient()
            c.request('nl.pool.ntp.org', version=3)
            time.sleep(2)
            res = c.request('nl.pool.ntp.org', version=3)

            return res.offset
        except Exception as e:
            print("POOL EXCEPTION: " + str(e))
            return 0
        except:
            print("POOL EXCEPION: Unknown error")
            return 0




