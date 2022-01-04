#!/usr/bin/env python3

import configparser
import argparse
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from requests.exceptions import RequestException, ConnectionError
from datetime import datetime

import os, time, sys
import numpy as np
import cv2
from threading import Thread
from imutils.video import FileVideoStream


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def showStream(url, name, stop, resize=None, savedir=None):
	print("{} --- Starting video stream from {} ... ".format(datetime.now().replace(microsecond=0), name), end="")

	#vs = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
	vs = FileVideoStream(url).start()

	if not vs or not vs.stream.isOpened():
		print("failed.")
		return
	else:
		print("started.")

	# let the buffer fill up a bit
	time.sleep(1.0)

	#vs.stream.set(cv2.CAP_FFMPEG) # ?

	frameRate = float(vs.stream.get(cv2.CAP_PROP_FPS))
	frameWidth  = int(vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	#print("{} --- Capturing input stream {}x{} @ {:.1f} frameRate".format(datetime.now().replace(microsecond=0), frameWidth, frameHeight, frameRate))

	# resize if parameters are given
	if resize and len(resize) == 2:
		frameWidth, frameHeight = resize

	if savedir:
		outFile = '{} {}.avi'.format(datetime.now().replace(microsecond=0), name).replace(':', '.')
		outFile = os.path.join(savedir, outFile)

		# Checks and deletes the output file
		# You can't reuse an existing file or it will throw an error
		if os.path.isfile(outFile):
		    os.remove(outFile)

		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		#out = cv2.VideoWriter(outFile, fourcc, 20.0, (frameWidth, frameHeight))
		out = cv2.VideoWriter(outFile, fourcc, frameRate, (frameWidth, frameHeight))
		#print("{} --- Writing output stream  {}x{} @ {:.1f} fps".format(datetime.now().replace(microsecond=0), frameWidth, frameHeight, frameRate))
	else:
		out = None

	# loop over the frames from the video stream
	while not stop() and vs.running():
		frame = vs.read()

		# and resize it
		if resize and len(resize) == 2:
			frame = cv2.resize(frame, resize)

		# save for video
		if out:
			out.write(frame)

		# show the output frame
		cv2.imshow(name, frame)

		# if the `q` key was pressed, break from the loop
		if (cv2.waitKey(1) & 0xFF) == ord("q"):
			break

	#print(" --- Stopping video stream from {} ... ".format(name), end="")
	print("{} --- Stopping video stream from {} ... ".format(datetime.now().replace(microsecond=0), name), end="")

	# cleanup
	if out:
		out.release()

	cv2.destroyAllWindows()
	vs.stop()

	print("stopped.")


class CamLifeview():
	def __init__(self, camera, resize=None, savedir=None, subtype=0):
		self.VIDEO_URL='rtsp://{user}:{password}@{host}:554/cam/realmonitor?channel=1&subtype=' + str(subtype)

		if 'host' in camera and camera['host']:
			self.url = self.VIDEO_URL.format(**camera)
		else:
			self.url = None

		if 'name' in camera and camera['name']:
			self.name = camera['name']
		else:
			self.name = camera['host']

		self.resize = resize
		self.savedir = savedir

		self.thread = None
		self.stopped = True


	def running(self):
		self.stopped = not (self.thread and self.thread.is_alive())
		return not self.stopped


	def start(self):
		if not self.stopped:
			if self.thread and self.thread.is_alive():
				print(" --- Stream already started.")
				return True #self

		if self.url:
			self.stopped = False

			self.thread = Thread(target=showStream, args=(self.url, self.name, lambda: self.stopped,), kwargs={'resize': self.resize, 'savedir': self.savedir})
			self.thread.start()

			return True #self
		else:
			print(" --- URL invalid.")
			return False #None


	def stop(self):
		if self.thread and not self.thread.is_alive():
			self.stopped = True

		if self.thread and not self.stopped:
			self.stopped = True


#
# Source:
# https://github.com/pnbruckner/homeassistant-config/blob/3ca2d9db735dfc026ec02f08bb00006a90730b4d/tools/test_events.py
#

class LoginError(Exception):
	"""A login error occcured"""


class EventMonitor():
	def __init__(self, camera, events='VideoMotion', resize=(640, 360), savedir=None, subtype=0):
		self.EVENTMGR_URL = 'http://{}:{}/cgi-bin/eventManager.cgi?action=attach&codes=[{}]'

		self.Camera = camera
		self.Lifeview = None
		self.Connected = False

		self.Events = events
		self.URL = self.EVENTMGR_URL.format(self.Camera['host'], self.Camera['port'], self.Events)

		self.resize = resize
		self.savedir = savedir
		self.subtype = subtype


	def onConnect(self):
		if not self.Lifeview:
			print('{} Connected to event manager on {} ({}).'.format(datetime.now().replace(microsecond=0), self.Camera['host'], self.Camera['name']))
			self.Lifeview = CamLifeview(self.Camera, resize=self.resize, savedir=self.savedir, subtype=self.subtype)
		self.Connected = True


	def onDisconnect(self, reason):
		if self.Lifeview:
			print('{} Disconnected from event manager on {} ({}). Reason: {}.'.format(datetime.now().replace(microsecond=0), self.Camera['host'], self.Camera['name'], reason))
			self.Lifeview.stop()
			self.Lifeview = None
		self.Connected = False


	def onEvent(self, event, action):
		if event not in self.Events:
			return

		print("{} Event '{}:{}' received from {}".format(datetime.now().replace(microsecond=0), event, action, self.Camera['name']))

		if action == 'Start':
			if self.Lifeview:
				if not self.Lifeview.start():
					self.Lifeview = None
		elif action == 'Stop':
			if self.Lifeview:
				self.Lifeview.stop()


	def connect(self, auth='digest', retries=0):
		response = None

		with requests.Session() as session:
			if 'user' in self.Camera and 'password' in self.Camera: #and self.Camera['user']:
				if auth == 'digest':
					session.auth = HTTPDigestAuth(self.Camera['user'], self.Camera['password'])
				else:
					session.auth = HTTPBasicAuth(self.Camera['user'], self.Camera['password'])

			for i in range(1, 2 + retries):
				try:
					response = session.get(self.URL, timeout=(3.05, None), stream=True, verify=True)
					if response.status_code == 401:
						raise LoginError
					response.raise_for_status()
				except LoginError:
					print("Login error! Please check username and password.")
					break
				except (RequestException, ConnectionError) as e:
					print(e)
					continue
				else:
					break

		return response


	def _lines(self, response):
		line = ''
		for char in response.iter_content(decode_unicode=True):
			line = line + char
			if line.endswith('\r\n'):
				yield line.strip()
				line = ''


	def start(self):
		Thread(target=self._start).start()


	def _start(self):
		reason = "Unknown"

		while True:
			response = self.connect()

			if response and response.status_code == 200:
				if not response.encoding:
					response.encoding = 'utf-8'

				try:
					for line in self._lines(response):
						if line == 'HTTP/1.1 200 OK':
							self.onConnect()

						if not line.startswith('Code='):
							continue

						event = dict()
						for KeyValue in line.split(';'):
							key, value = KeyValue.split('=')
							event[key] = value

						if 'action' in event:
							self.onEvent(event['Code'], event['action'])
					reason = "No response"
					#time.sleep(5)
				except KeyboardInterrupt:
					reason = "User terminated"
					break
				except Exception as e:
					reason = e
					break
				finally:
					self.onDisconnect(reason)
					response.close()
			else:
				print("Unable to connect.")
				break


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--config", type=str, default="camera.cfg",
	help="name or path of the config file (default: camera.cfg)")
	ap.add_argument("-o", "--out", type=str, default=None,
	help="output directory of the saved video stream (default: No saving) ")
	ap.add_argument("-r", "--resize", type=str, default=None,
	help="resize output format: WidthxHeight (default: No resizing)")
	ap.add_argument("-s", "--subtype", type=int, default=0,
	help="subtype of the stream to use: 0=HighRes, 1=LowRes (default: 0))")
	ap.add_argument("-q", "--quiet", action="store_true",
	help="suppresses console output (default: False)")
	args = vars(ap.parse_args())

	try:
		reSize = tuple([int(x) for x in args['resize'].lower().split('x')])
	except:
		reSize = None

	subType = args['subtype']

	if args['out']:
		saveDir = args['out']
		if not os.path.isdir(saveDir):
			saveDir = os.path.abspath(os.path.dirname(__file__))
	else:
		saveDir = None

	configFile = args['config']
	if not os.path.isfile(configFile):
		configFile = os.path.join(os.path.abspath(os.path.dirname(__file__)), configFile)
	if not os.path.isfile(configFile):
		print("No config file!")
		sys.exit(1)

	config = configparser.ConfigParser()
	config.read([configFile])

	if args['quiet']:
		devnull = open(os.devnull, 'w')
		sys.stdout = devnull

	for section in config.sections():
		camera = {
		  'name':     section,
		  'host':     config.get(section, 'host'),
		  'port':     int(config.get(section, 'port')),
		  'user':     config.get(section, 'user'),
		  'password': config.get(section, 'password')
		}
		EventMonitor(camera, resize=reSize, savedir=saveDir, subtype=subType).start()
