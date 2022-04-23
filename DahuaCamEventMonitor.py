#!/usr/bin/env python3

# pip install requests
# pip install nummpy
# pip install opencv-contrib-python
# pip install imutils

import configparser
import argparse
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from requests.exceptions import RequestException, ConnectionError
from datetime import datetime

import os, sys
import numpy as np
import cv2
from time import sleep, time
from multiprocessing import Process, Event, active_children
from imutils.video import FileVideoStream
from imutils.object_detection import non_max_suppression


glbl_streamParms = {
	'resize':  	None,
	'detect':	False,
	'savedir':	None,
	'playtime':	10
}

glbl_subType = 0


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def log(message):
	print(f"{datetime.now().replace(microsecond=0)} {message}")


def showStream(url, name, eventStop, resize=None, detect=False, savedir=None, playtime=10):
	log(f"[{name}] Starting video stream ... ")

	cv2.namedWindow(name)
	cv2.startWindowThread()

	vs = FileVideoStream(url).start()

	if not vs or not vs.stream.isOpened():
		log(f"[{name}] Failed to start Video stream.")
		return

	# let the buffer fill up a bit
	#sleep(1.0)

	#vs.stream.set(cv2.CAP_FFMPEG) # ?
	frameRate = float(vs.stream.get(cv2.CAP_PROP_FPS))
	frameWidth  = int(vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	#log("--- Capturing input stream {}x{} @ {:.1f} frameRate".format(frameWidth, frameHeight, frameRate))

	# resize if parameters are given
	if resize and len(resize) == 2:
		frameWidth, frameHeight = resize

	if detect:
		# initialize the HOG descriptor/person detector
		hog = cv2.HOGDescriptor()
		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

		hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}

	if savedir:
		outFile = f"{name}-{datetime.now().replace(microsecond=0)}.avi".replace(':', '.').replace(' ', '_')
		outFile = os.path.join(savedir, outFile)

		# Checks and deletes the output file
		# You cant have a existing file or it will through an error
		if os.path.isfile(outFile):
		    os.remove(outFile)

		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(outFile, fourcc, frameRate, (frameWidth, frameHeight))
		#log("--- Writing output stream {}x{} @ {:.1f} fps".format(frameWidth, frameHeight, frameRate))
	else:
		out = None

	stopped = False
	starttime = time()

	log(f"[{name}] Connected. Streaming video data ... ")
	# loop over the video stream frames
	while vs.running():
		try:
			# Make sure the stream is displayed for at least playtime seconds and then exit the loop
			if eventStop.is_set():
				if (time() - starttime) * 1000 > playtime * 1000:
					break
				# else remember that it was supposed to stop by setting the stopped flag to True
				else:
					stopped = True
			# If a new trigger event occurs after the stream was supposed to stop, reset the flag and the start time
			elif stopped:
				stopped = False
				starttime = time()
				log(f"[{name}] Timer reset due to new event while streaming.")

			# read frame from video stream
			frame = vs.read()

			# and resize it
			if resize and len(resize) == 2:
				frame = cv2.resize(frame, resize)

			if detect:
				#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				#gray = cv2.equalizeHist(gray)
				#objects, weights = hog.detectMultiScale(gray, **hogParams)
				objects, weights = hog.detectMultiScale(frame, **hogParams)

				#objects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in objects])
				objects = np.array([[x, y, x + w, y + h] for i, (x, y, w, h) in enumerate(objects) if weights[i] > 0.6])
				objects = non_max_suppression(objects, overlapThresh=0.6)

				# Draw a rectangle around the objects
				for (x1, y1, x2, y2) in objects:
					# the HOG detector returns slightly larger rectangles than the real objects.
					# so we slightly shrink the rectangles to get a nicer output.
					w = x2 - x1 + 1
					h = y2 - y1 + 1
					pad_w, pad_h = int(0.15 * w), int(0.05 * h)
					cv2.rectangle(frame, (x1 + pad_w, y1 + pad_h), (x2 - pad_w, y2 - pad_h), (0, 255, 0), 2)

			# save for video
			if out:
				out.write(frame)

			# show the output frame
			cv2.imshow(name, frame)
			#cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)

			# if the `q` key was pressed, break from the loop
			if (cv2.waitKey(1) & 0xFF) == ord("q"):
				break

		except KeyboardInterrupt:
			break

	# cleanup
	if out:
		out.release()

	#cv2.destroyAllWindows()
	cv2.destroyWindow(name)
	vs.stop()

	if sys.platform == 'darwin':
		# the following is necessary on the mac
		cv2.waitKey(1)

	log(f"[{name}] Video stream stopped.")


class CamLifeview():
	def __init__(self, camera, **streamParms):
		self.VIDEO_URL='rtsp://{user}:{password}@{host}:554/cam/realmonitor?channel=1&subtype={subtype}'
		#self.VIDEO_URK='http://{user}:{password}@{host}/cgi-bin/mjpg/video.cgi?channel=1&subtype={subtype}' # substream must be enabled with MJPEG encoding

		if 'host' in camera and camera['host']:
			self.url = self.VIDEO_URL.format(**camera)
		else:
			self.url = None

		if 'name' in camera and camera['name']:
			self.name = camera['name']
		else:
			self.name = camera['host']

		self.streamParms = streamParms

		self.process = None
		self.stopped = True

		self.eventStop = Event()


	def running(self):
		return self.process and self.process.is_alive()


	def start(self):
		self.eventStop.clear()
		if self.running():
			self.stopped = False
			log(f"[{self.name}] Video stream has already started.")

			return True #self
		elif self.url:
			self.stopped = False

			self.process = Process(target=showStream,
				args=(self.url, self.name, self.eventStop,),
				kwargs=self.streamParms)
			self.process.start()

			return True #self
		else:
			log(f"[{self.name}] No valid URL.")

			return False #None


	def stop(self):
		self.eventStop.set()
		self.stopped = True


#
# Source:
# https://github.com/pnbruckner/homeassistant-config/blob/3ca2d9db735dfc026ec02f08bb00006a90730b4d/tools/test_events.py
#

class LoginError(Exception):
	"""A login error occcured"""


class EventMonitor():
	def __init__(self, camera, **streamParms):
		self.EVENTMGR_URL = 'http://{host}:{port}/cgi-bin/eventManager.cgi?action=attach&codes=[{events}]'

		self.Camera = camera
		self.Lifeview = None
		self.Connected = False

		self.URL = self.EVENTMGR_URL.format(**camera)

		self.process = None
		self.streamParms = streamParms


	def onConnect(self):
		if not self.Lifeview:
			log(f"[{self.Camera['name']}] Connected to event manager on {self.Camera['host']}.")
			self.Lifeview = CamLifeview(self.Camera, **self.streamParms)
		self.Connected = True


	def onDisconnect(self, reason):
		if self.Lifeview:
			log(f"[{self.Camera['name']}] Disconnected from event manager on {self.Camera['host']}. Reason: {reason}.")
			self.Lifeview.stop()
			self.Lifeview = None
		self.Connected = False


	def onEvent(self, event, action):
		if event not in self.Camera['events']:
			return

		log(f"[{self.Camera['name']}] Received event '{event}:{action}'.")

		if action == 'Start':
			if self.Lifeview:
				if not self.Lifeview.start():
					self.Lifeview = None
		elif action == 'Stop':
			if self.Lifeview:
				self.Lifeview.stop()


	def connect(self, retries=0):
		response = None

		with requests.Session() as session:
			if 'user' in self.Camera and 'password' in self.Camera: #and self.Camera['user']:
				if 'auth' in self.Camera:
					if self.Camera['auth'] == 'digest':
						session.auth = HTTPDigestAuth(self.Camera['user'], self.Camera['password'])
					else:
						session.auth = HTTPBasicAuth(self.Camera['user'], self.Camera['password'])

			for i in range(1, 2 + retries):
				if i > 1:
					log(f"[{self.Camera['name']}] Retrying ...")
				try:
					response = session.get(self.URL, timeout=(3.05, None), stream=True, verify=True)
					if response.status_code == 401:
						raise LoginError
					response.raise_for_status()
				except LoginError:
					log(f"[{self.Camera['name']}] Login error! Please check username and password.")
					break
				except (RequestException, ConnectionError) as e:
					#log(f"[{self.Camera['name']}] Failed to retrieve data. Error: {e}")
					log(f"[{self.Camera['name']}] Failed to retrieve data.")
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
		self.process = Process(target=self._start)
		self.process.start()

		return self.process


	def _start(self):
		reason = "Unknown"

		while True:
			response = self.connect(retries=1)

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
					#sleep(5.0)
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
				log(f"[{self.Camera['name']}] Unable to connect.")
				break


def myexcepthook(exctype, value, traceback):
	pass
	#log(f"Value: {exctype}")
	#for p in active_children():
	#	p.terminate()


sys.excepthook = myexcepthook


if __name__ == '__main__':
	log("Starting Event Monitor ...")

	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--config", type=str, default="camera.cfg",
		help="name or path of the config file (default: camera.cfg)")
	ap.add_argument("-d", "--detect", action="store_true",
		help="enable people detection (default: False)")
	ap.add_argument("-o", "--out", type=str, default=None,
		help="output directory of the saved video stream (default: No saving) ")
	ap.add_argument("-p", "--playtime", type=int, default=10,
		help="video playtime in seconds after an event occured (default: 10)")
	ap.add_argument("-r", "--resize", type=str, default=None,
		help="resize output format: WidthxHeight (default: No resizing)")
	ap.add_argument("-s", "--subtype", type=int, default=0,
		help="subtype of the stream to use: 0=HighRes, 1=LowRes (default: 0))")
	ap.add_argument("-q", "--quiet", action="store_true",
		help="suppresses console output (default: False)")
	args = vars(ap.parse_args())

	glbl_streamParms['detect'] = args['detect']
	glbl_streamParms['playtime'] = args['playtime']

	try:
		glbl_streamParms['resize'] = tuple([int(x) for x in args['resize'].lower().split('x')])
	except:
		glbl_streamParms['resize'] = None #(640, 360)

	if args['out']:
		glbl_streamParms['savedir'] = args['out']
		if not os.path.isdir(glbl_streamParms['savedir']):
			glbl_streamParms['savedir'] = os.path.abspath(os.path.dirname(__file__))
	else:
		glbl_streamParms['savedir'] = None

	glbl_subType = args['subtype']

	configFile = args['config']
	if not os.path.isfile(configFile):
		configFile = os.path.join(os.path.abspath(os.path.dirname(__file__)), configFile)
	if not os.path.isfile(configFile):
		log("No config file! Exit.")
		sys.exit(1)

	if args['quiet']:
		devnull = open(os.devnull, 'w')
		sys.stdout = devnull

	config = configparser.ConfigParser()
	config.read([configFile])

	procList = []

	log(f"Reading config from {configFile} ...")
	for section in config.sections():
		try:
			camera = {
			  'name':		section,
			  'host':		config.get(section, 'host'),
			  'port':		int(config.get(section, 'port', fallback='80')),
			  'user':		config.get(section, 'user'),
			  'password':	config.get(section, 'password'),
			  'auth': 		config.get(section, 'auth', fallback='digest'),
			  'events': 	config.get(section, 'events', fallback='VideoMotion'),
			  'subtype':	glbl_subType
			}
			log(f"[{section}] Config okay.")
		except:
			log(f"[{section}] Error! Check your configuration.")

		p = EventMonitor(camera, **glbl_streamParms).start()
		procList.append(p)

	for p in procList:
		p.join()
