import cv2
import os
import math
import argparse
from os.path import isdir 
from os import listdir
from os.path import isfile, join
from skimage.transform import resize   # for resizing images
import dlib 
from imutils import face_utils
import numpy as np
import dict2xml
from dicttoxml import dicttoxml
import zlib
import sys
RECTANGLE_LENGTH = 48
from pathlib import Path
import re

class lipReading(object):

	def __init__(self):
		#initial value of stepSize
		self.face_cascade = cv2.CascadeClassifier('../opencv/haarcascade_frontalface_default.xml')
		self.mouth_cascade=cv2.CascadeClassifier('/Users/gupsekobas/opencv_contrib-4.0.1/modules/face/data/cascades/haarcascade_mcs_mouth.xml')
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor('../opencv/shape_predictor_68_face_landmarks.dat')
		self.xmouthpoints = []
		self.ymouthpoints = []
		self.image = None
		self.count = 0
		self.frame_dict =  {"Distance":{},"Angle":{}}
		self.distanceArray = np.zeros(19)
		self.angleArray = np.zeros(19)
		self.video_dict =  {}
		self.wordArray= []
		self.walk_dir = "../lrw/lipread_mp4"
		self.datasets = ["test", "train","val"]
		self.fileNum = None
		self.targetDir = None
		


	#this function calculate the difference of each point
	#in the mouth data to the 48th point.
	#mouth data has 20 points for every mouth detected.
	#it returns the normalized value of the distances in order that video resolution dont effect the output.
	def calculateDistanceOfMouthPoints(self):
		for i in range(1 , 20):
			#find the distance between 2 points
			self.distanceArray[i-1] = math.hypot(self.xmouthpoints[i] - self.xmouthpoints[0] , self.ymouthpoints[i] - self.ymouthpoints[0])
			#find the angle between 2 points
			self.angleArray[i-1] = math.atan2(self.xmouthpoints[i] - self.xmouthpoints[0], self.ymouthpoints[i] - self.ymouthpoints[0])

		norm = np.linalg.norm(self.distanceArray)
		self.distanceArray = self.distanceArray/norm

		for i in range(1 , 20):
			self.frame_dict["Distance"][str(i-1)]=self.distanceArray[i-1]
			self.frame_dict["Angle"][str(i-1)]= self.angleArray[i-1]

	def get_mouth(self):
		#image = cv2.imread(filename)
		#image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale image
		rects = self.detector(gray, 1)
		if len(rects) > 1:
			print( "ERROR: more than one face detected")
			return
		if len(rects) < 1:
			print( "ERROR: no faces detected")
			return 
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		rect = rects[0]
		shape = self.predictor(gray, rect)
		#shape = face_utils.shape_to_np(shape)
		#mouth = shape[lStart:lEnd]

		pad = 10
		self.xmouthpoints = [shape.part(x).x for x in range(lStart, lEnd)]
		self.ymouthpoints = [shape.part(x).y for x in range(lStart, lEnd)]
		#print(self.xmouthpoints)
		#print(self.ymouthpoints)
		self.calculateDistanceOfMouthPoints()
		
		maxx = max(self.xmouthpoints)
		minx = min(self.xmouthpoints)
		w = maxx - minx
		maxy = max(self.ymouthpoints)
		miny = min(self.ymouthpoints) 
		h = maxy- miny

		if w > h:
			rec_len = w
		else:
			rec_len = h

		w = rec_len +10
		h = rec_len +10
		minx= minx -5
		miny= miny -5
		'''
		for i in range(20):
			cv2.circle(self.image, (self.xmouthpoints[i],self.ymouthpoints[i]), radius=0, color=(0, 0, 255), thickness=2)
		'''
		crop_image = self.image[miny:miny + h,minx:minx+w]
		

		return crop_image

	def extract_mouth_data(self):
		


		mouth= self.get_mouth()
		if mouth is not None:
			#assuming there is only one face in the video
			#cv2.imwrite("frame%d.jpg" % self.count, self.image)     # save frame as JPEG file   
			mouth_filename =  self.targetDir  + "/"+ "mouth_{:05d}_{:05d}.jpg".format(self.fileNum ,self.count)
			#print(mouth.shape)
			mouth = cv2.resize(mouth, (RECTANGLE_LENGTH, RECTANGLE_LENGTH))
			#print(mouth.shape)
			cv2.imwrite(mouth_filename, mouth)
			print(mouth_filename, " saved")
			frame_key = "frame{:03d}".format( self.count)
			self.video_dict[frame_key]= self.frame_dict
			self.count = self.count+1

		else:
			print("no face written on file!")

	# Walk into directories in filesystem
	# Ripped from os module and slightly modified
	# for alphabetical sorting
	#
	def sortedWalk(self, top, topdown=True, onerror=None):
		from os.path import join, isdir, islink

		names = os.listdir(top)
		names.sort()
		dirs, nondirs = [], []

		for name in names:
			if isdir(os.path.join(top, name)):
				dirs.append(name)
			else:
				nondirs.append(name)

		if topdown:
			yield top, dirs, nondirs
		for name in dirs:
			path = join(top, name)
			if not os.path.islink(path):
				for x in self.sortedWalk(path, topdown, onerror):
					yield x
		if not topdown:
			yield top, dirs, nondirs

	def getFolderNamesInRootDir(self):
		

		print('walk_dir = ' + self.walk_dir)

		# If your current working directory may change during script execution, it's recommended to
		# immediately convert program arguments to an absolute path. Then the variable root below will
		# be an absolute path as well. Example:
		# walk_dir = os.path.abspath(walk_dir)
		print('walk_dir (absolute) = ' + os.path.abspath(self.walk_dir))

		for root, subdirs, files in self.sortedWalk(self.walk_dir):
			print('--\nroot = ' + root)
			for subdir in sorted(subdirs):
				print('\t- subdirectory ' + subdir)
				self.wordArray.append(subdir)
			break

	def createFoldersForEveryWord(self):
		for item in self.wordArray:
			
			Path("data/"+item).mkdir(parents=True, exist_ok=True)
			Path("data/"+item+"/test").mkdir(parents=True, exist_ok=True)
			Path("data/"+item+"/train").mkdir(parents=True, exist_ok=True)
			Path("data/"+item+"/val").mkdir(parents=True, exist_ok=True)

	def processVideos(self):
		print('walk_dir = ' + self.walk_dir)
		for item in self.wordArray:
			for subitem in self.datasets :
				sourceDir = self.walk_dir +"/" +item + "/" +subitem
				self.targetDir = "data" +"/" +item + "/" +subitem
				for root, subdirs, files in self.sortedWalk(os.path.abspath(sourceDir)):
						for file in files:
							if file.endswith(".mp4"):
								filepath = os.path.join(root, file)
								print("processing : ", filepath)
								print(re.findall('\d+', file[0:-4] ))
								self.fileNum = int(re.findall('\d+', file[0:-4] )[0])
								self.captureVideo(filepath,item)


		# If your current working directory may change during script execution, it's recommended to
		# immediately convert program arguments to an absolute path. Then the variable root below will
		# be an absolute path as well. Example:
		# walk_dir = os.path.abspath(walk_dir)
		print('walk_dir (absolute) = ' + os.path.abspath(self.walk_dir))

		for root, subdirs, files in self.sortedWalk(self.walk_dir):
			print('--\nroot = ' + root)
			for subdir in sorted(subdirs):
				print('\t- subdirectory ' + subdir)
				self.wordArray.append(subdir)
			break
	def captureVideo(self, videoFileName,word):
		self.count = 0
		vidcap = cv2.VideoCapture(videoFileName)
		success,self.image = vidcap.read()
		if success == True:
			tmp_array = self.extract_mouth_data()
			while success:
				
				success,self.image = vidcap.read()
				if success == True:
					print('Read a new frame: ', success)
					tmp_array = self.extract_mouth_data()

		xml =  dicttoxml(self.video_dict, custom_root='test', attr_type=False)
		filename = "{}/d_{}_{:05d}.hgk".format(self.targetDir, word,self.fileNum)
		#print(xml)
		print ("size1: ", sys.getsizeof(xml))

		print ("size2: ", sys.getsizeof(zlib.compress(xml)))
		f = open(filename, 'wb')
		f.write(zlib.compress(xml))
		f.close()
def main():

	lr = lipReading()
	lr.getFolderNamesInRootDir()
	lr.createFoldersForEveryWord()
	lr.processVideos()
	distanceArray = []


if __name__ == '__main__':
	main()