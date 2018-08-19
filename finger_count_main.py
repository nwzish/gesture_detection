import cv2
import os
import copy
from keras.models import load_model
import numpy as np

dataColor = (0,150,0)
font = cv2.FONT_HERSHEY_PLAIN
className = 'NONE'
fx , fy, fh = 10,50,45

counts = ['NONE','ONE','TWO','THREE','FOUR','FIVE']

def bm(img):
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (7,7),3)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	ret , new = cv2.threshold(img,25,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return new
def main():
	global font, size, fx, fy, fh

	x0, y0,size = 100,100,300

	model=load_model('model_finger_count3.h5')

	video = cv2.VideoCapture(0)
	cv2.namedWindow('Finger Detect', cv2.WINDOW_NORMAL)


	while True:

		_ , frame = video.read()
		frame = cv2.flip(frame,1)

		cv2.rectangle(frame,(x0,y0),(x0+size-1,y0+size-1),(0,0,255),12)

		region = frame[y0:y0+size,x0:x0+size]
		region = bm(region)

		img = np.float32(region)/255
		img = np.expand_dims(img,axis=0)
		img = np.expand_dims(img,axis = -1)
		pred = counts[np.argmax(model.predict(img)[0])]
		cv2.putText(frame, 'Finger count: %s' %(pred), (fx,fy),font, 2.0, (255,0,0),3,5)



		cv2.imshow('Finger Detect', frame)

		key =  cv2.waitKey(1) & 0xff

		if key==ord('q'):
			break

		elif key == ord('w'):
			y0 = max(y0 - 5, 0)
		elif key == ord('s'):
		    y0 = min(y0 + 5, frame.shape[0]-size)
		elif key == ord('a'):
			x0 = max(x0 - 5, 0)
		elif key == ord('d'):
			x0 = min(x0 + 5, frame.shape[1]-size)

	video.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	main()