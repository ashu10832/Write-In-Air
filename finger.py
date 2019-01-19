import cv2
import numpy as np 
import copy



rect1_tl = (620,240)
rect2_tl = (620,340)
rect3_tl = (620,440)
rect4_tl = (540,370)
rect5_tl = (700,370)

height = 30
width = 30

def getHistogram(frame):
	roi1 = frame[rect1_tl[1]:rect1_tl[1]+width,rect1_tl[0]:rect1_tl[0]+height]
	roi2 = frame[rect2_tl[1]:rect2_tl[1]+width,rect2_tl[0]:rect2_tl[0]+height]
	roi3 = frame[rect3_tl[1]:rect3_tl[1]+width,rect3_tl[0]:rect3_tl[0]+height]
	roi4 = frame[rect4_tl[1]:rect4_tl[1]+width,rect4_tl[0]:rect4_tl[0]+height]
	roi5 = frame[rect5_tl[1]:rect5_tl[1]+width,rect5_tl[0]:rect5_tl[0]+height]
	roi = np.concatenate((roi1,roi2,roi3,roi4,roi5),axis = 0)
	roi_hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

	return cv2.calcHist([roi_hsv],[0,1],None,[180,256],[0,180,0,256])


def drawRectangles(frame = 0):
	frame_with_rect = frame
	cv2.rectangle(frame_with_rect,rect1_tl,tuple(np.array(rect1_tl)+np.array((height,width))),(0,0,255),1)
	cv2.rectangle(frame_with_rect,rect2_tl,tuple(np.array(rect2_tl)+np.array((height,width))),(0,0,255),1)
	cv2.rectangle(frame_with_rect,rect3_tl,tuple(np.array(rect3_tl)+np.array((height,width))),(0,0,255),1)
	cv2.rectangle(frame_with_rect,rect4_tl,tuple(np.array(rect4_tl)+np.array((height,width))),(0,0,255),1)
	cv2.rectangle(frame_with_rect,rect5_tl,tuple(np.array(rect5_tl)+np.array((height,width))),(0,0,255),1)
	return frame_with_rect


def getMask(frame, histogram):
	frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	mask = cv2.calcBackProject([frame_hsv],[0,1],histogram,[0,180,0,256],1)
	_,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)



	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	mask = cv2.filter2D(mask,-1,kernel)

	kernel1 = np.ones((7,7),np.uint8)
	mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
	mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask = cv2.bilateralFilter(mask,5,75,75)

	return mask

def getMaxContour(mask):
	_, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	max = 0
	mi = 0
	for i in range(len(contours)):
		area = cv2.contourArea(contours[i])
		if area > max:
			max = area
			mi = i
	return contours[mi]

def drawDefects(frame_with_rect,maxContour,hull):
	defects = cv2.convexityDefects(maxContour,hull)

	for i in range(defects.shape[0]):
	    s,e,f,d = defects[i,0]
	    start = tuple(maxContour[s][0])
	    end = tuple(maxContour[e][0])
	    far = tuple(maxContour[f][0])
	    cv2.line(frame_with_rect,start,far,[255,0,0],2)
	    cv2.line(frame_with_rect,far,end,[0,255,0],2)
	    cv2.circle(frame_with_rect,far,5,[0,0,255],-1)
def getCentroid(contour):
	M = cv2.moments(contour)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return cx,cy

def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

def cropCharacter(canvas):
	print(canvas.shape)
	for i in range(canvas.shape[0]):
		for j in range(canvas.shape[1]):
			print('i {i}',i)
			print('j {j}',j)

			if canvas[i,j]!=255:
				canvas = canvas[i:canvas.shape[0],:]

	return canvas






def main():
	cap = cv2.VideoCapture(0)

	canvas = np.zeros((720,1280), np.uint8)

	far_points = []

	pressed = False
	isDrawing = False
	while True:
		_ , frame = cap.read()
		frame = cv2.flip(frame,flipCode = 1)
		originalFrame = copy.deepcopy(frame)
		originalFrame = drawRectangles(originalFrame)
		canvas[:,:] = 255



		key = cv2.waitKey(1)

		if key & 0xFF == ord('a'):
			pressed = True
			histogram = getHistogram(frame)

		if key & 0xFF == ord('d'):
			isDrawing = True

		if key & 0xFF == ord('c'):
			canvas[:,:] = 255


		if isDrawing:
			if len(far_points)>100:
				far_points.pop(0)
			far_points.append(far)
			for i in range(len(far_points)-1):
				cv2.line(originalFrame, far_points[i], far_points[i+1], (255,5,255), 10)
				cv2.line(canvas, far_points[i], far_points[i+1], (0,0,0), 10)

		if key & 0xFF == ord('f'):
			isDrawing = False
			#canvas = cropCharacter(canvas)
			cv2.imshow('LeftCrop', canvas)
			cv2.imwrite('Drawing.jpg', canvas)



		if pressed:
			mask = getMask(frame,histogram)
			maxContour = getMaxContour(mask)
			epsilon = 0.25*cv2.arcLength(maxContour,True) 
			approx = cv2.approxPolyDP(maxContour,epsilon,True)
			hull = cv2.convexHull(maxContour,returnPoints = False)
			drawDefects(originalFrame, maxContour, hull)
			defects = cv2.convexityDefects(maxContour,hull)
			far = farthest_point(defects, maxContour, getCentroid(maxContour))
			cv2.circle(originalFrame,far,10,[0,200,255],-1)
			


			cv2.imshow('Mask', mask)
			#cv2.imshow('Canvas', canvas)



		if key & 0xFF == ord('q'):
			break
		cv2.imshow('frame',originalFrame)

	cap.release()
	cv2.destroyAllWindows()



if __name__ == '__main__':
	main()
