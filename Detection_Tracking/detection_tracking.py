# Citation
# 1. http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html
# 2. http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html 
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')


def help_message():
    print("Usage: [Option_Number] [Input_Video] [Output_Directory]")
    print("[Option Number]")
    print("1 Camshift")
    print("2 Particle Filter")
    print("3 Kalman Filter")
    # print("4 Optical Flow")
    print("[Input_Video]")
    print("Path to the input video")
    print("[Output_Directory]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]


def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c, r, w, h = window
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices

def particleevaluator(back_proj, particle):
	return back_proj[particle[1], particle[0]]

def skeleton_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter,c+w/2,r+h/2))  # Write as 0,pt_x,pt_y
    im = frame[r:r + h, c:c + w]
    cv2.imwrite("./face.png", im)
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c, r, w, h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c, r, w, h))  # this is provided for you
    terminal_condition = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    # initialize the tracker
    # e.g. kf = cv2.KalmanFilter(4,2,0)

    # Initialise Kalman Filter
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.transitionMatrix = np.array([[1., 0., .1, 0.], [0., 1., 0., .1], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    kf.measurementMatrix = 1. * np.eye(2, 4)
    kf.processNoiseCov = 1e-5 * np.eye(4, 4)
    kf.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kf.errorCovPost = 1e-1 * np.eye(4, 4)
    kf.statePost = state

    #Initialise Particle Filter
    no_particles = 300
    hsvp = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_bp = cv2.calcBackProject([hsvp], [0], roi_hist, [0, 180], 1)
    init_pos = np.array([c + w/2.0, r + h/2.0], int)
    particles = np.ones((no_particles, 2), int) * init_pos
    #print particles
    f0 = particleevaluator(hist_bp, particles.T) * np.ones(no_particles)
    #print str(f0)
    weights = np.ones(no_particles)/no_particles
    stepsize = 20
    

    while (1):
        ret, frame = v.read()  # read another frame
        if ret == False:
            break

        # perform the tracking
        # e.g. cv2.meanShift, cv2.CamShift, or kalman.predict(), kalman.correct()
        if (file_name == "output_camshift.txt"):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            d = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.CamShift(d, track_window, terminal_condition)
            
            (c, r, w, h) = track_window
            output.write("%d,%d,%d\n" % (frameCounter,c+w/2,r+h/2))  # Write as frame_index,pt_x,pt_y
            
        elif file_name == "output_kalman.txt":
            prediction = kf.predict()
            
            (c1, r1, w1, h1) = detect_one_face(frame)
            if (c1 == 0 and r1 == 0 and w1 == 0 and h1 == 0):
            	# use prediction
                posterior = prediction
                #print "here --- Prediction for frame " + str(frameCounter) + "= " + str(prediction)                
            else:
                measurement = np.array([c1 + w1 / 2, r1 + h1 / 2], dtype='float64')
                posterior = kf.correct(measurement)
                #print "Posterior for frame " + str(frameCounter) + "= " + str(posterior)
            img = frame
            pts = np.int0(posterior)
            
            (c,r,w,h) = pts
            output.write("%d,%d,%d\n" % (frameCounter,c+w/2,r+h/2))  # Write as frame_index,pt_x,pt_y

        elif file_name == "output_particle.txt":
        	#print ""
        	np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
        	particles = particles.clip(np.zeros(2), np.array((frame.shape[1], frame.shape[0]))-1).astype(int)
        	hsvp = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        	hist_bp = cv2.calcBackProject([hsvp], [0], roi_hist, [0, 180], 1)
        	f = particleevaluator(hist_bp, particles.T)
        	weights = np.float32(f.clip(1))
        	weights /= np.sum(weights)
        	pos = np.sum(particles.T*weights, axis=1).astype(int)
        	
        	if(1./np.sum(weights**2) < no_particles/2.):
        		particles = particles[resample(weights),:]
        	output.write("%d,%d,%d\n" % (frameCounter, pos[0], pos[1]))  # Write as frame_index,pt_x,pt_y

        
        frameCounter = frameCounter + 1

    output.close()


if __name__ == '__main__':
    Option_number = -1

    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else:
        Option_number = int(sys.argv[1])
        if (Option_number > 4 or Option_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (Option_number == 1):
        skeleton_tracker(video, "output_camshift.txt")
    elif (Option_number == 2):
        skeleton_tracker(video, "output_particle.txt")
    elif (Option_number == 3):
        skeleton_tracker(video, "output_kalman.txt")
    