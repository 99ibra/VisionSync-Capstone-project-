import numpy as np
import cv2
from matplotlib import pyplot as plt


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


# Open both cameras
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)                    
cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)


while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    # Undistort and rectify images
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    #Creating a DisparityMap

    stereo = cv2.StereoSGBM_create(minDisparity = 10, numDisparities = 85, blockSize = 11)   #default values 10,85,11
    disparityMap = stereo.compute(frame_left, frame_right).astype(np.float32)/16
    
    disparityImg = cv2.normalize(src = disparityMap, dst = None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)   #used for normalizing the pixel intensity in an image to a predefined range 0 to 255, where 0 represents black and 255 represents white.
    disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)

    # # # Below is a different method to get a disparitymap
    # window_size = 5
    # min_disp = 5
    # nDispFactor = 8 # adjust this (14 is good)
    # num_disp = 16*nDispFactor-min_disp

    # stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp,blockSize=window_size,P1=8*3*window_size**2, P2=32*3*window_size**2,disp12MaxDiff=1,uniquenessRatio=15,speckleWindowSize=0,speckleRange=2,preFilterCap=63,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    
    # disparity = stereo.compute(frame_left,frame_right).astype(np.float32) / 16.0
    
    # disparityImg = cv2.normalize(src = disparity, dst = None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)


    # Show the frames
    cv2.imshow("frame right", frame_right) 
    cv2.imshow("frame left", frame_left)

    cv2.imshow("Disparity ColorMap",disparityImg)

    # # Display the disparity map in a plot
    # plt.imshow(disparity,'gray')
    # plt.colorbar()
    # plt.show()

   

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
