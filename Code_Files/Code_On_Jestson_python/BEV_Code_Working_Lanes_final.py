import cv2
import numpy as np
#import matplotlib.pyplot as plt
import time
import math
#from scipy import ndimage
#from collections import Counter
#from sklearn.cluster import DBSCAN 


def distance_between_lines(line1):
    """
    Calculates the shortest distance between two lines, represented by two points each.
    """
    x1, y1, x2, y2 = line1[0]
    center = 345 

    distance_l = 0
    distance_r = 0

    if x1 < 345 and x2 < 345:
        distance_l = (abs((x1 - center)) + abs((x2 - center)))/2       
        
    else:
        distance_r = (abs((x1 - center)) + abs((x2 - center)))/2


    return distance_l, distance_r
    
def region_of_interest_side(image):

    square = np.array([[
    (0, 180), (0,250), (640,250),
    (640, 180),]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
    
    
def mask_1():
		
	H_FR = np.array([[-7.07806084e-01, -2.33438170e+00, 5.68874674e+02],
	[ 7.79288351e-03, -1.31819362e+00, 3.49425808e+02],
	[-4.38923690e-05, -6.92180391e-03, 1.00000000e+00]])

	H_RT = np.array([[-2.07590604e-02, -2.96689587e+00, 4.91699739e+01],
	[-1.07246093e+00, -2.01773980e+00, 5.78396275e+02],
	[ 6.32894442e-05, -9.52450543e-03, 1.00000000e+00]])

	H_RR = np.array([[ 2.08465243e+00, -5.93162997e+00, -4.19885578e+02],
	[-1.59339167e-01, -3.68364873e+00, -4.33454556e+02],
	[-4.19260804e-04, -1.74941873e-02, 1.00000000e+00]])

	H_LT = np.array([[-1.00622301e-01, -5.52579047e+00, 8.35377637e+02],
	[ 1.62366336e+00, -3.02940289e+00, -4.05685870e+02],
	[-7.59785044e-04, -1.43052902e-02, 1.00000000e+00]])

	img_m = np.ones((480, 640, 3), dtype = np.uint8)
	img_m = 255* img_m

	img_m_f = cv2.warpPerspective(img_m, H_FR, (640, 480))
	img_m_f[240:,:].fill(0)
	img_m_r = cv2.warpPerspective(img_m, H_RT, (640, 480))
	img_m_r[:,:240].fill(0)
	img_m_b = cv2.warpPerspective(img_m, H_RR, (640, 480))
	img_m_l = cv2.warpPerspective(img_m, H_LT, (640, 480))

	img_fb = cv2.addWeighted(img_m_f,1,img_m_b,1,0)
	img_rl = cv2.addWeighted(img_m_r,1,img_m_l,1,0)

	img_bev_m1 = cv2.bitwise_and(img_fb,img_rl)
	img_bev_m2 = cv2.bitwise_xor(img_fb,img_rl)

	return img_bev_m1, img_bev_m2
		  

def undistortion(frame, map1, map2):
    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def warp(undistorted_image, homography):
    warp1 = cv2.warpPerspective(undistorted_image, homography, (640, 480))
    return warp1
    

def draw_the_lines(img,lines): 
  imge=np.copy(img)     
  blank_image=np.zeros((imge.shape[0],imge.shape[1],3),\
                                                   dtype=np.uint8)
  for line in lines:  
    for x1,y1,x2,y2 in line:
      cv2.line(blank_image,(x1,y1),(x2,y2),(255,255,255),thickness=3)
      imge = cv2.addWeighted(imge,0.8,blank_image,1,0.0) 
  return imge


def mask():
    image_FR = np.zeros((480, 640, 3), dtype = "uint8")
    image_FR.fill(255)
    

    image_RR = np.zeros((480, 640, 3), dtype = "uint8")
    image_RR.fill(255)
  

    H_FR = np.array([[-7.07806084e-01, -2.33438170e+00, 5.68874674e+02],
 [ 7.79288351e-03, -1.31819362e+00, 3.49425808e+02],
 [-4.38923690e-05, -6.92180391e-03, 1.00000000e+00]])

 
    H_RR = np.array([[ 2.08465243e+00, -5.93162997e+00, -4.19885578e+02],
 [-1.59339167e-01, -3.68364873e+00, -4.33454556e+02],
 [-4.19260804e-04, -1.74941873e-02, 1.00000000e+00]])

 
 
    warp1 = cv2.warpPerspective(image_FR, H_FR, (640, 480))
    warp1 = cv2.GaussianBlur(warp1, (0, 0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

    warp4 = cv2.warpPerspective(image_RR, H_RR, (640, 480))
    warp4 = cv2.GaussianBlur(warp4, (0, 0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    #warp4[:600,:].fill(0)
    
    return cv2.add(warp1,warp4)

def color_balance(image):
    b, g, r = cv2.split(image)
    B = np.mean(b)
    G = np.mean(g)
    R = np.mean(r)
    K = (R + G + B) / 3
    Kb = K / B
    Kg = K / G
    Kr = K / R
    cv2.addWeighted(b, Kb, 0, 0, 0, b)
    cv2.addWeighted(g, Kg, 0, 0, 0, g)
    cv2.addWeighted(r, Kr, 0, 0, 0, r)
    return cv2.merge([b,g,r])
    
def luminance_balance(images):
    [front,back,left,right] = [cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
                               for image in images]
    hf, sf, vf = cv2.split(front)
    hb, sb, vb = cv2.split(back)
    hl, sl, vl = cv2.split(left)
    hr, sr, vr = cv2.split(right)
    V_f = np.mean(vf)
    V_b = np.mean(vb)
    V_l = np.mean(vl)
    V_r = np.mean(vr)
    V_mean = (V_f + V_b + V_l +V_r) / 4
    vf = cv2.add(vf,(V_mean - V_f))
    vb = cv2.add(vb,(V_mean - V_b))
    vl = cv2.add(vl,(V_mean - V_l))
    vr = cv2.add(vr,(V_mean - V_r))
    front = cv2.merge([hf,sf,vf])
    back = cv2.merge([hb,sb,vb])
    left = cv2.merge([hl,sl,vl])
    right = cv2.merge([hr,sr,vr])
    images = [front,back,left,right]
    images = [cv2.cvtColor(image,cv2.COLOR_HSV2BGR) for image in images]
    return images
   
def stiching(image1,image2,image3,image4, common):
    '''
    #Blending From Here
    mask_img_white = common
    alpha = common
    sides = cv2.add(image2, image4)
    FR_RR = cv2.add(image1, image3)
    
    foreground = FR_RR
    background = sides
    
    
    foreground = foreground.astype(float)
    background = background.astype(float)
    
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    #To here
    '''
    dst= cv2.add(image2, image4)
    dst = cv2.add(dst, image3)
    outImage = cv2.add(dst, image1)
    
    
    return outImage
    
def region_of_interest_front(image):

    square = np.array([[
    (0, 0), (0,180), (640,180),
    (640, 0),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    
def region_of_interest_back(image):

    square = np.array([[
    (0, 310), (0,480), (640,480),
    (640, 310),]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def region_of_interest_side(image):

    square = np.array([[
    (0, 130), (0,290), (640,290),
    (640, 130),]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
    
def max_cluster_slope(Z):

    data = Z
    X = np.array(data).reshape(-1, 1)

    dbscan = DBSCAN(eps=3)
    dbscan.fit(X)

    labels = dbscan.labels_
    label_freq = Counter(labels)
    highest_freq_label = label_freq.most_common(1)[0][0]

    new_lst = [x for x in labels if x == highest_freq_label]

    cluster_elements = [Z[i] for i in range(len(labels)) if labels[i] == highest_freq_label]

    print('max_cluster_slope', (np.mean(cluster_elements)))
    print('labels', new_lst)

    return np.mean(cluster_elements)

def _slope_intercept_visualize(lines, y21, a, final_image_st):
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    slopes = np.round_((lines[:,3] - lines[:,1]) /(lines[:,2] - lines[:,0]),2)
    infinity = np.isinf(slopes)
    n_infinity = sum(infinity)
    print('Number of infinity', n_infinity)
    
    Z = np.round_(np.rad2deg(np.arctan(slopes)),2)
    
    Z = Z[~np.isnan(Z) & ~np.isinf(Z)]
    if n_infinity > 6:
        x = 90
        
    else:
        x = np.median(Z)
    
    #slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]
    #Z = np.rad2deg(np.arctan(slopes))
    
    '''
    lower_percentile = 5
    upper_percentile = 75

    lower_limit, upper_limit = np.percentile(Z, [lower_percentile, upper_percentile])
    ft = (Z > lower_limit) & (Z < upper_limit)
    Z = Z[ft]
    '''
    #x = (np.median(Z))
    print('X',x)
    
    if a==0 and len(Z)!=0:
        x11 = 350
        y11 = 130
        if np.tan(x) == 0:
            x21 = x11
        
        else:
            x21 = int(((y21-y11)+x11*np.tan(np.deg2rad(x)))/np.tan(np.deg2rad(x)))
        
        x0 = np.array([x11,y11])
        x01 = np.array([x21,y21])
        cv2.line(final_image_st, tuple(x0), tuple(x01), (0,0,255), 4)
        if x < 0:
            x = 90-abs(x)
            cv2.putText(final_image_st, 'Slope FR ' + str(round(x,2)) + ' CW', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            x = 90-abs(x)
            cv2.putText(final_image_st, 'Slope FR ' + str(round(x,2)) + ' ACW', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
	
    elif a==1 and len(Z)!=0:
        x11 = 350 
        y11 = 300
        y21 = 400
        if np.tan(x) == 0:
            x21 = x11
         
        else:
            x21 = int(((y21-y11)+x11*np.tan(np.deg2rad(x)))/np.tan(np.deg2rad(x)))
        
        x0 = np.array([x11,y11])
        x01 = np.array([x21,y21])
        cv2.line(final_image_st, tuple(x0), tuple(x01), (0,0,255), 4)
        
        if x < 0:
            x = 90-abs(x)
            cv2.putText(final_image_st, 'Slope RR ' + str(round(x,2)) + ' CW', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            x = 90-abs(x)
            cv2.putText(final_image_st, 'Slope RR ' + str(round(x,2)) + ' ACW', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(final_image_st, 'Slope RR ' +str(round(x,2)), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    else:
    	cv2.putText(final_image_st, 'Front or Rare Lines Not Detected', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
    return 0
 
# define a video capture object

prev_frame_time = 0
new_frame_time = 0


vid_RT = cv2.VideoCapture("v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink")
vid_RR = cv2.VideoCapture("v4l2src device=/dev/video4 ! video/x-raw, width=640, height=480  ! videoconvert ! video/x-raw,format=BGR ! appsink")
vid_LT = cv2.VideoCapture("v4l2src device=/dev/video6 ! video/x-raw, width=640, height=480  ! videoconvert ! video/x-raw,format=BGR ! appsink")
vid_FR = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480  ! videoconvert ! video/x-raw,format=BGR ! appsink")


K=np.array([[230.37912994378186, 0.0, 326.381228901319], [0.0, 230.61814452975904, 236.9152648878691], [0.0, 0.0, 1.0]])
D=np.array([[-0.007844798956048664], [-0.01887864867083691], [0.019987919856503687], [-0.006890329594431897]])

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (640,480),cv2.CV_16SC2)
mask_img_white = mask()


img_bev_m1,img_bev_m2 = mask_1()
car = cv2.imread('./car.png')


while(True):
      
    # Capture the video frame
    # by frame
    retFR, frameFR = vid_FR.read()
    retRT, frameRT = vid_RT.read()
    retLT, frameLT = vid_LT.read()
    retRR, frameRR = vid_RR.read()
    '''
    frameFR = color_balance(frameFR)
    frameRT = color_balance(frameRT)
    frameRR = color_balance(frameRR)
    frameLT = color_balance(frameLT)
    '''
    [frameFR, frameRT, frameRR, frameLT] = luminance_balance([frameFR, frameRT, frameRR, frameLT])
    

# Image with removing Distortion for all cameras

    undistorted_img_FR = undistortion(frameFR,map1, map2)
    
    undistorted_img_RR =  undistortion(frameRR,map1, map2)

    undistorted_img_RT = undistortion(frameRT,map1, map2)
    
    undistorted_img_LT = undistortion(frameLT,map1, map2)

# Homography Matrix
    
    H_FR = np.array([[-7.07806084e-01, -2.33438170e+00, 5.68874674e+02],
 [ 7.79288351e-03, -1.31819362e+00, 3.49425808e+02],
 [-4.38923690e-05, -6.92180391e-03, 1.00000000e+00]])

    H_RT = np.array([[-2.07590604e-02, -2.96689587e+00, 4.91699739e+01],
 [-1.07246093e+00, -2.01773980e+00, 5.78396275e+02],
 [ 6.32894442e-05, -9.52450543e-03, 1.00000000e+00]])
 
    H_RR = np.array([[ 2.08465243e+00, -5.93162997e+00, -4.19885578e+02],
 [-1.59339167e-01, -3.68364873e+00, -4.33454556e+02],
 [-4.19260804e-04, -1.74941873e-02, 1.00000000e+00]])
 
    H_LT = np.array([[-1.00622301e-01, -5.52579047e+00, 8.35377637e+02],
 [ 1.62366336e+00, -3.02940289e+00, -4.05685870e+02],
 [-7.59785044e-04, -1.43052902e-02, 1.00000000e+00]])

    
    # Wrap Prespective using the above Homography Matrixfinal_image_st
    warp_FR = warp(undistorted_img_FR, H_FR)
    warp_FR[240:,:].fill(0)
    
    warp_RT = warp(undistorted_img_RT, H_RT)
    warp_RT[:,:240].fill(0)
    #warp_RT[:,:400].fill(0)
    warp_RR = warp(undistorted_img_RR, H_RR)
    #warp_RR[:600,:].fill(0)
    warp_LT = warp(undistorted_img_LT, H_LT)
    #warp_LT[:,400:].fill(0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    final_image_st = stiching(warp_FR,warp_RT,warp_RR,warp_LT, mask_img_white.astype(float)/255)
    
    
    final_image_st_1  = cv2.bitwise_and(final_image_st,img_bev_m1)
    final_image_st_2  = cv2.bitwise_and(final_image_st,img_bev_m2)
    final_image_st = cv2.addWeighted(final_image_st_1,0.5,final_image_st_2,1,0)
    blank_image = np.zeros((480,640,3), np.uint8)
    
    gray = cv2.cvtColor(img_bev_m2, cv2.COLOR_BGR2GRAY)
    # Define the kernel for dilation
    kernel = np.ones((5,5), np.uint8)
    
    # Eroding the image
    eroded_img = cv2.erode(gray, kernel, iterations=1)
    
    gray = cv2.cvtColor(img_bev_m1, cv2.COLOR_BGR2GRAY)
    eroded_img_1 = cv2.erode(gray, kernel, iterations=1)
    
    warped = final_image_st_2
    warped_1 = final_image_st_1

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(warped, 160, 200, None, 3)
    #dst1 = cv2.Canny(bev_m_1, 100, 120, None, 3)

    warped_1= cv2.cvtColor(warped_1, cv2.COLOR_BGR2GRAY)
    dst_1 = cv2.Canny(warped_1, 160, 200, None, 3)#

    # Find the coordinates of the edges
    edge_coords_1 = np.nonzero(dst_1)

    # Find the coordinates of the edges
    edge_coords = np.nonzero(dst)
    # Create a new black image with one channels
    h, w = dst.shape
    black_img = np.zeros((h, w, 1), dtype=np.uint8)

    black_img_1 = np.zeros((h, w, 1), dtype=np.uint8)

    # Set the pixels at the edge coordinates to white
    black_img[edge_coords] = 255
    black_img_1[edge_coords_1] = 255
    img_dst = cv2.bitwise_and(black_img, eroded_img)
    dst_canny = cv2.Canny(img_dst, 110, 130, None, 3)

    img_dst_1 = cv2.bitwise_and(black_img_1, eroded_img_1)
    dst_canny_1 = cv2.Canny(img_dst_1, 110, 130, None, 3)

    warped = cv2.add(dst_canny,dst_canny_1)
    canny_image = warped
    
    #Creating the ROI for Lane Detection
    canny_image[:, 0:250] = 0
    canny_image[:,450:640] = 0
    canny_image[120:310,300:350].fill(0)
    
    
    # Find the coordinates of the edges
    canny_final = np.nonzero(canny_image)

    # Create a new black image with one channels
    h, w = canny_image.shape
    blank_img = np.zeros((h, w, 3), dtype=np.uint8)
    #blank_img = cv2.add(blank_img, car)
    blank_img[120:310,300:400].fill(255)
    #blank_img[149:243,325:376] = car
    
    # Set the pixels at the edge coordinates to white
    blank_img[canny_final] = 150
    
    
    #lines = cv2.HoughLinesP(canny_image, 2, np.pi/90, 20, np.array([]), minLineLength=17, maxLineGap=4) 
    #image_with_lines = draw_the_lines(blank_image,lines)  
    #Front
    
    lane_image_1  = np.copy(final_image_st)
    cropped_image_1 = region_of_interest_front(canny_image)
    
    lines_1 = cv2.HoughLinesP(cropped_image_1, 2, np.pi/90, 40, np.array([]), minLineLength=15, maxLineGap=5)
    _slope_intercept_visualize(lines_1, 50,0, blank_img)
    
    #Back
    
    lane_image_3  = np.copy(final_image_st)
    cropped_image_3 = region_of_interest_back(canny_image)
        
    lines_3 = cv2.HoughLinesP(cropped_image_3, 2, np.pi/90, 40, np.array([]), minLineLength=15, maxLineGap=5)
    _slope_intercept_visualize(lines_3, 460,1,blank_img)
    
    # Draw line

    #image_with_lines = draw_the_lines(final_image_st,lines_1)
    
    #Side_Distance
    
    lane_image_2  = np.copy(final_image_st)
    cropped_image_2 = region_of_interest_side(canny_image)
    lines_2 = cv2.HoughLinesP(cropped_image_2, 2, np.pi/90, 60, np.array([]), minLineLength=5, maxLineGap=4)
    distance_l = []
    distance_r = []
    d_l = 0
    d_r = 0
        
    # Drawing Left Lane
    if len(lines_2) != 0:
        for line in lines_2:
            d_l,d_r = distance_between_lines(line)
            if int(d_l) != 0:
                distance_l.append(int(d_l))
            
            if int(d_r) != 0:
                distance_r.append(int(d_r))
                
    else:
        cv2.putText(blank_img, 'Lane Not Detection', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 

    if len(distance_l):  
        dl_m =int(np.median(distance_l))
        p22 = np.array([345, 240])
        p23 = np.array([345-dl_m, 240])
        
        cv2.line(blank_img, tuple(p22), tuple(p23), (0,255,0), 3)
        # Add text near the left line
        text_l = str(round(dl_m, 2))
        cv2.putText(blank_img, text_l, (300-dl_m, 238), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
    else:
        cv2.putText(blank_img, 'Left Lane Not Detected', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Drawing Right Lane    
    if len(distance_r) != 0:
        dr_m =int(np.median(distance_r))
        p32 = np.array([345, 240])
        p33 = np.array([345+dr_m, 240])
        
        
        cv2.line(blank_img, tuple(p32), tuple(p33), (0,255,0), 3)
        text_r = str(round(dr_m, 2))
        cv2.putText(blank_img, text_r, (360+dr_m, 238), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 1, cv2.LINE_AA)
        
    else:
        cv2.putText(blank_img, 'Right Lane Not Detected', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 1, cv2.LINE_AA)
          
    
    cv2.circle(blank_img, (350,240), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.putText(blank_img, 'CAR', (320, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
    
    #blank_image = cv2.add(blank_image, car)
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    # converting the fps into integer
    fps = int(fps)
    fps = str(fps)

    cv2.imshow('Line_Images', blank_img)
    
    #cv2.putText(final_image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    #cv2.putText(image_with_lines, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.namedWindow('Stiched Image Final', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Stiched Image Final',1200,1200)
    cv2.putText(final_image_st, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Stiched Image Final', final_image_st)
    #cv2.putText(out_img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Frame Rate per second is :", frame_no)
print(start)
print(end)
  
# After the loop release the cap object
vid_FR.release()
vid_RR.release()
vid_RT.release()
vid_LT.release()
# Destroy all the windows
cv2.destroyAllWindows()
