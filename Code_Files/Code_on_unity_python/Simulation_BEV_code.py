'''
Universität Siegen
Naturwissenschaftlich-Technische Fakultät
Department Elektrotechnik und Informatik
Lehrstuhl für Regelungs- und Steuerungstechnik

Studienarbeit: Vehicle-Centric All-Around Bird's Eye View Local Mapping
Done By: Bidehyak Subedi, 1593042
         Sourav Poudyal, 1607167
Supervisior: Dr.-Ing. Nasser Gyagenda
'''
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import threading
import time
import math
import statistics
from collections import Counter
from sklearn.cluster import DBSCAN
import argparse


parser = argparse.ArgumentParser(
    description='BEV generator',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '-slope',
    '-s',
    type=str,
    default ='median',
    choices=['median', 'cluster'],
    help='choose between Clustering or median filter to set slope')

args = parser.parse_args()

# The IP address of the machine running Python
HOST = "0.0.0.0"

# The port numbers used for the network communication
PORT1 = 5555 #FT
PORT2 = 5557 #Lt
PORT3 = 5556 #RT 
PORT4 = 5558 #RR

#Homegraphy matrix
H_Front = np.array([[ 1.07925700e+00,  6.09261809e+00, -2.60161731e+01],
                    [-8.67471471e-04,  4.06327717e+00, -1.51819861e+02],
                    [-7.77621391e-06,  1.90725299e-02,  1.00000000e+00]])

H_Right = np.array([[-2.35002754e-01,  4.06414289e+01,  3.68431154e+03],
                    [ 1.12359731e+01,  3.00506236e+01, -3.41145949e+03],
                    [-5.77704629e-04,  1.39711320e-01,  1.00000000e+00]])

H_Left = np.array([[-1.82550719e-01, -2.33310221e+02,  1.61839443e+04],
                  [ 5.35024410e+01, -1.44871891e+02, -1.69284305e+04],
                  [-9.40466687e-04, -6.66967739e-01,  1.00000000e+00]])

H_Rear = np.array([[ 1.10704364e-01, -5.98863379e-01,  2.82972013e+02],
                  [ 3.47819477e-04, -4.29284318e-01,  2.64289244e+02],
                  [ 1.02370047e-06, -1.88186677e-03,  1.00000000e+00]])

#Creating a white image of Size 640*480
img_w = np.zeros([480,640,3],dtype=np.uint8)
img_w.fill(255)

#Using respective homography matrix to perspective transform the white image
#to create mask for each BEV 
bev_f_m = cv2.warpPerspective(img_w, H_Front, (640, 480), flags=cv2.INTER_LINEAR)
bev_r_m = cv2.warpPerspective(img_w, H_Right, (640, 480), flags=cv2.INTER_LINEAR)
bev_l_m = cv2.warpPerspective(img_w, H_Left, (640, 480), flags=cv2.INTER_LINEAR)
bev_b_m = cv2.warpPerspective(img_w, H_Rear, (640, 480), flags=cv2.INTER_LINEAR)

#Generating masks 1 (with common region) and mask 2(without common region) for BEV
fb = cv2.addWeighted(bev_f_m, 1, bev_b_m, 1, 0)
rl = cv2.addWeighted(bev_r_m, 1, bev_l_m, 1, 0)
bev_m_1 = cv2.bitwise_and(fb, rl)
bev_m_2 = cv2.bitwise_xor(fb, rl)

# Applying erosion to masks 1
gray = cv2.cvtColor(bev_m_2, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5), np.uint8)
eroded_img = cv2.erode(gray, kernel, iterations=1)


# Applying erosion to masks 2
gray_1 = cv2.cvtColor(bev_m_1, cv2.COLOR_BGR2GRAY)
kernel_1 = np.ones((5,5), np.uint8)
eroded_img_1 = cv2.erode(gray_1, kernel_1, iterations=1)

# Variables to store the incoming images from each port
img1 = None
img2 = None
img3 = None
img4 = None

#Function to get heading information
def _slope_intercept(lines, y21):

    #list of lines from hough line
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    #Finding the slope of the lines
    slopes = (lines[:,3] - lines[:,1]) /(lines[:,2] - lines[:,0])
    
    if args.slope == 'cluster':
    #using cluster to find the desired slope
        slope = max_cluster_slope(slopes)
    elif args.slope == 'median':
        slope = np.median(slopes)

    row = np.rad2deg(np.arctan(slope))

    st = ''
    #condition for heading information
    if slope > 0:
        row = 90 - row
        # if car rows more than 5 deg then wrong path
        if row > 5: 
            st ='ACW'
        else:
            st =''    
    else:
        row = 90 + row
        if row > 5:
            st = "CW"
        else:
            st = ""
    
    # center point of a car to define 
    # the fixed end of heading line
    x11 = 320 
    y11 = 240

    # finding the coordinate of other end of the heading vector
    # using the slope of the lane line
    x21 = int(((y21-y11)+x11*slope)/slope)
    x0 = np.array([x11,y11])
    x01 = np.array([x21,y21])

    return row, st, x0, x01

#Function to define clustering algorithm
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


# to return the front view region of interest for forward heading
def region_of_interest_front(image):

    square = np.array([[
    (0, 0), (0,180), (640,180),
    (640, 0),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#To visualize the hough line image
def display_lines_(image, lines, c):
    line_image = np.zeros_like(image)
    #print(lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), c, 4)
    return line_image

# to return the back view region of interest for backward heading
def region_of_interest_back(image):

    square = np.array([[
    (0, 270), (0,480), (640,480),
    (640, 270),]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

# to return the side view region of interest for position information
def region_of_interest_side(image):

    square = np.array([[
    (0, 170), (0,270), (640,270),
    (640, 170),]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

#Calculates the shortest distance between two lines, represented by two points each.
def distance_between_lines(line1):

    x1, y1, x2, y2 = line1[0]
    center = 320 

    #to store the distance of a car from left lane and right lane
    distance_l = 0
    distance_r = 0

    #condition to calculate the distances
    if x1 < 340 and x2 < 340:
        distance_l = (abs((x1 - center)) + abs((x2 - center)))/2       
        print('car at ' + str(distance_l) + ' px from left lane')
        
    else:
        distance_r = (abs((x1 - center)) + abs((x2 - center)))/2
        print('car at ' + str(distance_r) + ' px from right lane')


    return distance_l, distance_r
 
class ImageReceiver(BaseHTTPRequestHandler):

    def do_POST(self):
        global img1, img2, img3, img4
        # To Determine which port the data is coming from
        if self.server.server_address[1] == PORT1:
            img1 = self.process_image()

        elif self.server.server_address[1] == PORT2:
            img2 = self.process_image()

        elif self.server.server_address[1] == PORT3:
            img3 = self.process_image()

        elif self.server.server_address[1] == PORT4:
            img4 = self.process_image()
        
        # Check if both images have been received
        if img1 is not None and img2 is not None and img3 is not None and img4 is not None:


            # Transform the front image to BEV
            bev_f = cv2.warpPerspective(img1, H_Front, (640, 480), flags=cv2.INTER_LINEAR)

            # Transform the left image to BEV
            bev_r = cv2.warpPerspective(img2, H_Left, (640, 480), flags=cv2.INTER_LINEAR)

            # Transform the right image to BEV
            bev_l = cv2.warpPerspective(img3, H_Right, (640, 480), flags=cv2.INTER_LINEAR)

            # Transform the rear image to BEV
            bev_b = cv2.warpPerspective(img4, H_Rear, (640, 480), flags=cv2.INTER_LINEAR)

            # generate the stitched image
            fb = cv2.addWeighted(bev_f, 0.5, bev_b, 0.5, 0)
            rl = cv2.addWeighted(bev_r, 0.5, bev_l, 0.5, 0)
            bev = cv2.addWeighted(fb, 2, rl, 2, 0)

            # use the mask to create masked BEV image
            bev_am1 = cv2.bitwise_and(bev_m_1, bev)
            bev_am2 = cv2.bitwise_and(bev_m_2, bev)
            #Add the two masked BEV to generate final BEV image
            bev = cv2.addWeighted(bev_am1,0.53, bev_am2, 1, 0)

            
            '''
            Implementing algorithm to refine the canny edge image 
            at the common region
            '''
            warped = bev_am2
            warped_1 = bev_am1

            #canny edge of masked BEV image
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            dst = cv2.Canny(warped, 160, 200, None, 3)

            warped_1= cv2.cvtColor(warped_1, cv2.COLOR_BGR2GRAY)
            dst_1 = cv2.Canny(warped_1, 160, 200, None, 3)#


            # Find the coordinates of the edges
            edge_coords_1 = np.nonzero(dst_1)
            edge_coords = np.nonzero(dst)

            # Create a new black image with one channels
            h, w = dst.shape
            black_img = np.zeros((h, w, 1), dtype=np.uint8)
            black_img_1 = np.zeros((h, w, 1), dtype=np.uint8)


            # Set the pixels at the edge coordinates to white
            black_img[edge_coords] = 255
            black_img_1[edge_coords_1] = 255

            #eroding part of canny image to remove the line seperating the common region
            img_dst = cv2.bitwise_and(black_img, eroded_img)
            dst_canny = cv2.Canny(img_dst, 110, 130, None, 3)

            img_dst_1 = cv2.bitwise_and(black_img_1, eroded_img_1)
            dst_canny_1 = cv2.Canny(img_dst_1, 110, 130, None, 3)

            #Final canny image
            warped = cv2.add(dst_canny,dst_canny_1)
            #Slicing a region around the car to include only essential edges
            warped[155:285, 280:355].fill(0)
            #Final refined canny image
            canny_image = warped


            #Visualizing the front heading information
            cropped_image_1 = region_of_interest_front(canny_image)
            lines_1 = cv2.HoughLinesP(cropped_image_1, 2, np.pi/180, 100, np.array([]), minLineLength=50, maxLineGap=10)
            r, s, x1, y1 = _slope_intercept(lines_1, 50)

            if args.slope == 'cluster':
                t = '(using clustering)'
                text_ = 'Angle FR: '+str(round(r, 2)) +' deg, ' + s
            if args.slope == 'median':
                t = ''
                text_ = 'Angle FR: '+str(round(r, 2)) +' deg, ' + s
            cv2.putText(bev, t, (80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(bev, 'Heading:', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(bev, text_, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            #Visualizing the side position information
            cropped_image_2 = region_of_interest_side(canny_image)
            lines_2 = cv2.HoughLinesP(cropped_image_2, 2, np.pi/180, 100, np.array([]), minLineLength=2, maxLineGap=5)
            
            #To get the shortest distances of the car for the lane
            distance_l = []
            distance_r = []
            d_l = 0
            d_r = 0
            for line in lines_2:
                d_l,d_r = distance_between_lines(line)
                distance_l.append(d_l)
                distance_r.append(d_r)
            
            dl_m = int(max(distance_l))
            p22 = np.array([320, 240])
            p23 = np.array([320-dl_m, 240])

            dr_m =int(max(distance_r)-95)
            p32 = np.array([320, 240])
            p33 = np.array([320+dr_m, 240])

            #Draw left line for positioning
            cv2.line(bev, tuple(p22), tuple(p23), (255,0,0), 3)
            #Draw right line for positioning
            cv2.line(bev, tuple(p32), tuple(p33), (255,0,0), 3)

            #Calculate line in meter
            text_l = str(round(dl_m, 2))
            t_l = 'Left lane: ' + text_l
            # Add text near the left line
            cv2.putText(bev, 'Position:', (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(bev, t_l, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(bev, text_l, (300-dl_m, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Add text near the right line
            text_r = str(round(dr_m, 2))
            t_r = 'Road center: ' + text_r
            # Add text near the left line
            cv2.putText(bev, t_r, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(bev, text_r, (320+dr_m, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv2.LINE_AA)


            #Visualizing the back heading information
            cropped_image_3 = region_of_interest_back(canny_image)
            lines_3 = cv2.HoughLinesP(cropped_image_3, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=10)
            r, s, x1_, y1_ = _slope_intercept(lines_3, 400)    

            text_ = 'Angle RR: '+str(round(r, 2)) +' deg, ' + s

            cv2.putText(bev, text_, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw line
            cv2.line(bev, tuple(x1_), tuple(y1_), (0,0,255), 4)
            cv2.line(bev, tuple(x1), tuple(y1), (0,0,255), 4)


            # Display BEV image with heading information
            cv2.imshow("Detected Lines_3", bev)
            cv2.namedWindow('Detected Lines_3', cv2.WINDOW_NORMAL)
            # Display the canny edge image
            cv2.imshow("Canny", canny_image)
            cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
            

            #Display the Hough line image if required, options, line_1, lines_2, lines_3
            #cv2.imshow('line_img', display_lines_(bev, lines_1, (200,0,0)))

            #cv2.resizeWindow('Received Image',640,480)

            cv2.waitKey(1)

            # Reset the image variables
            img1 = None
            img2 = None
            img3 = None
            img4 = None

        # Send a response to the sender
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(b'OK')

    def process_image(self):
        # Read the data from the request
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)

        # Convert the byte array to an image
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        return img

def start_server(port):
    with socketserver.TCPServer((HOST, port), ImageReceiver) as httpd:
        httpd.serve_forever()

if __name__ == '__main__':
    # Start the HTTP servers in separate threads
    server1_thread = threading.Thread(target=start_server, args=(PORT1,))
    server1_thread.daemon = True
    server1_thread.start()

    server2_thread = threading.Thread(target=start_server, args=(PORT2,))
    server2_thread.daemon = True
    server2_thread.start()

    server2_thread = threading.Thread(target=start_server, args=(PORT3,))
    server2_thread.daemon = True
    server2_thread.start()

    server2_thread = threading.Thread(target=start_server, args=(PORT4,))
    server2_thread.daemon = True
    server2_thread.start()

    # Start the OpenCV window loop
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)
    cv2.destroyAllWindows()
