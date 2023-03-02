from imutils import face_utils
import numpy as np
import dlib
import cv2
import random
import time

#Initialise Webcam 
cap = cv2.VideoCapture(0)
time.sleep(3)  # Added few seconds of sleep to get the camera heated

detector = dlib.get_frontal_face_detector() #Initialise dlib face detector
#Make sure you download the dlib weight file in the same directory
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

#Function to draw circular point on image
def draw_point(img, p, color ) :
    cv2.circle( img, p, 1, color, -1, cv2.LINE_AA, 0 )


# Function to draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
        
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)



while (cap.isOpened()):
# Read webcam feed
    ret, image = cap.read()
# Flipped the image to avoid mirror effect
    image = np.flip(image, axis=1) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape) 
        img_orig = image.copy();
        size = image.shape
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect);

        # Insert points into subdiv
        for p in shape:
            subdiv.insert((p[0],p[1]))

        # Draw delaunay triangles
        draw_delaunay(image, subdiv,  (255, 255, 255));

        # Draw points
        for p in shape:
            draw_point(image, (p[0],p[1]),  (0, 0, 255))

        # Show results
        #out.write(image)
        cv2.imshow("Delaunay Triangle", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
