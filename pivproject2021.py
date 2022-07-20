import cv2 as cv2
import numpy as np
import sys
import random
import glob
from skimage.exposure import match_histograms

# name: select_points
# inputs: 
#        - u1: u coordinates of each aruco marker vertex in the template image
#        - u2: u coordinates of each aruco marker vertex in the original frame
#        - v1: v coordinates of each aruco marker vertex in the template image
#        - v2: v coordinates of each aruco marker vertex in the original frame 
#        - id2: list of all aruco marker ids detected in the original frame
# outputs:
#        - u1n: u coordinates of each detected aruco marker vertex in the template image
#        - u2n: u coordinates of each detected aruco marker vertex in the original frame 
#        - v1n: v coordinates of each detected aruco marker vertex in the template image
#        - v2n: v coordinates of each detected aruco marker vertex in the original frame 
# description: returns a new set of u and v coordinates, selecting only the ones belonging to the detected aruco markers
#              vertices in the original frame
def select_points(u1,u2,v1,v2,id2):

    u1n = np.float32([-1 for i in range(len(id2)*4)])
    u2n = np.float32([-1 for i in range(len(id2)*4)])
    v1n = np.float32([-1 for i in range(len(id2)*4)])
    v2n = np.float32([-1 for i in range(len(id2)*4)])
    aux = 0

    for i in id2:

        u1n[aux*4] = u1[i*4]
        u1n[aux*4+1] = u1[i*4+1]
        u1n[aux*4+2] = u1[i*4+2]
        u1n[aux*4+3] = u1[i*4+3]

        u2n[aux*4] = u2[i*4]
        u2n[aux*4+1] = u2[i*4+1]
        u2n[aux*4+2] = u2[i*4+2]
        u2n[aux*4+3] = u2[i*4+3]

        v1n[aux*4] = v1[i*4]
        v1n[aux*4+1] = v1[i*4+1]
        v1n[aux*4+2] = v1[i*4+2]
        v1n[aux*4+3] = v1[i*4+3]

        v2n[aux*4] = v2[i*4]
        v2n[aux*4+1] = v2[i*4+1]
        v2n[aux*4+2] = v2[i*4+2]
        v2n[aux*4+3] = v2[i*4+3]

        aux +=1

    return u1n, u2n, v1n, v2n


# name: detect_arucos
# inputs: 
#        - imagePath: string with the path to an image
#        - ArUCo_dict: string with the aruco tag type
# outputs:
#        - u1: u coordinates of each detected aruco marker vertex in the image
#        - v1: v coordinates of each detected aruco marker vertex in the image
#        - ids: set of ids corresponding to the detected aruco markers
# description: detects each aruco marker present in an image, returning the set of detected arucos marker (id) and 
#              the u and v coordinates of every vertex
def detect_arucos(imagePath, ArUCo_dict):
    # define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,   # the one that we use
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    # load the input image from disk and resize it
    #print("[INFO] loading image...")
    image = cv2.imread(imagePath) # load our input image
    #image = imutils.resize(image, width=600) # resize our image to a width of 600 pixels

    # verify that the supplied ArUCo tag exists and is supported by
    # OpenCV
    if ARUCO_DICT.get(ArUCo_dict, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
        ARUCO_DICT))
        sys.exit(0)

    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[ArUCo_dict]) # load the ArUco dictionary
    arucoParams = cv2.aruco.DetectorParameters_create() # instantiate our ArUco detector parameters
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
    parameters=arucoParams)
    u1 = np.float32([-1 for i in range(32)])
    v1 = np.float32([-1 for i in range(32)])

    # The cv2.aruco.detectMarkers results in a 3-tuple of:
    # corners: The (x, y)-coordinates of our detected ArUco markers
    # ids: The identifiers of the ArUco markers (i.e., the ID encoded in the marker itself)
    # rejected: A list of potential markers that were detected but ultimately rejected due 
    # to the code inside the marker not being able to be parsed

    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)   clockwise basically
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # the u and v are stored like this: [u0_tl,u0_tr,u0_br,u0_bl,u1_tl,u1_tr,...]
            # every aruco marker occupies 4 positions (from the index 4*aruco_marker_id to 4*aruco_marker_id + 3) 
            # (the first corresponds to its top left vertex, the second to its top right vertex,
            # the third to its bottom right vertex and the forth to its bottom left vertex)
            u1[4*markerID] = topLeft[0]
            u1[4*markerID+1] = topRight[0]
            u1[4*markerID+2] = bottomRight[0]
            u1[4*markerID+3] = bottomLeft[0]
            v1[4*markerID] = topLeft[1]
            v1[4*markerID+1] = topRight[1]
            v1[4*markerID+2] = bottomRight[1]
            v1[4*markerID+3] = bottomLeft[1]
    return [u1 , v1, ids]


# name: get_homography
# inputs: 
#        - u1: u coordinates of each aruco marker vertex in the template frame
#        - u2: u coordinates of each aruco marker vertex in the original image
#        - v1: v coordinates of each aruco marker vertex in the template frame
#        - v2: v coordinates of each aruco marker vertex in the original image
# outputs:
#        - H: homgraphy matrix
#        - error: error associated with the homography matrix
# description: computes the homography matrix from a given set of u and v values (H, error)
def get_homography(u1,v1,u2,v2): 

    #compute the homography H
    A = []
    for i in range(len(u1)):
        A+= [[u2[i], v2[i], 1, 0, 0, 0, -u1[i]*u2[i], -u1[i]*v2[i], -u1[i]]]
        A+= [[0, 0, 0, u2[i], v2[i], 1, -v1[i]*u2[i], -v1[i]*v2[i], -v1[i]]]

    U,S,V = np.linalg.svd(A)
    H = np.reshape(V[-1],(3,3))
    H = H/H[-1][-1]
    #re-project the points and obtain the error
    u12 = np.divide((np.matmul([H[0][i] for i in range(len(H))],([u2,v2,[1 for i in range(len(u2))]]))),(np.matmul([H[2][i] for i in range(len(H))],[u2,v2,[1 for i in range(len(u2))]])))
    v12 = np.divide((np.matmul([H[1][i] for i in range(len(H))],([u2,v2,[1 for i in range(len(u2))]]))),(np.matmul([H[2][i] for i in range(len(H))],[u2,v2,[1 for i in range(len(u2))]])))
    error = np.sum([((u1[i]-u12[i])**2+(v1[i]-v12[i])**2) for i in range(len(u1))])
    return H, error



# name: task1
# inputs: 
#        - path_to_template: string with the path to the template image
#        - path_to_image_input_folder: string with the path to the input folder
#        - path_to_output_folder: string with the path to the output folder
# outputs:
# description: executes the first task of the piv project
def task1(path_to_template, path_to_image_input_folder, path_to_output_folder):
    print("teste")
    u1,v1, i1=detect_arucos(path_to_template, "DICT_7X7_50")

    number_of_frames=0


    all_files = glob.glob(path_to_image_input_folder+"/*")
    print("teste")
    print(all_files)
    #template = cv2.imread(all_files[0])
    #cv2.imshow('teste', template)
    number_of_frames = len(all_files)

    for aux in range(number_of_frames):
        u2, v2, i2=detect_arucos(all_files[aux], "DICT_7X7_50")
        u1n, u2n, v1n, v2n = select_points(u1,u2,v1,v2,i2)
        frame = cv2.imread(all_files[aux],  cv2.IMREAD_UNCHANGED)
        template = cv2.imread(path_to_template,  cv2.IMREAD_UNCHANGED)
        h, status = get_homography(u1n,v1n,u2n,v2n) 
        warped_img = cv2.warpPerspective(frame,h, (template.shape[1],template.shape[0]))
        remove_hand_and_save(path_to_output_folder, aux, warped_img, template)
    return


# name: remove_hand_and_save
# inputs: 
#        - path_to_output_folder: string with the path to the output folder
#        - iteration: number of the current frame 
#        - frame: current frame
#        - template: template image
# outputs:
# description: used only for the first task of this project, this function removes the pixels corresponding to a hand
# and replaces them with a background image (either the template or the previous frame). An erosion + dilation process 
# is also used to remove artifacts and improve the overall quality of the resulting image alongisde a non-local means denoising step.
def remove_hand_and_save(path_to_output_folder,iteration, frame, template):
    # if this isnt the first frame, then the background image can be the previous frame
    if iteration > 0:
        background = cv2.imread(path_to_output_folder+'/rgb_'+f'{(iteration-1):04d}.jpg',  cv2.IMREAD_UNCHANGED)
        img_with_hand_removed = hand_removal(frame,background)
        kernel = np.ones((5,5), np.uint8)
        img_with_hand_removed = cv2.erode(img_with_hand_removed, kernel, iterations=1)
        img_with_hand_removed = cv2.dilate(img_with_hand_removed, kernel, iterations=1)
        img_with_hand_removed = cv2.fastNlMeansDenoisingColored(img_with_hand_removed,None,10,10,3,21)
        cv2.imwrite(path_to_output_folder+'/rgb_'+f'{iteration:04d}.jpg',img_with_hand_removed)
    # else, the background image has to be the template
    else:
        background = template
        img_with_hand_removed = hand_removal(frame,background)
        kernel = np.ones((5,5), np.uint8)
        img_with_hand_removed = cv2.erode(img_with_hand_removed, kernel, iterations=1)
        img_with_hand_removed = cv2.dilate(img_with_hand_removed, kernel, iterations=1)
        img_with_hand_removed = cv2.fastNlMeansDenoisingColored(img_with_hand_removed,None,10,10,3,21)
        cv2.imwrite(path_to_output_folder+'/rgb_'+f'{iteration:04d}.jpg',img_with_hand_removed)
    return

# name: hand_removal
# inputs: 
#        - frame: current frame
#        - background: background image
# outputs:
#        - no_hand: current frame without the hand
# description: used only for the first task of this project, this function removes the pixels corresponding to a hand
# and replaces them with a background image (either the template or the previous frame).
def hand_removal(frame, background):
    # Background
    bck = background
    # Image
    img = frame

    min_HSV = np.array([0, 20, 20], dtype = "uint8")
    max_HSV = np.array([33, 255, 255], dtype = "uint8")

    # We are converting the image from RGB into the HSV colour space
    #print(cv2.COLOR_BGR2HSV)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # We have created a mask for the hand by selecting a possible range of HSV colors that it can have:
    mask = cv2.inRange(imgHSV, min_HSV, max_HSV)

    # We are slicing the Hand
    imask = mask>0
    hand = np.zeros_like(img, np.uint8)
    #imgHSV = cv2.resize(imgHSV, (400,600))
    #cv2.imshow("hand",imgHSV)
    #cv2.waitKey(0)
    hand[imask] = img[imask]

    # Creates the transparent hand image by first extracting the background without the input image with the hand, 
    # and then extracting the area corresponding to the foreground object (hand) from the background image and adding these two images:
    bck_hand = cv2.bitwise_and(bck, bck, mask=imask.astype(np.uint8))
    no_hand = img.copy()
    no_hand = cv2.bitwise_and(no_hand, no_hand, mask=(np.bitwise_not(imask)).astype(np.uint8))
    no_hand = no_hand + bck_hand

    #cv2.imwrite("rgb_.png", no_hand)
    return no_hand


# name: our_sift
# inputs: 
#        - templategr_query: template image converted to the grayscale color space
#        - framegr_train: current frame converted to the grayscale color space
# outputs:
#        - kp1: keypoints found using the sift algorithm for the template image
#        - kp2: keypoints found using the sift algorithm for the current frame
#        - good: array containing all found good matches, using a Lowe's ratio test
# description: this function is used to obtain the keypoints in the template image and in the current frame.
# This is done by first applying the sift algorithm to grayscale versions of the current frame and corresponding template, 
# in order to extract keypoints and corresponding descriptors, and then matching these features by using a Flann based matcher.
# The matches are then assessed by a Lowe's ratio test, in order to identify the best matches
def our_sift(templategr_query, framegr_train): #receives the image in gray scale
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(templategr_query,None)
    kp2, des2 = sift.detectAndCompute(framegr_train,None)
    # match with flann
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test. 
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
        
    print('good_match:'+str(len(good)))
    return (kp1, kp2, good, matches)

# name: ransac
# inputs: 
#        - u1: u coordinates of the good matches found in the template image
#        - v1: v coordinates of the good matches found in the template image
#        - u2: u coordinates of the good matches found in the frame
#        - v2: v coordinates of the good matches found in the frame
#        - K: number of maximum iterations used for the ransac
#        - delta: thresholding term used to determine if a given homography error is acceptable
# outputs:
#        - u1max: u coordinates of the inliers found in the template image
#        - v1max: v coordinates of the inliers found in the template image
#        - u2max: u coordinates of the inliers found in the frame
#        - v2max: v coordinates of the inliers found in the frame
# description: extracts the coordinates of the features that result in a small homography error by 
# running the ransac algorithm
def ransac(u1,v1,u2,v2,K,delta):
    #select 4 random points
    inmax=[]
    found = False  # Condition to only get, at most, 50 inliers 
    k=0
    while k<K and not found:
        ind=random.sample(range(1,len(u1)), 4) #temos de selecionar os pontos correspondentes no frame e template
        u1h=[u1[ind[0]],u1[ind[1]],u1[ind[2]],u1[ind[3]]]
        v1h=[v1[ind[0]],v1[ind[1]],v1[ind[2]],v1[ind[3]]]
        u2h=[u2[ind[0]],u2[ind[1]],u2[ind[2]],u2[ind[3]]]
        v2h=[v2[ind[0]],v2[ind[1]],v2[ind[2]],v2[ind[3]]]     
        H, error =get_homography(u1h, v1h, u2h, v2h)
        inliers=[]
        u12 = np.divide((np.matmul([H[0][i] for i in range(len(H))],([u2,v2,[1 for i in range(len(u2))]]))),(np.matmul([H[2][i] for i in range(len(H))],[u2,v2,[1 for i in range(len(u2))]])))
        v12 = np.divide((np.matmul([H[1][i] for i in range(len(H))],([u2,v2,[1 for i in range(len(u2))]]))),(np.matmul([H[2][i] for i in range(len(H))],[u2,v2,[1 for i in range(len(u2))]])))
        j=0
        #calculating the error associated with each pixel
        for j in range(len(u1)):
            er = ((u1[j]-u12[j])**2+(v1[j]-v12[j])**2)**(1/2)
            if er<delta:
                inliers+=[j]
        #updating the list with the maximum inliers
        if len(inliers)>len(inmax):
            inmax=inliers
        if len(inliers)>=150:
            found = True
        k+=1
    print('n_inliers:'+ str(len(inmax)))
    u1max=[u1[i] for i in inmax]
    v1max=[v1[i] for i in inmax]
    u2max=[u2[i] for i in inmax]
    v2max=[v2[i] for i in inmax]
    return (u1max, v1max, u2max, v2max)


# name: task2
# inputs: 
#        - path_to_template: string with the path to the template image
#        - path_to_image_input_folder: string with the path to the input folder
#        - path_to_output_folder: string with the path to the output folder
# outputs:
# description: executes the second task of the piv project
def task2(path_to_template, path_to_image_input_folder, path_to_output_folder):
    number_of_frames=0
    all_files = glob.glob(path_to_image_input_folder+"/*")
    number_of_frames = len(all_files)
    h = np.array([])
    MIN_MATCH_COUNT = 15
    for aux in range(number_of_frames):
        print('n_frames:'+str(aux))
        frame = cv2.imread(all_files[aux])
        framegr=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        template = cv2.imread(path_to_template)
        templategr=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        (kp1, kp2, good, matches)=our_sift(templategr, framegr)    
        print(len(matches))    
        if len(good)>MIN_MATCH_COUNT: #performs ransac if the good matches are enough (bigger than 20)
            u1 = np.float32([ kp1[m.queryIdx].pt[0] for m in good ]) #template
            v1 = np.float32([ kp1[m.queryIdx].pt[1] for m in good ]) #template
            u2 = np.float32([ kp2[m.trainIdx].pt[0] for m in good ]) #frame
            v2 = np.float32([ kp2[m.trainIdx].pt[1] for m in good ]) #frame
            (u1in, v1in, u2in, v2in)=ransac(u1, v1, u2, v2, 1000, 5)
            if (len(u1in)> 14) or (len(u1in)>= 4 and h.size == 0):  #performs the homography if the inliers are enough (bigger than 14 or
                                                              #bigger than 4, if the a homography has not been computed before)
                h, status = get_homography(u1in,v1in,u2in,v2in)
                warped_img = cv2.warpPerspective(frame,h, (template.shape[1],template.shape[0]))
                print('error:'+str(status/len(u1in)))
                cv2.imwrite(path_to_output_folder+'/rgb_'+f'{aux:04d}.jpg',warped_img)
            else: # performs the homography with the previous frame, since there weren't enough inliers
                print ("Not enough inlier are found, try with previous frame")
                if aux==0:
                    aux=1
                    warped_img=template
                else:
                    frame_previous = cv2.imread(all_files[aux-1])
                    framegr_previous=cv2.cvtColor(frame_previous,cv2.COLOR_BGR2GRAY)
                    (kp1, kp2, good, matches)=our_sift(framegr_previous, framegr)                        
                    print('good_match:'+str(len(good)))
                    if len(good)>MIN_MATCH_COUNT:
                        u1 = np.float32([ kp1[m.queryIdx].pt[0] for m in good ]) #template
                        v1 = np.float32([ kp1[m.queryIdx].pt[1] for m in good ]) #template
                        u2 = np.float32([ kp2[m.trainIdx].pt[0] for m in good ]) #frame
                        v2 = np.float32([ kp2[m.trainIdx].pt[1] for m in good ]) #frame
                        (u1in, v1in, u2in, v2in)=ransac(u1, v1, u2, v2, 1000, 5)
                        h_next, status = get_homography(u1in,v1in,u2in,v2in) 
                        warped_img = cv2.warpPerspective(frame,h_next, (frame_previous.shape[1],frame_previous.shape[0]))
                        warped_img2 = cv2.warpPerspective(warped_img,h, (template.shape[1],template.shape[0]))
                        print('error:'+str(status/len(u1in)))
                        cv2.imwrite(path_to_output_folder+'/rgb_'+f'{aux:04d}.jpg',warped_img2)
                    else:
                        print ("Homography not possible with previous frame")
                        #cv2.imwrite(path_to_output_folder+'/rgb_'+f'{aux:04d}.jpg',warped_img)
        
        else: # performs the homography with the previous frame, since there weren't enough good matches
            if aux==0:
                    aux=1
                    warped_img=template
            else:
                print ("Not enough matches are found")
                frame_previous = cv2.imread(all_files[aux-1])
                framegr_previous=cv2.cvtColor(frame_previous,cv2.COLOR_BGR2GRAY)
                (kp1, kp2, good, matches)=our_sift(framegr_previous, framegr)
                if len(good)>MIN_MATCH_COUNT:
                    u1 = np.float32([ kp1[m.queryIdx].pt[0] for m in good ]) #template
                    v1 = np.float32([ kp1[m.queryIdx].pt[1] for m in good ]) #template
                    u2 = np.float32([ kp2[m.trainIdx].pt[0] for m in good ]) #frame
                    v2 = np.float32([ kp2[m.trainIdx].pt[1] for m in good ]) #frame
                    (u1in, v1in, u2in, v2in)=ransac(u1, v1, u2, v2, 1000, 5)
                    h_next, status = get_homography(u1in,v1in,u2in,v2in) 
                    warped_img = cv2.warpPerspective(frame,h_next, (frame_previous.shape[1],frame_previous.shape[0]))
                    warped_img2 = cv2.warpPerspective(warped_img,h, (template.shape[1],template.shape[0]))
                    print('error:'+str(status/len(u1in)))
                    cv2.imwrite(path_to_output_folder+'/rgb_'+f'{aux:04d}.jpg',warped_img2)
                else: #impossible to perform homography, copies previous output
                    print ("Homography not possible with previous frame")
                    cv2.imwrite(path_to_output_folder+'/rgb_'+f'{aux:04d}.jpg',warped_img)
    return


# name: task4
# inputs: 
#        - path_to_template: string with the path to the template image
#        - path_to_image_input_folder1: string with the path to one of the input folders
#        - path_to_image_input_folder2: string with the path to one of the input folders
#        - path_to_output_folder: string with the path to the output folder
# outputs:
# description: executes the forth task of the piv project
def task4(path_to_template, path_to_image_input_folder1, path_to_image_input_folder2, path_to_output_folder):
    frames = [] # array to store the frames for the median filter
    all_files1 = glob.glob(path_to_image_input_folder1+"/*")
    number_of_frames1 = len(all_files1)
    all_files2 = glob.glob(path_to_image_input_folder2+"/*")
    number_of_frames2 = len(all_files2)    
    number_of_frames=min(number_of_frames1,number_of_frames2) #only runs the code while there are images on both folders
    MIN_MATCH_COUNT=20
    aux=0
    while aux<number_of_frames: 
        print('n_frames:'+str(aux))
        frame1 = cv2.imread(all_files1[aux])
        frame1gr=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame2 = cv2.imread(all_files2[aux])
        frame2gr=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        template = cv2.imread(path_to_template)
        templategr=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        warped_img=template
        (kp1_1, kp2_1, good_1)=our_sift(templategr, frame1gr)     
        (kp1_2, kp2_2, good_2)=our_sift(templategr, frame2gr) 
        #computes the homography between the template and frames from the 1st folder 
        if len(good_1)>MIN_MATCH_COUNT: #performs ransac if the good matches are enough (bigger than 20)
            u1_1 = np.float32([ kp1_1[m.queryIdx].pt[0] for m in good_1 ]) #template
            v1_1 = np.float32([ kp1_1[m.queryIdx].pt[1] for m in good_1 ]) #template
            u2_1 = np.float32([ kp2_1[m.trainIdx].pt[0] for m in good_1 ]) #frame
            v2_1 = np.float32([ kp2_1[m.trainIdx].pt[1] for m in good_1 ]) #frame
            (u1in_1, v1in_1, u2in_1, v2in_1)=ransac(u1_1, v1_1, u2_1, v2_1, 1000, 5)
            if len(u1in_1)> 14: #performs the homography if the inliers are enough (bigger than 14)
                h_1, status = get_homography(u1in_1,v1in_1,u2in_1,v2in_1)
                warped_img_1 = cv2.warpPerspective(frame1,h_1, (template.shape[1],template.shape[0]))
                print('error:'+str(status/len(u1in_1)))
            else: # performs the homography with the previous frame, since there weren't enough inliers
                print ("Not enough inlier are found, try with previous frame")
                if aux==0:
                    aux=1
                    warped_img_1=template
                else: 
                    frame1_previous = cv2.imread(path_to_image_input_folder1+'/rgb_'+f'{(aux):04d}.jpg')
                    frame1gr_previous=cv2.cvtColor(frame1_previous,cv2.COLOR_BGR2GRAY)
                    (kp1_1, kp2_1, good_1)=our_sift(frame1gr_previous, frame1gr)                        
                    print('good_match:'+str(len(good_1)))
                    if len(good_1)>MIN_MATCH_COUNT:
                        u1_1 = np.float32([ kp1_1[m.queryIdx].pt[0] for m in good_1 ]) #template
                        v1_1 = np.float32([ kp1_1[m.queryIdx].pt[1] for m in good_1 ]) #template
                        u2_1 = np.float32([ kp2_1[m.trainIdx].pt[0] for m in good_1 ]) #frame
                        v2_1 = np.float32([ kp2_1[m.trainIdx].pt[1] for m in good_1 ]) #frame
                        (u1in_1, v1in_1, u2in_1, v2in_1)=ransac(u1_1, v1_1, u2_1, v2_1, 1000, 5)
                        h_next_1, status = get_homography(u1in_1,v1in_1,u2in_1,v2in_1) 
                        warped_img = cv2.warpPerspective(frame1,h_next_1, (frame1_previous.shape[1],frame1_previous.shape[0]))
                        warped_img_1 = cv2.warpPerspective(warped_img,h_1, (template.shape[1],template.shape[0]))
                        print('error:'+str(status/len(u1in_1)))
                    else:
                        print ("Homography not possible with previous frame")

        elif len(good_1)<=MIN_MATCH_COUNT: # performs the homography with the previous frame, since there weren't enough good matches
            print ("Not enough matches are found")
            if aux==0:
                aux=1
                warped_img_1=template
            else:
                frame1_previous = cv2.imread(path_to_image_input_folder1+'/rgb_'+f'{(aux):04d}.jpg')
                frame1gr_previous=cv2.cvtColor(frame1_previous,cv2.COLOR_BGR2GRAY)
                (kp1_1, kp2_1, good_1)=our_sift(frame1gr_previous, frame1gr)
                if len(good_1)>MIN_MATCH_COUNT:
                    u1_1 = np.float32([ kp1_1[m.queryIdx].pt[0] for m in good_1 ]) #template
                    v1_1 = np.float32([ kp1_1[m.queryIdx].pt[1] for m in good_1 ]) #template
                    u2_1 = np.float32([ kp2_1[m.trainIdx].pt[0] for m in good_1 ]) #frame
                    v2_1 = np.float32([ kp2_1[m.trainIdx].pt[1] for m in good_1 ]) #frame
                    (u1in_1, v1in_1, u2in_1, v2in_1)=ransac(u1_1, v1_1, u2_1, v2_1, 1000, 5)
                    h_next_1, status = get_homography(u1in_1,v1in_1,u2in_1,v2in_1) 
                    warped_img = cv2.warpPerspective(frame1,h_next_1, (frame1_previous.shape[1],frame1_previous.shape[0]))
                    warped_img_1 = cv2.warpPerspective(warped_img,h_1, (template.shape[1],template.shape[0]))
                    print('error:'+str(status/len(u1in_1)))
                else: #impossible to perform homography, copies previous output
                    print ("Homography not possible with previous frame")
        # %% same but for the other dataset
        if len(good_2)>MIN_MATCH_COUNT: #performs ransac if the good matches are enough (bigger than 20)
            u1_2 = np.float32([ kp1_2[m.queryIdx].pt[0] for m in good_2 ]) #template
            v1_2 = np.float32([ kp1_2[m.queryIdx].pt[1] for m in good_2 ]) #template
            u2_2 = np.float32([ kp2_2[m.trainIdx].pt[0] for m in good_2 ]) #frame
            v2_2 = np.float32([ kp2_2[m.trainIdx].pt[1] for m in good_2 ]) #frame
            (u1in_2, v1in_2, u2in_2, v2in_2)=ransac(u1_2, v1_2, u2_2, v2_2, 1000, 5)
            if len(u1in_2)> 14: #performs the homography if the inliers are enough (bigger than 14)
                h_2, status = get_homography(u1in_2,v1in_2,u2in_2,v2in_2)
                warped_img_2 = cv2.warpPerspective(frame2,h_2, (template.shape[1],template.shape[0]))
                print('error:'+str(status/len(u1in_2)))
            else: # performs the homography with the previous frame, since there weren't enough inliers
                print ("Not enough inlier are found, try with previous frame")
                if aux==0:
                    aux=1
                    warped_img_2=template
                else:
                    frame2_previous = cv2.imread(path_to_image_input_folder2+'/rgb_'+f'{(aux):04d}.jpg')
                    frame2gr_previous=cv2.cvtColor(frame2_previous,cv2.COLOR_BGR2GRAY)
                    (kp1_2, kp2_2, good_2)=our_sift(frame2gr_previous, frame2gr)                        
                    print('good_match:'+str(len(good_2)))
                    if len(good_2)>MIN_MATCH_COUNT:
                        u1_2 = np.float32([ kp1_2[m.queryIdx].pt[0] for m in good_2 ]) #template
                        v1_2 = np.float32([ kp1_2[m.queryIdx].pt[1] for m in good_2 ]) #template
                        u2_2 = np.float32([ kp2_2[m.trainIdx].pt[0] for m in good_2 ]) #frame
                        v2_2 = np.float32([ kp2_2[m.trainIdx].pt[1] for m in good_2 ]) #frame
                        (u1in_2, v1in_2, u2in_2, v2in_2)=ransac(u1_2, v1_2, u2_2, v2_2, 1000, 5)
                        #print(u1in, v1in, u2in, v2in)
                        #h_previous=h
                        h_next_2, status = get_homography(u1in_2,v1in_2,u2in_2,v2in_2) 
                        warped_img = cv2.warpPerspective(frame2,h_next_2, (frame2_previous.shape[1],frame2_previous.shape[0]))
                        warped_img_2 = cv2.warpPerspective(warped_img,h_2, (template.shape[1],template.shape[0]))
                        print('error:'+str(status/len(u1in_2)))
                        #cv2.imwrite(path_to_output_folder+'/rgb_'+f'{aux:04d}.jpg',warped_img2)
                    else:
                        print ("Homography not possible with previous frame")

        elif len(good_2)<=MIN_MATCH_COUNT: # performs the homography with the previous frame, since there weren't enough good matches
            print ("Not enough matches are found")
            if aux==0:
                aux=1
                warped_img_2=template
            else:
                frame2_previous = cv2.imread(path_to_image_input_folder2+'/rgb_'+f'{(aux):04d}.jpg')
                frame2gr_previous=cv2.cvtColor(frame2_previous,cv2.COLOR_BGR2GRAY)
                (kp1_2, kp2_2, good_2)=our_sift(frame2gr_previous, frame2gr)
                if len(good_2)>MIN_MATCH_COUNT:
                    u1_2 = np.float32([ kp1_2[m.queryIdx].pt[0] for m in good_2 ]) #template
                    v1_2 = np.float32([ kp1_2[m.queryIdx].pt[1] for m in good_2 ]) #template
                    u2_2 = np.float32([ kp2_2[m.trainIdx].pt[0] for m in good_2 ]) #frame
                    v2_2 = np.float32([ kp2_2[m.trainIdx].pt[1] for m in good_2 ]) #frame
                    (u1in_2, v1in_2, u2in_2, v2in_2)=ransac(u1_2, v1_2, u2_2, v2_2, 1000, 5)
                    h_next_2, status = get_homography(u1in_2,v1in_2,u2in_2,v2in_2) 
                    warped_img = cv2.warpPerspective(frame2,h_next_2, (frame2_previous.shape[1],frame2_previous.shape[0]))
                    warped_img_2 = cv2.warpPerspective(warped_img,h_2, (template.shape[1],template.shape[0]))
                    print('error:'+str(status/len(u1in_2)))
                else:
                    print ("Homography not possible with previous frame")
        
        #blending the two images obtained, giving them the same weight
        dst = cv2.addWeighted(warped_img_1, 0.5, warped_img_2, 0.5, 0)
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV) #Convert to hsv color system´
        h,s,v = cv2.split(hsv) #Divided into each component
        # match the histogram of the result to the template to reduce the excess brightness
        res = cv2.equalizeHist(v)
        hsv_2=cv2.merge([h,s,res])
        dst = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR) #Convert to bgr color system
        matched = match_histograms(dst, template)
        matched = match_histograms(matched, template)
        frames.append(matched) # store frame for median filter
        if ((aux+1)/20) >= 1: # if 20 frames have been stored            
            # compute the median of the 20 frames
            result2 = np.median(frames, axis=0).astype(dtype=np.uint8)
            frames.pop(0) # remove the first frame from the frames array, since he has already been computed 
            # save the frame
            cv2.imwrite(path_to_output_folder+'/rgb_'+f'{(aux-19):04d}.jpg',result2)
        aux+=1
    for temp in range(19):
        cv2.imwrite(path_to_output_folder+'/rgb_'+f'{(aux-18+temp):04d}.jpg',result2)
    return



# name: menu
# inputs: 
#        - task: string with the path to the template image
#        - path_to_template: string with the path to the input folder
#        - path_to_output_folder: string with the path to the output folder
#        - arg1: path to the image input folder (for task 1,2,3) or path to the images of camera 1 (for task 4)
#        - arg2: path to the images of camera 2 (for task 4)
# outputs:
# description: selects the correct task
def menu(task, path_to_template, path_to_output_folder,arg1,arg2):
    if task == "1": print("Task 1 - Paper Tablet with Template with Aruco Markers"), task1(path_to_template,arg1, path_to_output_folder)
    elif task == "2": print("Task 2 - Paper Tablet with Template without Aruco Markers"), task2(path_to_template,arg1, path_to_output_folder)
    elif task == "3": print("Task 3 - Paper Tablet with RGB and Depth Cameras"), print("Not to be implemented, try 1, 2 or 4")
    elif task == "4": print("Task 4 - Paper Tablet with Two RGB Cameras"), task4(path_to_template, arg1, arg2, path_to_output_folder)
    else: print("Task",task, "does not exist, please insert a digit between 1 and 4")
    return



if __name__ == "__main__":
      try:
         task = sys.argv[1]
         path_to_template = sys.argv[2]
         path_to_output_folder = sys.argv[3]
         arg1 = sys.argv[4]
         arg2 = -1
         if task == "4":
             arg2 = sys.argv[5]
         menu(task,path_to_template,path_to_output_folder,arg1,arg2)
      except IndexError:
          print("ERROR: INVALID CALL! Please call the papertablet program by using the following command \"python pivproject2021.py task some/dir/template.jpg  some/dir/output some/dir/input\"")

 
