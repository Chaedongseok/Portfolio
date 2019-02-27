
import numpy as np
import cv2
import time
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points
chess_hor = 9
chess_ver = 6
delaytime = 2000
points_center = np.zeros(2,np.float32)
objp = np.zeros((chess_ver*chess_hor,3), np.float32)
objp2 = np.zeros(((chess_ver-2)*(chess_hor-2),3), np.float32)
objp[:,:2] = np.mgrid[0:chess_hor,0:chess_ver].T.reshape(-1,2)
objp2[:,:2] = np.mgrid[0:(chess_hor-2),0:(chess_ver-2)].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cv2.namedWindow("Left", cv2.WND_PROP_FULLSCREEN)
cv2.namedWindow("Right",cv2.WND_PROP_FULLSCREEN)
#cv2.namedWindow("Left")
#cv2.namedWindow("Right")
cv2.setWindowProperty("Left",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("Right",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

imgLeft = cv2.imread('grayLP.png') # 480*640
imgRight = cv2.imread('grayRP.png')

grayL = cv2.cvtColor(imgLeft,cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgRight,cv2.COLOR_BGR2GRAY) #웹캠으로 본 이미지

cv2.imshow('Right',grayR)
cv2.imshow('Left',grayL)

cv2.imwrite('perImageL.png',grayL)
cv2.imwrite('perImageR.png',grayR)

for i in [1,2]:
##Left------------------------------------------------------------------------
    imgLeft = cv2.imread('perImageL.png')
    imgRight = cv2.imread('Black.png')
    
    cv2.imshow('Left', imgLeft)
    cv2.imshow('Right', imgRight)
    
    cv2.waitKey(delaytime)
    cap = cv2.VideoCapture(0) # 480*640
    ret, frame = cap.read() 
    cap.release()
    
    #cv2.destroyAllWindows()   
    
    # cv2.namedWindow('KnV')
    
    tic = time.time()
    
    img = frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_hor,chess_ver),None)
    print('Ready')
    # Find the chess board corners
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('Operate')
        cv2.cornerSubPix(gray,corners,(chess_hor,chess_ver),(-1,-1),criteria) #코너의 위치를 보정
    
    #Find the conversion point to crop the image(perspective transform)#804(hor)x572(ver) (572,804,3)
        img2 = cv2.imread('perImageL.png')
        grayimg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        perspec_ratio = (1024/768)/(640/480)
        
        Vertical_size = np.shape(img2)[0]
        Horizontal_size = np.shape(img2)[1]
        LeftDown = corners[chess_hor*chess_ver-chess_hor,0,:]
        RightDown = corners[-1,0,:]
        LeftUp = corners[0,0,:]
        RightUp = corners[chess_hor-1,0,:]
        
        Edge_points = [LeftUp,RightUp,LeftDown,RightDown]
        Edge_points = np.asarray(Edge_points)
        
        #For transform, Find Edge Points
        rec_height = np.min([LeftDown[1],RightDown[1]])-np.max([LeftUp[1],RightUp[1]])
        rec_width = np.min([RightUp[0],RightDown[0]])-np.max([LeftUp[0],LeftDown[0]])
        points_center[0] = np.mean([LeftUp[0],LeftDown[0],RightUp[0],RightDown[0]])
        points_center[1] = np.mean([LeftUp[1],LeftDown[1],RightUp[1],RightDown[1]])
        
        # Move to ZeroMean
        Edge_cenzero = Edge_points - points_center
        scale_size = 1.0
        Edge_scaling_CZ = Edge_cenzero*np.min([Vertical_size*scale_size/rec_height,Horizontal_size*scale_size/rec_width])
        Edge_scaling_CZ[:,0] = Edge_scaling_CZ[:,0]/perspec_ratio
        
        Perspective_points = np.float32(Edge_scaling_CZ+[Horizontal_size*1/2,Vertical_size*1/2])
    
        Perspec_trans_points = np.float32([[0,0],[Horizontal_size,0], # The point to be transformed
                                           [0,Vertical_size],[Horizontal_size,Vertical_size]]) #leftDown, RightDown, LeftUp, RightUp
        
        M = cv2.getPerspectiveTransform(Perspective_points,Perspec_trans_points)
        perImage = cv2.warpPerspective(grayimg2,M,(Horizontal_size,Vertical_size))
    
        cv2.imwrite('perImageL.png',perImage)
        #DistanceUp = (np.max(corners[:chess_hor,0,0])-np.min(corners[:chess_hor,0,0]))/(chess_hor-scale_size)
        #DistanceDown = (np.max(corners[chess_hor:,0,0])-np.min(corners[chess_hor:,0,0]))/(chess_hor-scale_size)
        #Vertical_points =corners[:,0,1]
        #Vertical_points_reshape = np.reshape(Vertical_points,[chess_ver,chess_hor])
        #DistanceLeft = (np.max(Vertical_points_reshape[:,0])-np.min(Vertical_points_reshape[:,-1]))/(chess_ver-scale_size)
        #DistanceRight = (np.max(Vertical_points_reshape[:,0])-np.min(Vertical_points_reshape[:,-1]))/(chess_ver-scale_size)
    
    #perspective transform 
        Perspective_points = np.float32([LeftUp,RightUp, # points to transformation
                                         LeftDown,RightDown]) 
        
        NVertical_size = np.shape(perImage)[0]
        NHorizontal_size = np.shape(perImage)[1]
        Perspec_trans_points = np.float32([[0,0],[NHorizontal_size,0], # The point to be transformed
                                           [0,NVertical_size],[NHorizontal_size,NVertical_size]]) #leftDown, RightDown, LeftUp, RightUp
        
        M = cv2.getPerspectiveTransform(Perspective_points,Perspec_trans_points)
        ObjectImage = cv2.warpPerspective(img,M,(Horizontal_size,Vertical_size))
        cv2.imwrite('ObjectImage.png',ObjectImage)
    
    #cv2.destroyAllWindows() 
    imgLeft = cv2.imread('perImageL.png')
    imgRight = cv2.imread('Black.png')
    
    cv2.imshow('Left', imgLeft)
    cv2.imshow('Right', imgRight)
    
    cv2.waitKey(delaytime)
    cap = cv2.VideoCapture(0) # 480*640
    ret, frame = cap.read() 
    cap.release()
    
    #cv2.destroyAllWindows()  
    cv2.imwrite('LeftP.png',frame)
    
    
    ##Right------------------------------------------------------------------------
    imgLeft = cv2.imread('Black.png')
    imgRight = cv2.imread('perImageR.png')
    
    cv2.imshow('Left', imgLeft)
    cv2.imshow('Right', imgRight)
    
    cv2.waitKey(delaytime)
    cap = cv2.VideoCapture(0) # 480*640
    ret, frame = cap.read() 
    cap.release()
    
    #cv2.destroyAllWindows()   
    
    # cv2.namedWindow('KnV')
    
    tic = time.time()
    
    img = frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_hor,chess_ver),None)
    print('Ready')
    # Find the chess board corners
    # If found, add object points, image points (after refining them)
    if ret == True:
    
        print('Operate')
        cv2.cornerSubPix(gray,corners,(chess_hor,chess_ver),(-1,-1),criteria) #코너의 위치를 보정
    
    #Find the conversion point to crop the image(perspective transform)#804(hor)x572(ver) (572,804,3)
        img2 = cv2.imread('perImageR.png')
        grayimg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        perspec_ratio = (1024/768)/(640/480)
        
        Vertical_size = np.shape(img2)[0]
        Horizontal_size = np.shape(img2)[1]
        LeftDown = corners[chess_hor*chess_ver-chess_hor,0,:]
        RightDown = corners[-1,0,:]
        LeftUp = corners[0,0,:]
        RightUp = corners[chess_hor-1,0,:]
        
        Edge_points = [LeftUp,RightUp,LeftDown,RightDown]
        Edge_points = np.asarray(Edge_points)
        
        #For transform, Find Edge Points
        rec_height = np.min([LeftDown[1],RightDown[1]])-np.max([LeftUp[1],RightUp[1]])
        rec_width = np.min([RightUp[0],RightDown[0]])-np.max([LeftUp[0],LeftDown[0]])
        points_center[0] = np.mean([LeftUp[0],LeftDown[0],RightUp[0],RightDown[0]])
        points_center[1] = np.mean([LeftUp[1],LeftDown[1],RightUp[1],RightDown[1]])
        
        # Move to ZeroMean
        Edge_cenzero = Edge_points - points_center
        scale_size = 1.0
        Edge_scaling_CZ = Edge_cenzero*np.min([Vertical_size*scale_size/rec_height,Horizontal_size*scale_size/rec_width])
        Edge_scaling_CZ[:,0] = Edge_scaling_CZ[:,0]/perspec_ratio
        
        Perspective_points = np.float32(Edge_scaling_CZ+[Horizontal_size*1/2,Vertical_size*1/2])
    
        Perspec_trans_points = np.float32([[0,0],[Horizontal_size,0], # The point to be transformed
                                           [0,Vertical_size],[Horizontal_size,Vertical_size]]) #leftDown, RightDown, LeftUp, RightUp
        
        M = cv2.getPerspectiveTransform(Perspective_points,Perspec_trans_points)
        perImage = cv2.warpPerspective(grayimg2,M,(Horizontal_size,Vertical_size))
        cv2.imwrite('perImageR.png',perImage)
        #DistanceUp = (np.max(corners[:chess_hor,0,0])-np.min(corners[:chess_hor,0,0]))/(chess_hor-scale_size)
        #DistanceDown = (np.max(corners[chess_hor:,0,0])-np.min(corners[chess_hor:,0,0]))/(chess_hor-scale_size)
        #Vertical_points =corners[:,0,1]
        #Vertical_points_reshape = np.reshape(Vertical_points,[chess_ver,chess_hor])
        #DistanceLeft = (np.max(Vertical_points_reshape[:,0])-np.min(Vertical_points_reshape[:,-1]))/(chess_ver-scale_size)
        #DistanceRight = (np.max(Vertical_points_reshape[:,0])-np.min(Vertical_points_reshape[:,-1]))/(chess_ver-scale_size)
    
    #perspective transform 
        Perspective_points = np.float32([LeftUp,RightUp, # points to transformation
                                         LeftDown,RightDown]) 
         
        NVertical_size = np.shape(perImage)[0]
        NHorizontal_size = np.shape(perImage)[1]
        Perspec_trans_points = np.float32([[0,0],[NHorizontal_size,0], # The point to be transformed
                                           [0,NVertical_size],[NHorizontal_size,NVertical_size]]) #leftDown, RightDown, LeftUp, RightUp
        
        M = cv2.getPerspectiveTransform(Perspective_points,Perspec_trans_points)
        ObjectImage = cv2.warpPerspective(img,M,(Horizontal_size,Vertical_size))
        cv2.imwrite('ObjectImage.png',ObjectImage)
        
    #cv2.destroyAllWindows() 
                
    imgLeft = cv2.imread('black.png')
    imgRight = cv2.imread('perImageR.png')
    
    cv2.imshow('Left', imgLeft)
    cv2.imshow('Right', imgRight)
    
    cv2.waitKey(delaytime)
    cap = cv2.VideoCapture(0) # 480*640
    ret, frame = cap.read() 
    cap.release()
    
    cv2.destroyAllWindows()  
    cv2.imwrite('RightP.png',frame)
    
    imgLeft = cv2.imread('perImageL.png')
    imgRight = cv2.imread('perImageR.png')      
      
    cv2.imshow('Left', imgLeft)
    cv2.imshow('Right', imgRight)
    
    print('Press')
    cv2.waitKey(delaytime)
    cap = cv2.VideoCapture(0) # 480*640
    ret, frame = cap.read() 
    cap.release()
    
    cv2.imwrite('TotalP.png',frame)
    cv2.destroyAllWindows()  


