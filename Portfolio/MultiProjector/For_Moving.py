
import numpy as np
import cv2

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

imgLeft = cv2.imread('LeftP.png') # 480*640
imgRight = cv2.imread('RightP.png')

grayL = cv2.cvtColor(imgLeft,cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgRight,cv2.COLOR_BGR2GRAY) #웹캠으로 본 이미지
cv2.imshow('Right',grayR)
cv2.imshow('Left',grayL)
cv2.waitKey(delaytime)
print('Ready')
# Find the chess board corners
retL, cornersL = cv2.findChessboardCorners(grayL, (chess_hor,chess_ver),None)
retR, cornersR = cv2.findChessboardCorners(grayR, (chess_hor,chess_ver),None)


# If found, add object points, image points (after refining them)
if retL == True and retR == True:
    #print(corners)
    print('Find Left')
    cv2.cornerSubPix(grayL,cornersL,(chess_hor,chess_ver),(-1,-1),criteria) #코너의 위치를 보정
    cv2.imshow('Left',grayL)
    cv2.waitKey(delaytime)

#    Vertical_size = np.shape(img2)[0]
#    Horizontal_size = np.shape(img2)[1]
    LeftDownL = cornersL[chess_hor*chess_ver-chess_hor,0,:]
    RightDownL = cornersL[-1,0,:]
    LeftUpL = cornersL[0,0,:]
    RightUpL = cornersL[chess_hor-1,0,:]
    
    Edge_points = [LeftUpL,RightUpL,LeftDownL,RightDownL]
    Edge_points = np.asarray(Edge_points)

    print('Find Right')
    cv2.cornerSubPix(grayR,cornersR,(chess_hor,chess_ver),(-1,-1),criteria)
    cv2.imshow('Right',grayR)
    cv2.waitKey(delaytime)
    
    LeftDownR = cornersR[chess_hor*chess_ver-chess_hor,0,:]
    RightDownR = cornersR[-1,0,:]
    LeftUpR = cornersR[0,0,:]
    RightUpR = cornersR[chess_hor-1,0,:]
    
    img2L = cv2.imread('perImageL.png') #projector Left 1024*768
    img2R = cv2.imread('perImageR.png') #projector Right 1024*768
    grayLP = cv2.cvtColor(img2L,cv2.COLOR_BGR2GRAY)
    grayRP = cv2.cvtColor(img2R,cv2.COLOR_BGR2GRAY)
    
    Vertical_size = np.shape(img2L)[0]
    Horizontal_size = np.shape(img2L)[1]
    ZeroSpace = np.zeros([np.size(img2L,axis=0),np.size(img2L,axis=1)])
    ZeroMean = [np.size(img2L,axis=0)/2,np.size(img2L,axis=1)/2]
    
    #scaling ------------------------------------------------------------------
    ZeroSpace = np.zeros([np.size(img2L,axis=0),np.size(img2L,axis=1)],np.uint8)
    ZeroMean = [np.size(img2L,axis=0)/2,np.size(img2L,axis=1)/2]
    ZeroSpace = np.asarray(ZeroSpace)
    ZeroMean = np.asarray(ZeroMean)
    Scale_ratio = (LeftUpL[1]-LeftDownL[1])/(LeftUpR[1]-LeftDownR[1])
    if Scale_ratio > 1:
        print('ScaleL')
        SLeft = cv2.resize(grayLP, (int(round(np.size(grayLP,axis = 1)/Scale_ratio)), int(round(np.size(grayLP,axis = 0)/Scale_ratio))), interpolation=cv2.INTER_CUBIC)
        SRight = img2R
        SimgM = [np.size(SLeft,axis=0)/2,np.size(SLeft,axis=1)/2]
        imgPlus = np.asarray(SLeft)
        MeanDist=ZeroMean-SimgM
        ZeroSpace[int(MeanDist[0]):int(MeanDist[0]+np.size(imgPlus,axis=0)),int(MeanDist[1]):int(MeanDist[1]+np.size(imgPlus,axis=1))] = imgPlus
        grayLP = ZeroSpace
    else:
        Scale_ratio = Scale_ratio
        print('ScaleR')
        SRight = cv2.resize(grayRP, (int(round(np.size(grayRP,axis = 1)*Scale_ratio)), int(round(np.size(grayRP,axis = 0)*Scale_ratio))), interpolation=cv2.INTER_CUBIC)
        SLeft = img2L
        SimgM = [np.size(SRight,axis=0)/2,np.size(SRight,axis=1)/2]
        imgPlus = np.asarray(SRight)
        MeanDist=ZeroMean-SimgM
        ZeroSpace[int(MeanDist[0]):int(MeanDist[0]+np.size(imgPlus,axis=0)),int(MeanDist[1]):int(MeanDist[1]+np.size(imgPlus,axis=1))] = imgPlus
        grayRP = ZeroSpace
        
    cv2.imwrite('ScaleR.png', grayRP)
    cv2.imwrite('ScaleL.png', grayLP)
    
    cv2.imwrite('grayLP.png',grayLP)
    cv2.imwrite('grayRP.png',grayRP)
    
    cv2.imwrite('Scaleimg.png',ZeroSpace)
    
    cv2.imshow('Left', grayLP)
    cv2.imshow('Right', grayRP)
    
    cv2.waitKey(delaytime)
    cap = cv2.VideoCapture(0) # 480*640
    ret, frame = cap.read() 
    cap.release()
    
    cv2.imwrite('Scaled.png',frame)
    cv2.destroyAllWindows() 

    
    
    #Moving image-------------------------------------------------------------- 실제 제작때에는 여러번 for문 사용
    for i in [1,2,3]:
        
        imgLeft = cv2.imread('LeftP.png') # 480*640
        imgRight = cv2.imread('RightP.png')
        
        grayL = cv2.cvtColor(imgLeft,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgRight,cv2.COLOR_BGR2GRAY) #웹캠으로 본 이미지
        
        cv2.imshow('Right',grayR)
        cv2.imshow('Left',grayL)
        cv2.waitKey(delaytime)
        print('Ready')
        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, (chess_hor,chess_ver),None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (chess_hor,chess_ver),None)
        
        
        # If found, add object points, image points (after refining them)
        if retL == True and retR == True:
            #print(corners)
            print('Find Left')
            cv2.cornerSubPix(grayL,cornersL,(chess_hor,chess_ver),(-1,-1),criteria) #코너의 위치를 보정
            cv2.imshow('Left',grayL)
            cv2.waitKey(delaytime)
        
        #    Vertical_size = np.shape(img2)[0]
        #    Horizontal_size = np.shape(img2)[1]
            LeftDownL = cornersL[chess_hor*chess_ver-chess_hor,0,:]
            RightDownL = cornersL[-1,0,:]
            LeftUpL = cornersL[0,0,:]
            RightUpL = cornersL[chess_hor-1,0,:]
            
            Edge_points = [LeftUpL,RightUpL,LeftDownL,RightDownL]
            Edge_points = np.asarray(Edge_points)
        
            print('Find Right')
            cv2.cornerSubPix(grayR,cornersR,(chess_hor,chess_ver),(-1,-1),criteria)
            cv2.imshow('Right',grayR)
            cv2.waitKey(delaytime)
            
            LeftDownR = cornersR[chess_hor*chess_ver-chess_hor,0,:]
            RightDownR = cornersR[-1,0,:]
            LeftUpR = cornersR[0,0,:]
            RightUpR = cornersR[chess_hor-1,0,:]
           
        grayLP = cv2.imread('grayLP.png')
        grayRP = cv2.imread('grayRP.png')
        grayLP = cv2.cvtColor(grayLP,cv2.COLOR_BGR2GRAY)
        grayRP = cv2.cvtColor(grayRP,cv2.COLOR_BGR2GRAY)
        retLL, cornersLL = cv2.findChessboardCorners(grayLP, (chess_hor,chess_ver),None)
        retRR, cornersRR = cv2.findChessboardCorners(grayRP, (chess_hor,chess_ver),None)
        
        if retLL == True and retRR == True:
            print('Moving')
            #3cv2.drawChessboardCorners(grayLP, (chess_hor,chess_ver), cornersLL,retLL)
            #cv2.imshow('Left',grayLP)
            #cv2.waitKey(delaytime)
            cv2.cornerSubPix(grayLP,cornersLL,(chess_hor,chess_ver),(-1,-1),criteria) #코너의 위치를 보정
            #cv2.imshow('Left',grayLP)
            #cv2.waitKey()
            
            cv2.cornerSubPix(grayRP,cornersRR,(chess_hor,chess_ver),(-1,-1),criteria) #코너의 위치를 보정
    
            #cv2.imshow('Left',grayRP)
            #cv2.waitKey()
                    
            LeftDownLL = cornersLL[chess_hor*chess_ver-chess_hor,0,:]
            RightDownLL = cornersLL[-1,0,:]
            LeftUpLL = cornersLL[0,0,:]
            RightUpLL = cornersLL[chess_hor-1,0,:]
                
            LeftDownRR = cornersRR[chess_hor*chess_ver-chess_hor,0,:]
            RightDownRR = cornersRR[-1,0,:]
            LeftUpRR = cornersRR[0,0,:]
            RightUpRR = cornersRR[chess_hor-1,0,:]
            
            ZeroSpace = np.zeros([np.size(img2L,axis=0),np.size(img2L,axis=1)],np.uint8)
            if Scale_ratio > 1: # Left BIG axis 0 가로 axis1 세로
                print('1')
                BetDist_hor = int(round((RightUpL[1]-LeftUpR[1])/(LeftUpL[1]-LeftDownL[1])*(LeftUpLL[1]-LeftDownLL[1]))) # height
                #BetDist_hor = int(round((RightUpLL[0]-LeftUpRR[0])*(1024/640)))
                BetDist_ver = int(round((RightUpL[0]-LeftUpR[0])/(LeftUpL[0]-RightUpL[0])*(LeftUpLL[0]-RightUpLL[0]))) # width
                #BetDist_ver = int(round((RightUpLL[1]-LeftUpRR[1])*(1024/640))) # width
                ######################################################################################################################################
                if BetDist_hor>=0 and BetDist_ver>=0: # 오른쪽아래 왼쪽위로 올려야함
                    BetDist_hor = np.abs(BetDist_hor)
                    BetDist_ver = np.abs(BetDist_ver)
                    ZeroSpace[0:np.size(ZeroSpace,axis=0)-BetDist_hor,0:np.size(ZeroSpace,axis=1)-BetDist_ver] = grayLP[BetDist_hor:,BetDist_ver:]  
                    print('moving!1')
                elif BetDist_hor>=0 and BetDist_ver<0: #오른쪽 위 왼쪽 아래로
                    BetDist_hor = np.abs(BetDist_hor)
                    BetDist_ver = np.abs(BetDist_ver)
                    ZeroSpace[0:np.size(ZeroSpace,axis=0)-BetDist_hor,BetDist_ver:] = grayLP[BetDist_hor:,0:np.size(ZeroSpace,axis=1)-BetDist_ver] 
                    print('moving!2')
                    
                elif BetDist_hor<0 and BetDist_ver>=0: # 왼쪽 아래 오른쪽 위로
                    BetDist_hor = np.abs(BetDist_hor)
                    BetDist_ver = np.abs(BetDist_ver)
                    ZeroSpace[BetDist_hor:,0:np.size(ZeroSpace,axis=1)-BetDist_ver] = grayLP[BetDist_hor:,BetDist_ver:] 
                    print('moving!3')
                    
                else:#왼쪽 위 오른쪽 아래로
                    BetDist_hor = np.abs(BetDist_hor)
                    BetDist_ver = np.abs(BetDist_ver)
                    ZeroSpace[BetDist_hor:,BetDist_ver:] = grayLP[:np.size(ZeroSpace,axis=0)-BetDist_hor,:np.size(ZeroSpace,axis=1)-BetDist_ver] 
                    print('moving!4')                
                grayLP=ZeroSpace
                cv2.imwrite('grayRP.png',grayRP)
                cv2.imwrite('grayLP.png',grayLP)
                
            else: # Right BIG
                print('2')
                BetDist_hor = int(round((LeftUpR[1]-RightUpL[1])/(LeftUpR[1]-LeftDownR[1])*(LeftUpRR[1]-LeftDownRR[1])))# height
                BetDist_ver = int(round((LeftUpR[0]-RightUpL[0])/(LeftUpR[0]-RightUpR[0])*(LeftUpRR[0]-RightUpRR[0])))# width
                if BetDist_hor>=0 and BetDist_ver>=0: # 오른쪽 아래 왼쪽위로 올려야함
                    BetDist_hor = np.abs(BetDist_hor)
                    BetDist_ver = np.abs(BetDist_ver)
                    ZeroSpace[0:np.size(ZeroSpace,axis=0)-BetDist_hor,0:np.size(ZeroSpace,axis=1)-BetDist_ver] = grayRP[BetDist_hor:,BetDist_ver:]  
                    print('moving!1')
                elif BetDist_hor>=0 and BetDist_ver<0: #오른쪽 위 왼쪽 아래로
                    BetDist_hor = np.abs(BetDist_hor)
                    BetDist_ver = np.abs(BetDist_ver)
                    ZeroSpace[0:np.size(ZeroSpace,axis=0)-BetDist_hor,BetDist_ver:] = grayRP[BetDist_hor:,0:np.size(ZeroSpace,axis=1)-BetDist_ver] 
                    print('moving!2')
                    
                elif BetDist_hor<0 and BetDist_ver>=0: # 왼쪽 아래 오른쪽 위로
                    BetDist_hor = np.abs(BetDist_hor)
                    BetDist_ver = np.abs(BetDist_ver)
                    ZeroSpace[BetDist_hor:,0:np.size(ZeroSpace,axis=1)-BetDist_ver] = grayRP[BetDist_hor:,BetDist_ver:] 
                    print('moving!3')
                    
                else:#왼쪽 위 오른쪽 아래로
                    BetDist_hor = np.abs(BetDist_hor)
                    BetDist_ver = np.abs(BetDist_ver)
                    ZeroSpace[BetDist_hor:,BetDist_ver:] = grayRP[:np.size(ZeroSpace,axis=0)-BetDist_hor,:np.size(ZeroSpace,axis=1)-BetDist_ver] 
                    print('moving!4')
                grayRP=ZeroSpace
                cv2.imwrite('grayRP.png',grayRP)
                cv2.imwrite('grayLP.png',grayLP)                    
                
            imgLeft = cv2.imread('grayLP.png')
            imgRight = cv2.imread('grayRP.png')
                
            cv2.imshow('Left', imgLeft)
            cv2.imshow('Right', imgRight)
                
            cv2.waitKey(delaytime)
            cap = cv2.VideoCapture(0) # 480*640
            ret, frame = cap.read() 
            cap.release()
            
            cv2.imwrite('Sampleresult.png',frame)
            cv2.imwrite('Moving.png',ZeroSpace)
            print(BetDist_hor)
            print(BetDist_ver)
            
#Save-----------------------------------------------------------------------------------------------------------------------------------
            imgLeft = cv2.imread('grayLP.png')
            imgRight = cv2.imread('black.png')
            
            cv2.imshow('Left', imgLeft)
            cv2.imshow('Right', imgRight)
            
            cv2.waitKey(delaytime)
            cap = cv2.VideoCapture(0) # 480*640
            ret, frame = cap.read() 
            cap.release()
            
            cv2.destroyAllWindows()  
            cv2.imwrite('LeftP.png',frame)

            imgLeft = cv2.imread('black.png')
            imgRight = cv2.imread('grayRP.png')
            
            cv2.imshow('Left', imgLeft)
            cv2.imshow('Right', imgRight)
            
            cv2.waitKey(delaytime)
            cap = cv2.VideoCapture(0) # 480*640
            ret, frame = cap.read() 
            cap.release()
            
            cv2.destroyAllWindows()  
            cv2.imwrite('RightP.png',frame)

            imgLeft = cv2.imread('grayLP.png')
            imgRight = cv2.imread('grayRP.png')
            
            cv2.imshow('Left', imgLeft)
            cv2.imshow('Right', imgRight)
            
            cv2.waitKey(delaytime)
            cap = cv2.VideoCapture(0) # 480*640
            ret, frame = cap.read() 
            cap.release()
            
            cv2.destroyAllWindows()  
            cv2.imwrite('Total.png',frame)

'''        
        #Combine image---------------------------------------------------------
        retLC, cornersLC = cv2.findChessboardCorners(grayLP, (chess_hor,chess_ver),None)
        retRC, cornersRC = cv2.findChessboardCorners(grayRP, (chess_hor,chess_ver),None)      
    
        if retLC == True and retRC == True:
            print('Combine')
            
            cv2.cornerSubPix(grayLP,cornersLC,(chess_hor,chess_ver),(-1,-1),criteria) #코너의 위치를 보정

            #cv2.imshow('Left',grayLP)
            #cv2.waitKey()
            
            cv2.cornerSubPix(grayRP,cornersRC,(chess_hor,chess_ver),(-1,-1),criteria) #코너의 위치를 보정

            #cv2.imshow('Left',grayRP)
            #cv2.waitKey()
            
            LeftUpLC = cornersLC[chess_hor*chess_ver-chess_hor,0,:]
            LeftDownLC = cornersLC[-1,0,:]
            RightUpLC = cornersLC[0,0,:]
            RightDownLC = cornersLC[chess_hor-1,0,:]
                
            RightDownRC = cornersRC[chess_hor*chess_ver-chess_hor,0,:]
            LeftDownRC = cornersRC[-1,0,:]
            LeftDownRC = cornersRC[0,0,:]
            LeftUpRC = cornersRC[chess_hor-1,0,:]
            GrayLP=grayLP
            cv2.imwrite('GRAYLP.png',grayLP)

            GrayRP=grayRP
            grayLP[:,int(np.min([RightUpLC[0],RightDownLC[0]])):] = grayRP[:,int(np.max([LeftUpRC[0],LeftDownRC[0]])):
                int(np.max([LeftUpRC[0],LeftDownRC[0]])+np.size(grayLP[:,int(np.min([RightUpLC[0],RightDownLC[0]])):],axis=1))]
            grayRP[:,:int(np.max([LeftUpRC[0],LeftDownRC[0]]))] = grayLP[:,int(np.min([RightUpLC[0],RightDownLC[0]])-
                          np.size(grayRP[:,:int(np.max([LeftUpRC[0],LeftDownRC[0]]))],axis=1)):int(np.min([RightUpLC[0],RightDownLC[0]]))]
    
            cv2.imwrite('0807result.png',grayRP)
            '''

        

            
