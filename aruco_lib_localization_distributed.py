

import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
from copy import deepcopy


from threading import Thread, Lock


class LocalizationTracker():
    def __init__(self,
                id_to_find,
                marker_size,
                src,
                camera_matrix,
                camera_distortion,
                camera_size=[640,480],
                show_video=False
                ):
        
        """
        The initialization of the class for localization...
        
        """
        
        
        #---Aruco settings for perspective transform mnap
        #------------ Define Reference Tags
        self.id_1  = 50
        self.id_2  = 60
        self.marker_size  = 10 #- [cm]

        
        #---Aruco settings for position
        self.id_to_find     = id_to_find
        self.marker_size    = marker_size
        self._show_video    = show_video
        self.marker_size_warp  = 10 #4.29 # 0.0429 #- [cm]
        
        
        #--- agentmera setting
        self.src = src
        self._camera_matrix = camera_matrix
        self._camera_distortion = camera_distortion
        
        #--- Lopp of the funcitons
        self.is_detected    = False
        self._kill          = False
        
        #--- 180 deg rotation matrix around the x axis
        self._R_flip      = np.zeros((3,3), dtype=np.float32)
        self._R_flip[0,0] = 1.0
        self._R_flip[1,1] =-1.0
        self._R_flip[2,2] =-1.0

        #--- Define the aruco dictionary
        self._aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self._parameters  = aruco.DetectorParameters_create()


        #--- Capture the videocamera (this may also be a video or a picture)
        self._cap = cv2.VideoCapture(self.src)
        #-- Set the camera size as the one it was calibrated with
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_size[1])

        ret, frame = self._cap.read()
        print('frame ', frame.shape)
        
        #-- Font for the text in the image
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self._t_read      = time.time()
        self._t_detect    = self._t_read
        self._t_pos_detect = self._t_detect
        
        self.fps_read    = 0.0
        self.fps_detect  = 0.0   
        self.fps_pos_detect  = 0.0 
        
        #----------------THREADING------------------\
        self.frame = self._cap.read()
        self.started_frame = False
        self.read_frame_lock = Lock()
        
        self.ref_track_started = False
        self.read_ref_lock = Lock()
        self.ref_perspective = self.frame
        self.ref_frame =  self.frame
        self.read_ref_frame_lock = Lock()
        
        self.pos_frame = self.ref_frame
        self.target_pos = (False, self.frame, [[], []])
        self.pos_track_started = False
        self.read_pos_lock = Lock()
        self.read_pos_frame_lock = Lock()
        
        #-----OBSTACLES----------------------\
        self.obs_track_started = False
        self.read_obs_lock = Lock()
        self.obs_frame =  self.frame
        self.read_obs_frame_lock = Lock()

        
        
        
        
        
        
    def _rotationMatrixToEulerAngles(self,R):
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    
        def isRotationMatrix(R):
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6        
        assert (isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def _update_fps_read(self):
        t           = time.time()
        self.fps_read    = 1.0/(t - self._t_read+0.000000001)
        self._t_read      = t
        
    def _update_fps_detect(self):
        t           = time.time()
        self.fps_detect  = 1.0/(t - self._t_detect+0.000000001)
        self._t_detect      = t    
        
    def _update_fps_pos_detect(self):
        t           = time.time()
        self.fps_pos_detect  = 1.0/(t - self._t_detect+0.000000001)
        self._t_pos_detect      = t  

    def stop(self):
        self._kill = True
        
    def start_pos_track(self):
        if self.pos_track_started:
            print('reference tracking already started')
            return None
        
        self.pos_track_started = True
        self.thread_pos = Thread(target=self.update_pos, 
                                 kwargs={'verbose':True, 'loop':False})
        self.thread_pos.start()
        return self   


    def update_pos(self, verbose = True, loop = True):
        """
        This function is the responsible for the dectection of the position of the markers
        in the map it could be the robot or the target. 
        """
        while self.pos_track_started:
            
            found_ref, frame = self.read_ref()
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            gray2 = frame.copy()
            #OpenCV stores color images in Blue, Green, Red

            aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
            parameters  = aruco.DetectorParameters_create()

            #-- Find all the aruco markers in the image
            corners, ids, rejected = aruco.detectMarkers(image=gray, 
                                                         dictionary=aruco_dict,
                                                         parameters=parameters,
                                                         cameraMatrix=self._camera_matrix, 
                                                         distCoeff=self._camera_distortion
                                                        )

            #definition of the return items of this function
            marker_position = []
            marker_pixel_position = []
            pos_marker_found = False

            #return (pos_marker_found, gray , marker_position)
            if ids is not None and ids[0] == self.id_to_find:#change
                pos_marker_found = True
                self._update_fps_pos_detect()
                ret = aruco.estimatePoseSingleMarkers(corners, 
                                                      self.marker_size_warp, #change
                                                      self._camera_matrix, 
                                                      self._camera_distortion)

                #getting the center of the marker as the final position
                corner_0 = corners[0][0]

                center_x = (corner_0[0]+corner_0[2])/2
                center_y = (corner_0[1]+corner_0[3])/2

                marker_pixel_position  = center_x

                
                #-- Unpack the output, get only the first
                rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

                
                marker_position = [tvec[0],  tvec[1], tvec[2]]
                
                ###### --------------FRAME REFERENCE UPDATE-------------------------
                    
                if verbose:
                    #drawing the marker in the map
                    aruco.drawDetectedMarkers(gray2, corners)
                    #drawing the posion in the map
                    cv2.circle(gray2, tuple(center_x), 2, (0, 0, 255), -1)
                    cv2.circle(gray2, tuple(center_x), 3, (0, 0, 255), -1)
                    #adding the position infors in the map
                    
                    marker_position_text = "Position Marker x=%4.3f  y=%4.3f  z=%4.3f | fps pos = %.2f"%(tvec[0], tvec[1], tvec[2], self.fps_pos_detect )
                    if found_ref:
                        cv2.putText(gray2, marker_position_text, (0, 100), self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(gray2, marker_position_text, (0, 100), self.font, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
                        cv2.putText(gray2, 'PERSPECTIVE is not CALCULATED', (0, 150), self.font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        
                        
            else:
                #self.target_pos = 
                #gray2 = frame.copy()
                #---------------FRAME INFOS
                if verbose:
                    cv2.putText(gray2, 'Position marker found not', (0, 100),  self.font, 0.5, (0, 0,255 ), 1, cv2.LINE_AA)
            if verbose:
                pass
                #print( "Nothing detected - pos fps = %.0f"%self.fps_pos_detect)
            
            if not loop:
                self.read_pos_frame_lock.acquire()
                self.pos_frame =(found_ref, gray2)
                
                self.read_pos_frame_lock.release()

                self.read_pos_lock.acquire()
                self.target_pos = (pos_marker_found, frame, [marker_position, marker_pixel_position])
                self.read_pos_lock.release()
                #return (pos_marker_found, frame, [marker_position, marker_pixel_position])
                self._update_fps_detect()
                #print( "Ref FPS = %.2f"%(self.fps_detect))
                #break
                
    def read_pos(self) :
        self.read_pos_lock.acquire()
        pos_target = self.target_pos#.copy()
        self.read_pos_lock.release()
        
        return pos_target
    
    def read_pos_frame(self) :
        self.read_pos_frame_lock.acquire()
        #pos_frame = self.pos_frame#.copy()
        pos_frame = deepcopy(self.pos_frame)
        self.read_pos_frame_lock.release()
        
        return pos_frame
    
    def stop_pos(self) :
        #self.cap.release()
        self.pos_track_started = False
        self.thread_pos.join()
        #self._cap.release()
        print('Position Thread terminated')    
    

    #---REFERENCE FRAME---------------------------
    def start_ref_track(self):
        if self.ref_track_started:
            print('reference tracking already started')
            return None
        
        self.ref_track_started = True
        self.thread_ref = Thread(target=self.update_ref, kwargs={'verbose':True, 'loop':False})
        self.thread_ref.start()
        return self
    
    def update_ref(self, loop=True, verbose=False, show_video=None):
        while self.ref_track_started:
            self._kill = False
            if show_video is None: show_video = self._show_video

            
            #preparing the reuturn itiem from the function
            ref_marker_found = False
            pos_marker_found = False 
            x = y = z = 0

            positions = []

            while not self._kill:
                #-- Read the camera frame
                #ret, frame = self._cap.read()
                
                ret, frame  = self.read_frame()

                result = frame.copy()
                gray_ref = frame.copy()

                self._update_fps_read()

                #-- Convert in gray scale
                gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                #OpenCV stores color images in Blue, Green, Red

                #-- Find all the aruco markers in the image
                corners, ids, rejected = aruco.detectMarkers(image=gray,
                                                             dictionary=self._aruco_dict, 
                                parameters=self._parameters,
                                cameraMatrix=self._camera_matrix, 
                                distCoeff=self._camera_distortion)

                #............................../|\

                if ids is not None and  np.where(ids == self.id_1)[0] >= 0 and np.where(ids == self.id_2)[0] >= 0 :
                    #print('ids', ids)
                    ref_marker_found = True
                    index_corner_1 = np.where(ids == self.id_1)[0][0]
                    index_corner_2 = np.where(ids == self.id_2)[0][0]


                    ret = aruco.estimatePoseSingleMarkers(corners, 
                                                          self.marker_size, 
                                                          self._camera_matrix, 
                                                          self._camera_distortion)


                    #-- Unpack the output, get only the first
                    rvec, tvec = ret[0][index_corner_2,0,:], ret[1][index_corner_2,0,:]
                    
                   
                    #____________________________________|||||||| PERSPECTIVE TRANSFORM |||||||||_________________
                    #the perspective transform part

                    vals = corners[index_corner_1][0]
                    vals_2 = corners[index_corner_2][0]

                    pts1 = np.float32([vals[0], vals[1], vals_2[3], vals_2[2]])

                    pts2 = np.float32([[5, 5], [55, 5], [5, 125], [55, 125] ])

                    matrix = cv2.getPerspectiveTransform(pts1, pts2)

                    result = cv2.warpPerspective(frame, matrix, (640, 480))
                    
                    ###### --------------FRAME REFERENCE UPDATE-------------------------
                    aruco.drawDetectedMarkers(gray_ref, corners)
                    str_position = "Refference Marker Position x=%4.0f  y=%4.0f  z=%4.0f | fps ref = %.2f"%(tvec[0], tvec[1], tvec[2], self.fps_detect)
                    cv2.putText(gray_ref, str_position, (0, 100),  self.font, 0.5, (11, 198,255 ), 1, cv2.LINE_AA)
                else:
                    result = frame
                    
                    #---------FRAME INFO
                    cv2.putText(gray_ref, 'No reference found', (0, 50),  self.font, 0.5, (0, 0,240 ), 1, cv2.LINE_AA)
                #if not loop: return(ref_marker_found, pos_marker_found, frame, positions)
                #----------------------THREADING--------------------------
                if not loop:
                    
                    self.read_ref_frame_lock.acquire()
                    self.ref_frame = (ref_marker_found, gray_ref)
                    self.read_ref_frame_lock.release()
                    
                    self.read_ref_lock .acquire()
                    self.ref_perspective =(ref_marker_found,  result)#(ref_marker_found, pos_marker_found, frame, positions)
                    self.read_ref_lock .release()
                    
                    self._update_fps_detect()
                    #print( "Ref FPS = %.2f"%(self.fps_detect))
                    break
                
    def read_ref(self) :
        self.read_ref_lock.acquire()
        ref_perspective = self.ref_perspective#.copy()
        self.read_ref_lock.release()
        
        
        return ref_perspective
    
    def read_ref_frame(self) :
        self.read_ref_frame_lock.acquire()
        ref_frame_info = deepcopy(self.ref_frame)#.copy()
        self.read_ref_frame_lock.release()
        
        
        return ref_frame_info
    
    def stop_ref(self) :
        #self.cap.release()
        self.ref_track_started = False
        self.thread_ref.join()
        #self._cap.release()
        print('REFERENCE FRAME THREAD TERMINATED')
        
        
    def __exit__(self, exc_type, exc_value, traceback) :
        self._cap.release()
        
        
    #------INITIAL FRAME---------------------------------------------
    def start_frame(self) :
        if self.started_frame :
            print( "already started frame!!")
            return None
        self.started_frame = True
        self.thread_frame = Thread(target=self.update_frame, args=())
        self.thread_frame.start()
        return self

    def update_frame(self) :
        while self.started_frame :
            (grabbed, frame) = self._cap.read()
            self.read_frame_lock.acquire()
            #self.grabbed, 
            self.frame = (grabbed, frame)
            self.read_frame_lock.release()
            
        
    def read_frame(self) :
        self.read_frame_lock.acquire()
        frame = self.frame#tuple(list(self.frame))#.copy()
        self.read_frame_lock.release()
        
        self._update_fps_read()
        
        #print('read_frame FPS = %.2f '%self.fps_read)
        
        return frame
    
    def stop_frame(self) :
        #self.cap.release()
        self.started_frame = False
        self.thread_frame.join()
        self._cap.release()
        print('Stop Frame Thread')
        
    #------OBSTICALES FRAME-----------------------------------------------
    def start_obs_frame(self) :
        if self.obs_track_started :
            print( "already started obtacles frame!!")
            return None
        self.obs_track_started = True
        self.thread_obs = Thread(target=self.update_obs_frame, args=())
        self.thread_obs.start()
        return self

    def update_obs_frame(self) :
        while self.started_frame :
            ret, frame = self.read_ref()  
            #result = frame.copy()

            lower_range = np.array([32, 100, 100])
            upper_range = np.array([80, 255, 255])


            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_range, upper_range)

            #cv2.imshow('image', frame)
            #cv2.imshow('mask', mask)
            #coord = cv2.findNonZero(mask)
            #coord = np.reshape(coord, (-1, 2)) 
            #print('mask ', mask.shape)
            #return mask
            
            self.read_obs_frame_lock.acquire()
            self.obs_frame = (ret, mask)
            self.read_obs_frame_lock.release()
            
        
    def read_obs_frame(self) :
        self.read_obs_frame_lock.acquire()
        obstacles = self.obs_frame#tuple(list(self.frame))#.copy()
        self.read_obs_frame_lock.release()
        
        #self._update_fps_read()
                
        return obstacles
    
    def stop_obs_frame(self) :
        #self.cap.release()
        self.started_frame = False
        self.thread_obs.join()
        print('OBSTACLES THREAD TERMINATED')
    
    