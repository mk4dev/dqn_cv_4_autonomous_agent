camera_id = 1 #set id camera here


# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from kivy.uix.label import Label 

#+++++++++++++++++++++++++Image processing++++++++++++

import numpy as np
import cv2

#+++++++++++++++++++ Memory management
from guppy import hpy
import gc



# Importing the Dqn module 
from ai_dqn import Dqn

# Importing the localization module
from aruco_lib_localization_distributed import LocalizationTracker

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0 # the total number of points in the last drawing
length = 0 # the length of the last drawing

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9
action2rotation = [0, 20, -20] # action = 0 => no rotation, action = 1 => rotate 20 degres, action = 2 => rotate -20 degres
last_reward = 0 # initializing the last reward
scores = [] # initializing the mean score curve (sliding window of the rewards) with respect to time

# Initializing the map
first_update = True # using this trick to initialize the map only once

sand = np.zeros((640,480)) 
def init():
    #global sand # sand is an array that has as many cells as our graphic interface has pixels. Each cell has a one if there is sand, 0 otherwise.
    global goal_x # x-coordinate of the goal (where the agent has to go, that is the airport or the downtown)
    global goal_y # y-coordinate of the goal (where the agent has to go, that is the airport or the downtown)
    sand = np.zeros((longueur,largeur)) # initializing the sand array with only zeros
    print('(longueur,largeur)', (longueur,largeur))
    
    goal_x = 20 # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the agent gets bad reward if it touches the wall)
    goal_y = largeur - 20 # the goal to reach is at the upper left of the map (y-coordinate)
    global first_update
    first_update = False # trick to initialize the map only once
    
    

# Initializing the last distance
last_distance = 0

# Creating the agent class (to understand "NumericProperty" and "ReferenceListProperty", see kivy tutorials: https://kivy.org/docs/tutorials/pong.html)

class Agent(Widget):

    angle = NumericProperty(0) # initializing the angle of the agent (angle between the x-axis of the map and the axis of the agent)
    rotation = NumericProperty(0) # initializing the last rotation of the agent (after playing the action, the agent does a rotation of 0, 20 or -20 degrees)
    velocity_x = NumericProperty(0) # initializing the x-coordinate of the velocity vector
    velocity_y = NumericProperty(0) # initializing the y-coordinate of the velocity vector
    velocity = ReferenceListProperty(velocity_x, velocity_y) # velocity vector
    sensor1_x = NumericProperty(0) # initializing the x-coordinate of the first sensor (the one that looks forward)
    sensor1_y = NumericProperty(0) # initializing the y-coordinate of the first sensor (the one that looks forward)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) # first sensor vector
    sensor2_x = NumericProperty(0) # initializing the x-coordinate of the second sensor (the one that looks 30 degrees to the left)
    sensor2_y = NumericProperty(0) # initializing the y-coordinate of the second sensor (the one that looks 30 degrees to the left)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y) # second sensor vector
    sensor3_x = NumericProperty(0) # initializing the x-coordinate of the third sensor (the one that looks 30 degrees to the right)
    sensor3_y = NumericProperty(0) # initializing the y-coordinate of the third sensor (the one that looks 30 degrees to the right)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y) # third sensor vector
    signal1 = NumericProperty(0) # initializing the signal received by sensor 1
    signal2 = NumericProperty(0) # initializing the signal received by sensor 2
    signal3 = NumericProperty(0) # initializing the signal received by sensor 3

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
      
        #print('agent.pos', (Vector(*self.velocity) + self.pos) )
        #self.pos =pos# Vector(*self.velocity) + self.pos # updating the position of the agent according to its last position and velocity
        self.rotation = rotation # getting the rotation of the agent
        self.angle = self.angle + self.rotation # updating the angle
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos # updating the position of sensor 1
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos # updating the position of sensor 2
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos # updating the position of sensor 3
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400. # getting the signal received by sensor 1 (density of sand around sensor 1)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400. # getting the signal received by sensor 2 (density of sand around sensor 2)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400. # getting the signal received by sensor 3 (density of sand around sensor 3)
        if self.sensor1_x > longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10: # if sensor 1 is out of the map (the agent is facing one edge of the map)
            self.signal1 = 1. # sensor 1 detects full sand
        if self.sensor2_x > longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10: # if sensor 2 is out of the map (the agent is facing one edge of the map)
            self.signal2 = 1. # sensor 2 detects full sand
        if self.sensor3_x > longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10: # if sensor 3 is out of the map (the agent is facing one edge of the map)
            self.signal3 = 1. # sensor 3 detects full sand

class Ball1(Widget): # sensor 1 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball2(Widget): # sensor 2 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass
class Ball3(Widget): # sensor 3 (see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)
    pass


class Goal(Widget): # Goal to reach
    pass


# Creating the game class (to understand "ObjectProperty", see kivy tutorials: kivy https://kivy.org/docs/tutorials/pong.html)

class Game(Widget):
    
    def __init__(self):
        super().__init__()
        
        self._t_game  = time.time()
        self.fps_game = 0.0
        #--- Define Tag
        id_to_find  = 60#72
        marker_size  = 10 #- [cm]

        #--- Get the camera calibration path
        calib_path  = "./cam_01/vx1000/" #"./cam_01/sm_g4/"#

        camera_matrix   = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
        camera_distortion   = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',') 
        
        self.aruco_tracker = LocalizationTracker(id_to_find=1, 
                                           marker_size=10, 
                                           show_video=True,
                                           src = camera_id,
                                           camera_matrix=camera_matrix, 
                                           camera_distortion=camera_distortion)
        
        #INITIALIZE THREADS ******* PLEASE BE SURE IF YES
        """"""
        self.aruco_tracker.start_frame()
        self.aruco_tracker.start_ref_track()
        self.aruco_tracker.start_pos_track()
        self.aruco_tracker.start_obs_frame()
        
        self.show_ref_frame = False
        self.show_pos_frame = False
        self.show_obs_frame = False
        
        self.tex = Texture.create(size=(640, 480), colorfmt='rgba')
        
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    agent = ObjectProperty(None) # getting the agent object from our kivy file
    ball1 = ObjectProperty(None) # getting the sensor 1 object from our kivy file
    ball2 = ObjectProperty(None) # getting the sensor 2 object from our kivy file
    ball3 = ObjectProperty(None) # getting the sensor 3 object from our kivy file
    
    goal = ObjectProperty(None)

    def serve_agent(self): # starting the agent when we launch the application
        self.agent.center = self.center # the agent will start at the center of the map
        self.agent.velocity = Vector(6, 0) # the agent will start to go horizontally to the right with a speed of 6
        
    def _update_fps_game(self):
        t           = time.time()
        self.fps_game    = 1.0/(t - self._t_game)
        self._t_game      = t
    
    #@profile
    def update(self, dt): # the big update function that updates everything that needs to be updated at each discrete time t when reaching a new state (getting new signals from the sensors)

        global brain # specifying the global variables (the brain of the agent, that is our AI)
        global last_reward # specifying the global variables (the last reward)
        global scores # specifying the global variables (the means of the rewards)
        global last_distance # specifying the global variables (the last distance from the agent to the goal)
        global goal_x # specifying the global variables (x-coordinate of the goal)
        global goal_y # specifying the global variables (y-coordinate of the goal)
        global longueur # specifying the global variables (width of the map)
        global largeur # specifying the global variables (height of the map)
        
        #global sand 
        
        
        #title = Label(text=('Hello world'),font_size=str(12) + 'sp', markup=True)
                          
        #self.main_text.text = str('Hello world') #not working
        
        
        
        """
        goal_x = postions[1][0]#20 # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the agent gets bad reward if it touches the wall)
        goal_y = postions[1][1]#largeur - 20
        """
        longueur = self.width # width of the map (horizontal edge)
        largeur = self.height # height of the map (vertical edge)
        
        
        
        #(ref_m, pos_m,frame, postions ) = self.aruco_tracker.track(verbose=True, loop=False)
        (pos_m,frame, postions ) = self.aruco_tracker.read_pos()
        ref_m = False
        if pos_m == True:
            ref_m = True
        #cv2.imshow('Position Frame', frame)
        
        
        if self.show_ref_frame:
            found_frame, frame  = self.aruco_tracker.read_frame()
            found_frame_info, frame_info = self.aruco_tracker.read_ref_frame()
            cv2.imshow('Frame Original', frame)
            cv2.imshow('Frame Original with infos', frame_info)
            #break
            
        if self.show_pos_frame:
            found_perspective, perspective = self.aruco_tracker.read_ref()
            found_frame_pos, pos_frame = self.aruco_tracker.read_pos_frame()
            cv2.imshow('Perspective', perspective)
            cv2.imshow('Perspective with position', pos_frame)
            
        if self.show_obs_frame:
            found_frame_obs, obs_frame = self.aruco_tracker.read_obs_frame()
            cv2.imshow('Perspective with obstacles', obs_frame)

        global sand
            
            
        if ref_m and pos_m:
            goal_x = int(postions[1][0])
            goal_y = int(largeur -postions[1][1])
            
            
            
        #print(dir(self))
            
        #print('kivy ', self.width, self.height , 'positions', postions )
        
        if first_update : # trick to initialize the map only once
            init()
            print('===? ', first_update)
            #first_update=False # redudent
            
        
      
        found_frame_obs, obs_frame = self.aruco_tracker.read_obs_frame()
        
        #cv2.imshow("CV2 Image",  sand, )#cv2.rotate(cv2.ROTATE_90_COUNTERCLOCKWISE)
        if found_frame_obs: #ref_m and pos_m:
            found_frame_obs, obs_frame = self.aruco_tracker.read_obs_frame()
            
            
            sand = sand+cv2.rotate(obs_frame, cv2.ROTATE_90_CLOCKWISE)
            
       
        with self.children[3].canvas.before:
            self.children[3].canvas.before.clear()
            self.opacity = 0.5
            
            if found_frame_obs: #ref_m and pos_m:
                found_frame_obs, obs_frame = self.aruco_tracker.read_obs_frame()

                #sand = obstacles.copy()

                buf1 = cv2.flip(obs_frame, 0)
                
                #buf1 = cv2.merge([buf1,buf1,buf1])#*(0.255)
                buf1 = cv2.cvtColor(buf1, cv2.COLOR_GRAY2RGB)
                
                #cv2.imshow("CV2 Image", buf1)
                
                buf = buf1.tostring()
                self.tex.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

                self.rect = Rectangle(texture=self.tex, size=(640 , 480),
                              pos=(0, 0 ))
        
                #print('image of obtacles', buf1.shape, type(buf1))
            
        xx = goal_x - self.agent.x # difference of x-coordinates between the goal and the agent
        yy = goal_y - self.agent.y # difference of y-coordinates between the goal and the agent
        
        orientation = Vector(*self.agent.velocity).angle((xx,yy))/180. # direction of the agent with respect to the goal (if the agent is heading perfectly towards the goal, then orientation = 0)
        last_signal = [self.agent.signal1, self.agent.signal2, self.agent.signal3, orientation, -orientation] # our input state vector, composed of the three signals received by the three sensors, plus the orientation and -orientation
        action = brain.update(last_reward, last_signal) # playing the action from our ai (the object brain of the dqn class)
        scores.append(brain.score()) # appending the score (mean of the last 100 rewards to the reward window)
        
        rotation = action2rotation[action] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        self.agent.move(rotation) # moving the agent according to this last rotation angle
        distance = np.sqrt((self.agent.x - goal_x)**2 + (self.agent.y - goal_y)**2) # getting the new distance between the agent and the goal right after the agent moved
        self.ball1.pos = self.agent.sensor1 # updating the position of the first sensor (ball1) right after the agent moved
        self.ball2.pos = self.agent.sensor2 # updating the position of the second sensor (ball2) right after the agent moved
        self.ball3.pos = self.agent.sensor3 # updating the position of the third sensor (ball3) right after the agent moved

        if sand[int(self.agent.x),int(self.agent.y)] > 0: # if the agent is on the sand
            self.agent.velocity = Vector(1, 0).rotate(self.agent.angle) # it is slowed down (speed = 1)
            last_reward = -1 # and reward = -1
        else: # otherwise
            self.agent.velocity = Vector(6, 0).rotate(self.agent.angle) # it goes to a normal speed (speed = 6)
            last_reward = -0.1 # and it gets bad reward (-0.2)
            if distance < last_distance: # however if it getting close to the goal
                last_reward = 0.2 #0.1 # it still gets slightly positive reward 0.1
            else:
                last_reward = -0.1 # //// testing with negative reward on being far from goal
                
        if self.agent.x < 10: # if the agent is in the left edge of the frame
            self.agent.x = 10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.agent.x > self.width-10: # if the agent is in the right edge of the frame
            self.agent.x = self.width-10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.agent.y < 10: # if the agent is in the bottom edge of the frame
            self.agent.y = 10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1
        if self.agent.y > self.height-10: # if the agent is in the upper edge of the frame
            self.agent.y = self.height-10 # it is not slowed down
            last_reward = -1 # but it gets bad reward -1

        if distance < 100: # when the agent reaches its goal
            goal_x = self.width - goal_x # the goal becomes the bottom right corner of the map (the downtown), and vice versa (updating of the x-coordinate of the goal)
            goal_y = self.height - goal_y # the goal becomes the bottom right corner of the map (the downtown), and vice versa (updating of the y-coordinate of the goal)

            #####
            
         #####
        self.goal.x = goal_x
        self.goal.y = goal_y  
            
            
        # Updating the last distance from the agent to the goal
        last_distance = distance
    
    
    
  
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None
        
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        
        key = keycode[1]
        
        
        if key == 'escape':
            self.aruco_tracker.stop_ref()
            self.aruco_tracker.stop_pos()
            self.aruco_tracker.stop_obs_frame()
            self.aruco_tracker.stop_frame()
            cv2.destroyAllWindows()
            App.get_running_app().stop()
            
            #keyboard.release()
        elif key == 'r':
            if self.show_ref_frame:
                self.show_ref_frame = not self.show_ref_frame
                cv2.destroyWindow('Frame Original')
                cv2.destroyWindow('Frame Original with infos')
            else:
                self.show_ref_frame = not self.show_ref_frame
                
        elif key == 'p':
            if self.show_pos_frame:
                self.show_pos_frame = not self.show_pos_frame
                cv2.destroyWindow('Perspective')
                cv2.destroyWindow('Perspective with position')
            else:
                self.show_pos_frame = not self.show_pos_frame
                
        elif key == 'o':
            if self.show_obs_frame:
                self.show_obs_frame = not self.show_obs_frame
                cv2.destroyWindow('Perspective with obstacles')
            else:
                self.show_obs_frame = not self.show_obs_frame
        
        elif key == 'q':
            self.show_ref_frame = False
            self.show_pos_frame = False
            cv2.destroyAllWindows()
            #App.get_running_app().stop()
            #keyboard.release()
        
        elif key == 'm':   
            h = hpy()
            print('heap memory \n', h.heap())
            
# Painting for graphic interface (see kivy tutorials: https://kivy.org/docs/tutorials/firstwidget.html)
from kivy.graphics import Color, Line, Rectangle
import random
from kivy.graphics import RenderContext, Color, Rectangle, BindTexture

class MyPaintWidget(Widget):

    def on_touch_down(self, touch): # putting some sand when we do a left click
        global length,n_points,last_x,last_y
        with self.canvas:
           
            
            Color(0.8,0.7,0)
            d=10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch): # putting some sand when we move the mouse while pressing left
        global length,n_points,last_x,last_y
        if touch.button=='left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20*density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y
            
    def function_here():
        pass
            

# API and switches interface (see kivy tutorials: https://kivy.org/docs/tutorials/pong.html)

from kivy.core.window import Window

class agentApp(App):

    def build(self): # building the app
        Window.size = (640, 480)
        
        parent = Game()
        
   
        
        parent.serve_agent()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear', size =(70, 40))
        savebtn = Button(text='save',pos=(parent.width,0), size =(70, 40))
        loadbtn = Button(text='load',pos=(2*parent.width,0), size =(70, 40), background_color = (25/255, 211/255, 218/255, .85))
        #title = Label(text=('Hello world'),font_size=str(12) + 'sp', markup=True)
                          
        
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        
        #parent.add_widget(layout)
        
        self.title = 'DRL and CV for Autonomous Robot'
        
        
        #parent.add_widget(title)
        
        return parent

    def clear_canvas(self, obj): # clear button
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj): # save button
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj): # load button
        print("loading last saved brain...")
        brain.load()
        
    
    


        
# Running the app
if __name__ == '__main__':
    agentApp().run()
