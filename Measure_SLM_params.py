import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
import cv2
import time as t
import os
import shutil
import copy
import math 


class Measurment(object):
    
    def __init__(self, path_to_data, zone_coordinates, zone_size, skiped_img,   path_to_result = None, ):

        # we need find  upper left corner of each point area in  any redactor 
        self.first_point_coord_x = zone_coordinates[0][0]  
        self.first_point_coord_y = zone_coordinates[0][1] 
        self.second_point_coord_x = zone_coordinates[1][0]
        self.second_point_coord_y = zone_coordinates[1][1]
        self.third_point_coord_x = zone_coordinates[2][0]
        self.third_point_coord_y = zone_coordinates[2][1]

        # size of each point area  
        self.zone_size = zone_size

        # number of images, which we want to skip
        self.skiped_img = skiped_img

        # checking path  of images
        self.path_to_data = self.path_data(path_to_data)

        # checking path  for result 
        self.path_to_resault = self.path_res(path_to_result)

        #array of names of all images  
        self.img_names = os.listdir(self.path_to_data)

        # 1-dim arrays of intensities of each point area 
        self.first_point_data, self.second_point_data, self.third_point_data = self.get_data()
        
        
        
    # mthod for calculate time_parametrs of pulse 
    def  calculate_params (self, interval):
        start_time = t.time()
        self.intrval = interval
        data = [self.first_point_data, self.second_point_data, self.third_point_data]
        self.calculated_time_points = [] 
        self.calculated_value_points = []
        self.max_point_data = []
        for point in range(3): 
            self.max_point_data.append(self.max_research(data[point], interval))
            front_01_value = []
            front_09_value = []
            back_09_value = []
            back_01_value = []
            front_09_time = []
            front_01_time = []
            back_09_time = []
            back_01_time = []
            iteration = 0
            frame_controller = 0
            num_of_pulses = len(self.max_point_data[point])
            puls = 0
            last_time = 0
            last_value = 0
            controller = 0
            controller_for_interval = 0
            controller_front_01 = 0
            controller_front_09 = 0
            controller_back_01 = 0
            controller_back_09 = 0
            for time, value  in  enumerate(data[point]):
                if puls == num_of_pulses:
                    break
                if (
                    value >= self.max_point_data[point][puls] * 0.1 
                    and last_value <= self.max_point_data[point][puls] 
                    and controller_front_01 == 0
                    and frame_controller == 0
                    ):
                    error_01_front_value = value - self.max_point_data[point][puls] * 0.1 
                    error_01_front_last_value = self.max_point_data[point][puls] * 0.1  - value
                    if error_01_front_value <=  error_01_front_last_value:
                        best_value_01_front = value
                        best_time = time
                    else:
                        best_value_01_front = last_value
                        best_time = last_time
                    
                    controller_front_01 += 1
                    controller += 1
                    frame_controller += 1
                    front_01_value.append(best_value_01_front)
                    front_01_time.append(best_time)
                elif (
                      value >= self.max_point_data[point][puls] * 0.9 
                      and last_value <= self.max_point_data[point][puls] * 0.9 
                      and controller_front_09 == 0
                      and frame_controller == 0
                      ):
                    error_09_front_value = value - self.max_point_data[point][puls] * 0.9
                    error_09_front_last_value = self.max_point_data[point][puls] * 0.9  - value
                    if error_09_front_value <=  error_09_front_last_value:
                        best_value_09_front = value
                        best_time = time
                    else:
                        best_value_09_front = last_value
                        best_time = last_time
                    controller_front_09 += 1
                    controller += 1
                    frame_controller += 1
                    front_09_value.append(best_value_09_front)
                    front_09_time.append(best_time)
                elif (
                      value <= self.max_point_data[point][puls] * 0.9 
                      and last_value >= self.max_point_data[point][puls] * 0.9 
                      and controller_back_09 == 0
                      and frame_controller == 0
                      ):
                    error_09_back_value = self.max_point_data[point][puls] * 0.9 - value
                    error_09_back_last_value = value - self.max_point_data[point][puls] * 0.9
                    if error_09_back_value <=  error_09_back_last_value:
                        best_value_09_back = value
                        best_time = time
                    else:
                        best_value_09_back = last_value
                        best_time = last_time                        
                    controller_back_09 += 1
                    controller += 1
                    frame_controller += 1
                    back_09_value.append(best_value_09_back)
                    back_09_time.append(best_time)
                elif (
                      value <= self.max_point_data[point][puls] * 0.1 
                      and last_value >= self.max_point_data[point][puls] * 0.1 
                      and controller_back_01 == 0
                      and frame_controller == 0
                     ):
                    error_01_back_value = self.max_point_data[point][puls] * 0.1 - value
                    error_01_back_last_value = value - self.max_point_data[point][puls] * 0.1
                    if error_01_back_value <=  error_01_back_last_value:
                        best_value_01_back = value
                        best_time = time
                    else:
                        best_value_01_back = last_value
                        best_time = last_time                        
                    controller_back_01 += 1
                    controller += 1
                    frame_controller += 1
                    back_01_value.append(best_value_01_back)
                    back_01_time.append(best_time)
                    

                
                
                
                if frame_controller != 0:
                        iteration += 1
                        if  iteration == 20:
                            frame_controller = 0
                            iteration = 0
                
                
                if controller == 4:
                    controller_for_interval += 1
                    if controller_for_interval == 100:
                        puls += 1
                        controller_for_interval = 0
                        controller_front_01 = 0
                        controller_front_09 = 0
                        controller_back_09 = 0
                        controller_back_01 = 0
                        controller = 0
                    
                    
                last_value = copy.copy(value)
                last_time = copy.copy(time)
            
            
            time = [
                    np.array(front_01_time), np.array(front_09_time),
                    np.array(back_09_time), np.array(back_01_time)
                    ]
            value = [np.array(front_01_value),np.array(front_09_value),
                     np.array(back_09_value),np.array(back_01_value)
                    ]
            
            self.calculated_time_points.append(time)
            self.calculated_value_points.append(value)
        self.calculate()
        print("The parameters data is calculated  --- %s seconds ---" % (t.time() - start_time))
            
            
        
    def calculate(self):
        self.full_front = np.array([])
        self.full_back = np.array([])
        self.full_width  = np.array([])
        self.full_period = []
        full_09_front = np.array([])
        full_09_back =  np.array([])
        for point in range(3):
            
            
            front_time = self.calculated_time_points[point][1] - self.calculated_time_points[point][0]
            back_time = self.calculated_time_points[point][3] - self.calculated_time_points[point][2]
            impuls_width = self.calculated_time_points[point][2] - self.calculated_time_points[point][1]
            
            full_09_back = np.hstack([full_09_back, self.calculated_time_points[point][2]])
            self.full_front = np.hstack([self.full_front, front_time])
            self.full_back = np.hstack([self.full_back, back_time])
            self.full_width = np.hstack([self.full_width, impuls_width])
        
        
        self.num_of_pulses = sum([len(element) for element in self.max_point_data])  
        i = 0
        full_09_back = np.sort(full_09_back)
        while i < self.num_of_pulses - 1:
            self.full_period.append(full_09_back[i + 1] - full_09_back[i])
            i += 1

        self.full_period = np.array(self.full_period)
        
        print("Number of front times = ", len(self.full_front))
        print("Number of back times  = ", len(self.full_back))
        print("Numper of pulse's widths = ", len(self.full_width))
        print("Number of periods  = ", len(self.full_period))
        print("Numper of pulses = ", self.num_of_pulses )
        print("For getting  the parameters, use the method 'get_params(confidence probability)'")
        print("For getting list of parameters, use attribute 'params' ")
        self.params = [self.full_front, self.full_back, self.full_width, self.full_period]
        
        
        
    
    def get_params(self,conf_prob):
        self.parametrs_char  = []
        self.conf_prob = conf_prob
        for params in self.params:
             self.parametrs_char.append(self.error(params, conf_prob))
        names = ["FRONT", "BACK", "WIDTH" , "PERIOD"]
                
        print ("                              Time parameters  of SLM                        ")
        
        for i in range(4):
            print(" {}  Mean = {}    NSTD = {}   VAR = {}  CONFIDENCE INTERVAL = {}   CONFIDENCE PROBABILITY {}  ".format(names[i],
                str(self.parametrs_char [i][0]), str(self.parametrs_char [i][1]), str(self.parametrs_char [i][2]), 
                str(self.parametrs_char [i][3]), self.conf_prob)
                )



    def save(self, frame_rate):
        # save 1_dim of time points 
        points = ['first', 'second', 'third']
        value = [self.first_point_data, self.second_point_data, self.third_point_data]
        for i in range(3):
            time = np.arange(len(value[i])) / frame_rate
            np.savetxt(
                        self.path_to_resault + "\\" + '{}_point_data.txt'.format(points[i]),
                        np.hstack((time[np.newaxis, :].T, value[i][np.newaxis, :].T))
                        )

        # save time parameters 
        names = ["FRONT", "BACK", "WIDTH" , "PERIOD"]
        file = open(self.path_to_resault + '\\' + 'time_parameters.txt', "w")
        file.write("                              Time parameters  of SLM                       ")
        for i in range(4): 
            file.write(" {}  Mean = {}    NSTD = {}   VAR = {}  CONFIDENCE INTERVAL = {}   CONFIDENCE PROBABILITY {}  ".format(names[i],
                str(self.parametrs_char [i][0]), str(self.parametrs_char [i][1]), str(self.parametrs_char [i][2]), 
                str(self.parametrs_char [i][3]), self.conf_prob)
                )
        file.close()

        #save max_points 
        np.savetxt(self.path_to_resault + '\\' + 'max_points_of_every_pulse.txt', np.array(self.max_point_data[:][0])[np.newaxis, :].T)

        #save front 
        np.savetxt(self.path_to_resault + '\\' + 'fronts.txt', self.full_front[np.newaxis, :].T)
        #save back
        np.savetxt(self.path_to_resault + '\\' + 'fronts.txt', self.full_back[np.newaxis, :].T)
        #save width
        np.savetxt(self.path_to_resault + '\\' + 'fronts.txt', self.full_width[np.newaxis, :].T)
        #save periods
        np.savetxt(self.path_to_resault + '\\' + 'fronts.txt', self.full_period[np.newaxis, :].T)

        

        
            
        

    def error(self, data, conf_prob):
        mean = np.mean(data)
        n = len(data)
        from scipy.stats import t 
        student_coef = round(t.ppf((1 + conf_prob )/2, n - 1), 2)
        var = np.var(data)
        interval = student_coef * math.sqrt((var)/n) 
        nskd = math.sqrt(var)
        return np.around(np.array([mean, nskd,  var, interval]),decimals=3 )
        
    # this function    
    def get_data(self):
        start_time = t.time()
        first_point_data = []
        second_point_data = []
        third_point_data = []
        imgs_min = []
        for i in range(self.skiped_img, len(self.img_names)):
            img = np.array(Image.open(self.path_to_data + "\\" + self.img_names[i]))
            first_point_area = img[
                                    self.first_point_coord_x:self.first_point_coord_x + self.zone_size,
                                    self.first_point_coord_y:self.first_point_coord_y + self.zone_size
                                   ]
            second_point_area = img[
                                    self.second_point_coord_x:self.second_point_coord_x + self.zone_size,
                                    self.second_point_coord_y:self.second_point_coord_y + self.zone_size
                                    ]
            third_point_area = img[
                                    self.third_point_coord_x:self.third_point_coord_x + self.zone_size,
                                    self.third_point_coord_y:self.third_point_coord_y + self.zone_size
                                   ]
                                   
            first_point_data.append(np.sum(first_point_area))
            second_point_data.append(np.sum(second_point_area))
            third_point_data.append(np.sum(third_point_area))    

            #imgs_min need for normalization
            img_min = np.min(img)
            imgs_min.append(img_min)
        imgs_min = np.mean(np.array(imgs_min))  

        # this formula need for only one cycle in this function 
        # you can check it yourself  
        first_point_data = np.array(first_point_data) -  self.zone_size**2*imgs_min
        second_point_data = np.array(second_point_data) -  self.zone_size**2*imgs_min
        third_point_data  =np.array(third_point_data) -  self.zone_size**2*imgs_min
        print("The data is unpacked  --- %s seconds ---" % (t.time() - start_time))
        return first_point_data, second_point_data, third_point_data
        
    
            
    def max_research(self, point_data, interval):
        start = 0
        max_point_data = []
        point_data_len = len(point_data)
        while start + interval < point_data_len:
            informative_data = point_data[start:start + interval]
            max_point_data.append(np.max(informative_data))
            start += interval
        return np.array(max_point_data) 
          
        
        

    def path_res(self, path_to_result):
        if path_to_result is not None:
            if os.path.exists(path_to_result):
                print("This directory already exists. Do you want rewrite ? ")
                correct_answ = False
                while correct_answ is False:
                    user_answ = int(input("  YES - 1 / NO - 0"))
                    if user_answ != 0 and user_answ != 1:
                        print("Incorrect input")
                    else: 
                        user_answ = bool(user_answ)
                        correct_answ = True
                if user_answ:
                    self.del_rewrite(path_to_result)
                    return path_to_result
                else:
                    new_path = input("Write new path ")
                    self.create_folder(new_path)
                    return new_path
            else:
                self.create_folder(path_to_result)
                return path_to_result
        else:
            default_path  = os.getcwd() + "\\Mesurments"
            if os.path.exists(default_path): 
                print("This directory already exists. Do you want rewrite ? ")
                correct_answ = False
                while correct_answ is False:
                    user_answ = int(input("  YES - 1 / NO - 0 "))
                    if user_answ != 0 and user_answ != 1:
                        print("Incorrect input")
                    else: 
                        user_answ = bool(user_answ)
                        correct_answ = True
                if user_answ:
                    self.del_rewrite(default_path)
                    return default_path
                else:
                    i = 1
                    default_path = os.getcwd() + "\\Mesurments_" + str(i)
                    while os.path.exists(default_path):
                        c = os.getcwd() + "\\Mesurments_" + str(i)
                        i += 1
                    self.create_folder(default_path)
                    return default_path
            else:
                self.create_folder(default_path)
                return default_path
            
    def path_data(self, path_to_data):
        if path_to_data != None:
            return path_to_data 
        else:
            print("Incorrect path. Directory wasn't found ")
            exit()
                           
    def del_rewrite(self, path):
        try:
            shutil.rmtree(path)
        except OSError:
            print ("Failed to remove directory %s" % path)
            exit()
        else:
            print ("Directory %s was removed " % path)
            
        self.create_folder(path)
    

    def create_folder(self, path):
        try:
            os.mkdir(path)
        except OSError:
            print (" Failed to create directory %s " % path)
            exit()
        else:
            print ("Directory %s was created " % path)


res = Measurment(
                path_to_data = "E:\\учёба\\7 семестр\\УИР и Практика 7 сем\\result_21_12_21_v1",
                zone_coordinates = ((12,251),(12,145),(12,40)), 
                zone_size = 20, 
                skiped_img = 650,
                
    )
res.calculate_params(1500)
res.get_params(0.99)
res.save(1000)
