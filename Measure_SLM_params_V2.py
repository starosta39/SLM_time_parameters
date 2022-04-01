
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import time as t
import os
import shutil
import math
from scipy.signal import find_peaks, peak_widths 

class Measurment(object):
    
    def __init__(self, path_to_data, zone_coordinates, zone_size, skiped_img, framerate,   path_to_result = None, stop =None):
        self.framerate = framerate
        self.stop = stop
        # we need find  upper left corner of each point area in  any redactor 
        self.coordinats = zone_coordinates
        self.num_of_points = len(zone_coordinates)
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
        self.first_point_data, self.second_point_data, self.third_point_data = self.get_data(
                                                                                                skiped_img, 
                                                                                                stop,
                                                                                                path_to_data,
                                                                                                zone_coordinates,
                                                                                                zone_size
                                                                                            )
        
        # mthod for calculate time_parametrs of pulse 
    def  calculate_params (self, search_interval=None, height=None, threshold = None, prominence=None ):
        start_time = t.time()
        self.search_interval = search_interval
        data = [self.first_point_data, self.second_point_data, self.third_point_data]
        self.time_line = np.arange(self.final - self.skiped_img ) / self.framerate
        self.left_up_ind = [] 
        self.right_up_ind = []
        self.left_down_ind = [] 
        self.right_down_ind = [] 
        self.up_level = []
        self.down_level = []
        self.max_peak_ind = []
        for i in range(self.num_of_points):
            max_time, max_signal, max_signal_ind = self.find_max(   
                                                                    self.time_line,
                                                                    data[i],
                                                                    height,
                                                                    threshold,
                                                                    prominence,
                                                                    search_interval
                                                                )
            
            _, down_level, left_coord_down, right_coord_down = peak_widths(data[i], max_signal_ind, rel_height=0.9)
            left_coord_down = np.round(left_coord_down).astype('int')
            right_coord_down = np.round(right_coord_down).astype('int')
            
            _, up_level, left_coord_up, right_coord_up = peak_widths(data[i], max_signal_ind, rel_height=0.1)
            left_coord_up = np.round(left_coord_up).astype('int')
            right_coord_up = np.round(right_coord_up).astype('int')
            self.left_up_ind.append(left_coord_up)
            self.right_up_ind.append(right_coord_up)
            self.left_down_ind.append(left_coord_down)
            self.right_down_ind.append(right_coord_down)
            self.up_level.append(up_level)
            self.down_level.append(down_level)
            self.max_peak_ind.append(max_signal_ind)
        all_levels, coord = self.drop_last_peak(
                                                [  
                                                   self.max_peak_ind,
                                                   self.up_level,
                                                   self.down_level,
                                                ],
                                                [  
                                                   self.left_up_ind,
                                                   self.right_up_ind,
                                                   self.left_down_ind,
                                                   self.right_down_ind
                                                ],
                                                self.max_peak_ind
                                                )
        self.max_peak_ind, self.up_level, self.down_level = all_levels
        self.left_up_ind, self.right_up_ind, self.left_down_ind, self.right_down_ind = coord
        self.calculate()
        print("The parameters data is calculated  --- %s seconds ---" % (t.time() - start_time))
            
    # mthod for calculate time_parametrs of pulse 
    
    def  calculate(self):
        self.front = np.array([])
        self.back = np.array([])
        self.width  = np.array([])
        right_up_time = np.array([])     
        for i in range(self.num_of_points):
            self.front = np.hstack([self.front, self.time_line[self.left_up_ind[i]] - self.time_line[self.left_down_ind[i]]])
            self.back = np.hstack([self.back, self.time_line[self.right_down_ind[i]] - self.time_line[self.right_up_ind[i]]])
            self.width = np.hstack([self.width, self.time_line[self.right_up_ind[i]] - self.time_line[self.left_up_ind[i]]])
            right_up_time = np.hstack([right_up_time, self.time_line[self.right_up_ind[i]]])
        right_up_time = np.sort(right_up_time)
        self.period = np.diff(right_up_time)
        print("Number of front times = ", len(self.front))
        print("Number of back times  = ", len(self.back))
        print("Numper of pulse's widths = ", len(self.width))
        print("Number of periods  = ", len(self.period))
        print("For getting  the parameters, use the method 'get_params(confidence probability)'")
        print("For getting list of parameters, use attribute 'params' ")
        
        self.params = [
                        self.front*self.framerate, 
                        self.back*self.framerate,
                        self.width*self.framerate, 
                        self.period*self.framerate
                        ]

        
    def drop_last_peak(self, data, points,  max_peak_ind):
        lens = list(map(len, max_peak_ind))
        eq_check = all(lens[i]==lens[self.num_of_points-i-1] for i in range(self.num_of_points))
        if eq_check:
            best_len = lens[0]-1
        else:
            best_len = min(lens)
        for level in data:
            for i in range(len(level)):
                level[i] = level[i][:best_len]
        for point_place in points :
            for i in range(len(point_place)):
                point_place[i] = point_place[i][:best_len]
        return [data, points]
    
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

    def save(self):
        # save 1_dim of time points 
        points = ['first', 'second', 'third']
        value = [self.first_point_data, self.second_point_data, self.third_point_data]
        for i in range(3):
            np.savetxt(
                        self.path_to_resault + "\\" + '{}_point_data.txt'.format(points[i]),
                        np.hstack((self.time_line[np.newaxis, :].T, value[i][np.newaxis, :].T))
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
        max_point_data = np.array([value[i][self.max_peak_ind[i]] for i in range(len(value))])
        np.savetxt(self.path_to_resault + '\\' + 'max_points_of_every_pulse.txt', np.array(max_point_data[:][0])[np.newaxis, :].T)

        #save front 
        np.savetxt(self.path_to_resault + '\\' + 'fronts.txt', self.front[np.newaxis, :].T)
        #save back
        np.savetxt(self.path_to_resault + '\\' + 'backs.txt', self.back[np.newaxis, :].T)
        #save width
        np.savetxt(self.path_to_resault + '\\' + 'widths.txt', self.width[np.newaxis, :].T)
        #save periods
        np.savetxt(self.path_to_resault + '\\' + 'period.txt', self.period[np.newaxis, :].T)

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
    def get_data(self, skiped_img, stop, path_to_data,  coordinates, zone_size):
        start_time = t.time()
        first_point_data = []
        second_point_data = []
        third_point_data = []
        imgs_min = []
        if stop is None:
            self.final = len(self.img_names)  
        else:
            self.final = stop
        for i in range(skiped_img, self.final):
            img = np.array(Image.open(path_to_data + "\\" + self.img_names[i]))
            first_point_area = img[
                                    coordinates[0][0]:coordinates[0][0] + zone_size,
                                    coordinates[0][1]:coordinates[0][1] + zone_size
                                   ]
            second_point_area = img[
                                    coordinates[1][0]:coordinates[1][0] + zone_size,
                                    coordinates[1][1]:coordinates[1][1] + zone_size
                                    ]
            third_point_area = img[
                                    coordinates[2][0]:coordinates[2][0] + zone_size,
                                    coordinates[2][1]:coordinates[2][1] + zone_size
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
        first_point_data = np.array(first_point_data) -  zone_size**2*imgs_min
        second_point_data = np.array(second_point_data) -  zone_size**2*imgs_min
        third_point_data  = np.array(third_point_data) -  zone_size**2*imgs_min
        first_point_data = first_point_data - first_point_data.min()
        second_point_data = second_point_data - second_point_data.min()
        third_point_data = third_point_data - third_point_data.min()
        print("The data is unpacked  --- %s seconds ---" % (t.time() - start_time))
        return first_point_data, second_point_data, third_point_data
              
    def find_max(self, time_line, signal, height, threshold, prominence, distance ):
        max_ind = find_peaks(signal, height=height, threshold=threshold, prominence =  prominence, distance = distance)
        return [time_line[max_ind[0]], signal[max_ind[0]], max_ind[0]]
        
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

SLM = Measurment(
                path_to_data = "E:\\учёба\\7 семестр\\УИР и Практика 7 сем\\result_21_12_21_v1",
                zone_coordinates = ((12,251),(12,145),(12,40)), 
                framerate=1000,
                zone_size = 20, 
                skiped_img = 650
                ) 
SLM.calculate_params(search_interval=500,height = 1.5e+6 )
SLM.get_params(0.99)
SLM.save()