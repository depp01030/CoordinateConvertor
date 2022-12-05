# -*- coding: utf-8 -*-
"""


#function 1. 輸出包含點的所有step
start = time.time()
for i in range(6000):
    contain_step_lst = AllStepObj.get_containing_polygon(candi_point)
end = time.time()
diff = end - start

#funtcion 2. 確認點有沒有在特定的step
is_in_step = AllStepObj.check_if_point_in_step(candi_point, 'pcb')
#function 3. to convert position of test point from source step into target step
把source step 的point_list 座標 轉換成 target step
new_point_lst = AllStepObj.step_coordinate_conversion(target_step = 'panel',
                                                      source_step = 'pcb',
                                                      point_lst = candi_point)
  


"""
#%%
import numpy as np
import os
import sys
import time
import requests
import re
import copy
from datetime import datetime
import math
os.chdir(r'C:\Users\User\Desktop\Python\Git Repo\CamDataExtractor')
#%%
class CoordinateConvertor:
    
    def __init__(self,job_name, ancestor):

        all_step_dict = {}
        self.ancestor = ancestor
        self.all_step_dict = parse_sr_info_to_all_step_dict(job_name, ancestor, all_step_dict )
        self.job_name = job_name

    def step_coordinate_convert(self,
                                target_step,
                                source_step,
                                point_list):
        all_step_dict = self.all_step_dict
        if target_step not in self.all_step_dict.keys():
            raise ValueError('step {0} is not found in step list.'.format(target_step))
        elif target_step == source_step:
            raise ValueError('why??? why are u input the same step??')            
        elif source_step == self.ancestor:
            raise ValueError('why??? what do u want me to convert to ??')
        if target_step == self.ancestor :
            mapping_array_list = all_step_dict[source_step]['mapping_array']
        else: # create a mapping array list,convert from source step to target step
            source_mapping_list = all_step_dict[source_step]['mapping_array']
            loop_list = copy.deepcopy(source_mapping_list)
            child_step = source_step
            parent_step = all_step_dict[source_step]['parent_step']    
            while parent_step != target_step:
                next_loop_list = []
                parent_mapping_list = all_step_dict[parent_step]['mapping_array']
                for source_array in loop_list:
                    for parent_array in parent_mapping_list:
                        
                                                
                        next_loop_list.append( parent_array @ source_array )
                child_step = parent_step                
                loop_list = copy.deepcopy(next_loop_list)
                parent_step = all_step_dict[child_step]['parent_step']    
                print(parent_step)
            mapping_array_list = loop_list
            # mapping_array_list = all_step_dict[source_step]['mapping_array']

        return_list = []
        points_array = np.r_[np.array(point_list).T, np.ones((1,len(point_list)))]
        for mapping_array in mapping_array_list:
            
            _points_array = mapping_array @ points_array
            new_point_list = np.delete(_points_array ,2, axis= 0).T.tolist()
        
            return_list.append(new_point_list)

        return return_list
        
    def get_containing_step(self, point):

        '''
        Purpose: To return a list which shows  all step in polygon_dict,
                 that the point is in.
        Input:
            point : [float, float] -> a single point.
            point = [5,7.5]
        return : list of steps : [str, str,...]
                 -> step in self.polygon_dict that contains input point.
        '''
        
        containing_point_step_list = []
        for step, step_dict in self.all_step_dict.items():
            is_in_polygon = False
            polygon_lst = step_dict['sr_profile_list']
            for polygon in polygon_list:
                is_in_polygon = self.point_in_polygon(polygon, point)
                if is_in_polygon :
                    containing_point_step_list.append(step)
                    break

        

        return containing_point_step_list
    
    def check_if_point_in_step(self,
                                point,
                                polygon_name):
        '''
        Purpose: To check if point is in the step of 'polygon_name'.
        Input: 
            point : [float, float] -> a single point.
                    point = [0,0]
            polygon_name : str -> name of step.
        '''
        if polygon_name not in self.all_step_dict.keys():
            raise ValueError('step {0} is not found in step list.'.format(target_step))
        polygon_list = self.all_step_dict[polygon_name]['sr_profile_list']
        is_in_step  = False
        for polygon in polygon_list:
            is_in_polygon = self.point_in_polygon(polygon, point)
            if is_in_polygon :
                is_in_step = True
                break
        return is_in_step

        
    def point_in_polygon(self,
                         polygon,
                         point):
        """
        Raycasting Algorithm to find out whether a point is in a given polygon.
        Performs the even-odd-rule Algorithm to find out whether a point is in a given polygon.
        This runs in O(n) where n is the number of edges of the polygon.
         *
        :param polygon: an array representation of the polygon where polygon[i][0] is the x Value of the i-th point and polygon[i][1] is the y Value.
        :param point:   an array representation of the point where point[0] is its x Value and point[1] is its y Value
        :return: whether the point is in the polygon (not on the edge, just turn < into <= and > into >= for that)
        """
    
        # A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
        odd = False
        # For each edge (In this case for each point of the polygon and the previous one)
        i = 0
        j = len(polygon) - 1
        while i < len(polygon) - 1:
            i = i + 1
            # If a line from the point into infinity crosses this edge
            # One point needs to be above, one below our y coordinate
            # ...and the edge doesn't cross our Y corrdinate before our x coordinate (but between our x coordinate and infinity)
    
            if (((polygon[i][1] > point[1]) != (polygon[j][1] > point[1])) and (point[0] < (
                    (polygon[j][0] - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1])) +
                                                                                polygon[i][0])):
                # Invert odd
                odd = not odd
            j = i
        # If the number of crossings was odd, the point is in the polygon
        return odd        
#%% Util

'''
{'gSRangle': [0, 0, 0, 0],
 'gSRdx': [0, 0, 0, 0],
 'gSRdy': [0, 0, 0, 0],
 'gSRmirror': ['no', 'no', 'no', 'no'],
 'gSRnx': [1, 1, 1, 1],
 'gSRny': [1, 1, 1, 1],
 'gSRstep': ['sub', 'sub', 'sub', 'sub'],
 'gSRxa': [0, 0, 4.488189, 4.488189],
 'gSRxmax': [4.3307087, 4.3307087, 8.8188977, 8.8188977],
 'gSRxmin': [0, 0, 4.488189, 4.488189],
 'gSRya': [0, 4.488189, 0, 4.488189],
 'gSRymax': [4.3307087, 8.8188977, 4.3307087, 8.8188977],
 'gSRymin': [0, 4.488189, 0, 4.488189]}
'''


def get_prof_from_odb(job_name, sr_step):
    # job_name  = 'pang-am0008-xxxx'
    # sr_step = 'panel'
    path = '{0}/steps/{1}/'.format(job_name, sr_step)
    profile_file = open(os.path.join(path,'profile.txt'),'r')
    profile_line_text = profile_file.readlines()
    profile_file.close()
    
    profile = []
    for i, line_text in enumerate(profile_line_text):
        line_text = line_text.strip()
        if 'OB' in line_text and 'I' in line_text :
            x = float(line_text.split(' ')[1])
            y = float(line_text.split(' ')[2])
            profile.append([x,y])
        elif 'OS' in line_text:
            x = float(line_text.split(' ')[1])
            y = float(line_text.split(' ')[2])            
            profile.append([x,y])
            
        elif 'OC' in line_text:
            xs = float(profile_line_text[i-1].split(' ')[1])
            ys = float(profile_line_text[i-1].split(' ')[2])
            # xs,ys = last_xy    
            xe = float(line_text.split(' ')[1])
            ye = float(line_text.split(' ')[2])
            xc = float(line_text.split(' ')[3])
            yc = float(line_text.split(' ')[4])
            direction = 'CW' if line_text.split(' ')[5] == "N" else 'CCW'
            arc_point_list = curve_interpolation([xs,ys], [xe,ye], [xc,yc], direction)
    
            profile.extend(arc_point_list)
        
    return profile

def curve_interpolation(pt_s, pt_e, pt_c, direction, seg_len = 0.2):
    # seg_len = 0.002 #inch
    xs, ys = pt_s
    xe, ye = pt_e
    xc, yc = pt_c 
    
    angle = get_vector_angle([xs - xc, ys - yc],
                             [xe - xc, ye - yc],
                             direction)
    if angle == 0:
        angle = 360
        
        
    circle_r = ((xs-xc)**2 + (ys-yc) **2) **(1/2)
    arc_length = 2 * math.pi * circle_r * (angle/360)
    interpolation_num = int( arc_length / seg_len)
    shift_angle = angle / interpolation_num
    
    interpolartion_list = [[xs, ys]]
    for i in range(interpolation_num):
        angle = i * shift_angle
        angle = math.radians(angle)
        new_x= (x-datum[0])*math.cos(angle) + (y-datum[1])*math.sin(angle) + datum[0]
        new_y = (y-datum[1])*math.cos(angle) - (x-datum[0])*math.sin(angle) + datum[1]
        new_x, new_y
        interpolartion_list.append([new_x, new_y])
    interpolartion_list.append([xe, ye])
    
    return interpolartion_list
    

def get_vector_angle(v1, v2, direction):
    # v2 = [9,0]
    dot = v1[0]*v2[0] + v1[1] * v2[1]
    cos_theta = dot / (v2[0]**2 + v2[1]**2)    
    cross = v1[0] * v2[1] - v1[1]*v2[0]
    if direction == 'CCW':
        if cross < 0 : #angle exceed 180
            angle = 180 * math.acos(cos_theta)/ math.pi
            angle = 360 - angle
        else:
            angle = 180 * math.acos(cos_theta)/ math.pi
        angle = -angle
    else:
        if cross > 0 : #angle exceed 180
            angle = 180 * math.acos(cos_theta)/ math.pi
            angle = 360 - angle
        else:
            angle = 180 * math.acos(cos_theta)/ math.pi
    
    return angle


def get_sr_dict_from_odb(job_name, sr_step):

    
    sr_step_dict = {'gSRangle': [],
                     'gSRdx': [],
                     'gSRdy': [],
                     'gSRmirror': [],
                     'gSRnx': [],
                     'gSRny': [],
                     'gSRstep': [],
                     'gSRxa': [],
                     'gSRxmax': [],
                     'gSRxmin': [],
                     'gSRya': [],
                     'gSRymax': [],
                     'gSRymin': []}
        
    path = '{0}/steps/{1}/'.format(job_name, sr_step)
    stephdr_file = open(os.path.join(path,'stephdr.txt'),'r')
    stephdr_line_text = stephdr_file.readlines()
    stephdr_file.close()
    flag = False
    for line_text in stephdr_line_text:
        line_text = line_text.strip()
        if 'STEP-REPEAT' in line_text:
            flag = True
        if '}' in line_text:
            flag = False
        if flag:
            if 'NAME' in line_text:
                sr_step_dict['gSRstep'].append(line_text.split('=')[1].lower())
            elif 'DX' in line_text:
                sr_step_dict['gSRdx'].append(float(line_text.split('=')[1]))
            elif 'DY' in line_text:
                sr_step_dict['gSRdy'].append(float(line_text.split('=')[1]))
            elif 'NX' in line_text:
                sr_step_dict['gSRnx'].append(int(line_text.split('=')[1]))
            elif 'NY' in line_text:
                sr_step_dict['gSRny'].append(int(line_text.split('=')[1]))
            elif 'ANGLE' in line_text:
                sr_step_dict['gSRangle'].append(int(line_text.split('=')[1]))
            elif 'MIRROR' in line_text:
                sr_step_dict['gSRmirror'].append(line_text.split('=')[1].lower())        
            elif 'X' in line_text:
                sr_step_dict['gSRxa'].append(float(line_text.split('=')[1]))
            elif 'Y' in line_text:
                sr_step_dict['gSRya'].append(float(line_text.split('=')[1]))
    
    return sr_step_dict
  

#%%
    
def parse_sr_info_to_all_step_dict(job_name, 
                                   sr_step,
                                   all_step_dict):
    
    '''
    job = genClasses.Job(os.environ['JOB'])    
    matrix_info =  genClasses.Matrix(job).getInfo()
    step_object = genClasses.Step(job, 'panel')  
    
    sr_step       : str  -> step name.
    all_step_dict : dict -> a dict keep all sr_step_dict
    
    
    -----------------------------------------------------
    test: 
        sr_step = 'sub'
        sr_step = 'valor'
        sr_step = 'pcb'
        sr_step = 'pcb'
        all_step_dict = {}
    '''
    
    '======================================================'
    standard_step_dict = {'step_name': '',
                         'profile'  : [],
                         'sr_profile_list'  : [],
                         'child_step' : [],
                         'parent_step': '',
                         'mirror' : [False],
                         'angle' : [0.0],
                         'dx' : [0.0],
                         'dy' : [0.0],
                         'ax' : [0.0],
                         'ay' : [0.0],
                         'nx' : [0.0],
                         'ny' : [0.0],
                         'mapping_array' : [],
                         'ancestor_mapping_array' : []}
    if all_step_dict == {} : #very first one. no parent.
        all_step_dict[sr_step] = copy.deepcopy(standard_step_dict)
        all_step_dict[sr_step]['mapping_array'] = [np.identity(3)]
        all_step_dict[sr_step]['ancestor_mapping_array'] = [np.identity(3)]
        all_step_dict[sr_step]['step_name']  = sr_step
    '======================================================================='
    #all child enter loop only one time (whatever they have same step name.)

    ## get and parse profile ##
    step_profile = get_prof_from_odb(job_name, sr_step)
    ## get sr_infomation ##
    sr_step_dict = get_sr_dict_from_odb(job_name, sr_step)
    
    all_step_dict[sr_step]['profile']  = step_profile
    sr_profile_list = profile_list_convert(step_profile,
                                            all_step_dict[sr_step]['ancestor_mapping_array'])
    print(sr_step, len(sr_profile_list))
    all_step_dict[sr_step]['sr_profile_list'] = sr_profile_list 
    if sr_step_dict['gSRstep'] != []: #not last leaf, got child
    
        all_step_dict[sr_step]['child_step'] = sr_step_dict['gSRstep']
        all_step_dict[sr_step]['angle'] = sr_step_dict['gSRangle']
        all_step_dict[sr_step]['dx'] = sr_step_dict['gSRdx']
        all_step_dict[sr_step]['dy'] = sr_step_dict['gSRdy']
        all_step_dict[sr_step]['mirror'] = sr_step_dict['gSRmirror']
        all_step_dict[sr_step]['nx'] = sr_step_dict['gSRnx']
        all_step_dict[sr_step]['ny'] = sr_step_dict['gSRny']
        all_step_dict[sr_step]['ax'] = sr_step_dict['gSRxa']
        all_step_dict[sr_step]['ay'] = sr_step_dict['gSRya']
        #loop all child
        for i, child_step in enumerate(all_step_dict[sr_step]['child_step']):
            angle = math.radians(all_step_dict[sr_step]['angle'][i])
            dx    = all_step_dict[sr_step]['dx'][i]
            dy    = all_step_dict[sr_step]['dy'][i]
            mirror= all_step_dict[sr_step]['mirror'][i]
            nx    = all_step_dict[sr_step]['nx'][i]
            ny    = all_step_dict[sr_step]['ny'][i]
            ax    = all_step_dict[sr_step]['ax'][i]
            ay    = all_step_dict[sr_step]['ay'][i]
            delta = [(nx-1) * dx + ax , (ny-1) * dy + ay]
            
            shift_array    =  np.array([[1,0,delta[0]],[0,1,delta[1]],[0,0,1]])
            rotation_array = np.array([[np.cos(angle), -np.sin(angle), 0],
                                       [np.sin(angle),  np.cos(angle), 0],
                                       [            0,              0, 1]])
            M_array = shift_array @ rotation_array
            # mirro_matrix = []
            if child_step not in all_step_dict: #first born.
                all_step_dict[child_step] = copy.deepcopy(standard_step_dict)
                all_step_dict[child_step]['parent_step'] = sr_step
                all_step_dict[child_step]['step_name'] = child_step
            child_step_dict = copy.deepcopy(all_step_dict[child_step])
                
            child_step_dict['mapping_array'] = [M_array] 
            # all_step_dict[child_step]['mapping_array'].append(M_array)
            all_step_dict = child_array_convert(child_step_dict, all_step_dict)
            
        
        # looking for grand child
        for child_step in list(set(all_step_dict[sr_step]['child_step'])):
            parse_sr_info_to_all_step_dict(job_name, child_step, all_step_dict,)
        
    #-----------------------------------#
        
    return all_step_dict 
   
#%%
# step_profile = get_prof_from_odb(job_name, 'panel')
def profile_list_convert(step_profile, mapping_array_list):
    
    step_profile_list = []
    one_array = np.ones((1,len(step_profile))).T
    profile_array = np.c_[np.array(step_profile), one_array]
    profile_array = profile_array.transpose()
    for mapping_array in mapping_array_list:    
        pass
        _profile_array = np.dot(mapping_array, profile_array)
        _profile_array = _profile_array .transpose()
        _profile_array = np.delete(_profile_array , -1, axis = 1)
        step_profile_list.append(_profile_array.tolist())
    return step_profile_list 


# all_step_dict['valor']
# parent_mapping_array  = all_step_dict['panel']['mapping_array']
def child_array_convert(child_step_dict, all_step_dict):
    '''
    each child will map their parent's mapping array here.
    '''
    
    # child_step_dict = all_step_dict[child_step]
    parent_step = child_step_dict['parent_step']
    child_mapping_array = child_step_dict['mapping_array'][0]
    child_mapping_array_list = []
    parent_step_dict = all_step_dict[parent_step]
    
    for parent_mapping_array in parent_step_dict['ancestor_mapping_array']:
        extend_array = np.dot( parent_mapping_array, child_mapping_array)
                        
        child_mapping_array_list.append(extend_array)
        
        
    all_step_dict[child_step_dict['step_name']]['mapping_array'].append(child_mapping_array)
    all_step_dict[child_step_dict['step_name']]['ancestor_mapping_array'].extend(child_mapping_array_list)
    return all_step_dict


#%%plotter

import matplotlib.pyplot as plt
def main_plot(polygon_dict = None, polygon_list = None, point_list = [],point_array = None):
    '''
    this function is only for checking result.
    in other words, this function used only in dev mode.
    '''
    #檢視
    
    
    color_list= ['blue', 'red', 'green', 'purple']
    i = 0
    
    plt.figure(figsize=(20, 20))
        
    if point_list:
        for point in point_list:
            plt.plot(point[0], point[1], "o", color = 'red')
    if polygon_dict:
        for step, polygon_list_in_dict in polygon_dict.items():   # loop all step
            color_index = i % len(color_list)
            color = color_list[color_index]
            for polygon in polygon_list_in_dict['sr_profile_list']:# loop all polygon in selected step
                x_list, y_list = zip(*polygon)
#                plt.plot(x_list, y_list, "o", color = 'black',markersize=3)
                plt.plot(x_list,y_list,"-",color = color )
                
            i += 1
    if polygon_list:
        color_index = i % len(color_list)
        color = color_list[color_index]
        for polygon in polygon_list:        # loop all polygon in selected step
            x_list, y_list = zip(*polygon)
            plt.plot(x_list, y_list, "o", color = 'black',markersize=3)
            plt.plot(x_list,y_list,"-",color = color )
    if point_array is not None:
        for point in point_array:
            plt.plot(point[0],point[1], "o", color = 'blue',markersize = 20)
        i += 1
    plt.xticks(range(-2,20,1),size = 30)
    plt.yticks(range(-2,20,1),size = 30)

    # axis_max = max(int(max(x_list))+5,int(max(y_list))+5)
    # axis_min = min(int(min(x_list))-5,int(min(y_list))-5)    
    # plt.xticks(range(axis_min,axis_max,5))
    # plt.yticks(range(axis_min,axis_max,5))
    plt.grid(color='r', linestyle='dotted', linewidth=1)
    plt.gca().set_aspect("equal")
#%%

def main(job_name):
    print('hi')
   
    ori_all_step_dict = {}
    sr_step = 'panel'
    ori_all_step_dict = parse_sr_info_to_all_step_dict(job_name, sr_step, ori_all_step_dict )

    pass
#%%
# if __name__ == "__main__":
#     print("****************************************")
#     print('************Program execute*************')
#     print("****************************************")


#     '''
#     os.environ['JOB'] = 'MTKX-OT001C-A1D0'.lower()
#     os.environ['JOB'] = 'pang-am0002-v1d01-juri'
    
#     '''
job_name = 'pang-am0008-xxxx'

sr_step = 'panel'
all_step_dict = {}
s = time.time()
# all_step_dict  = parse_sr_info_to_all_step_dict(job_name, sr_step, all_step_dict,)
print(time.time()-s)


s = time.time()

convertor = CoordinateConvertor(job_name, 'panel')
n = 1
candi_list = [[x,y] for x , y in zip(range(n),range(n))]
# candi_list = [[1,1]]
new_list = convertor.step_coordinate_convert('panel','pcb', candi_list )


print(time.time()-s)
main_plot(convertor.all_step_dict ,polygon_list = new_list )
# main_plot(convertor.all_step_dict,polygon_list = new_list )
# main_plot(polygon_dict = all_step_dict )
    
# #%%
# import numpy as np
# candi_array  = np.array([[2,2,1],
#                          [2,0,1]])
# candi_array = candi_array.transpose()  
# import math
# theta = 90
# theta = math.radians(theta)
# rotation_array = np.array([[np.cos(theta),-np.sin(theta),0],
#                             [np.sin(theta), np.cos(theta),0],
#                             [    0,              0,         1]])


# delta = [0,-2]
# shift_array = np.array([[1,0,delta[0]],[0,1,delta[1]],[0,0,1]])

# # candi_array = np.dot(rotation_array,candi_array)
# candi_array = rotation_array @ shift_array @  candi_array
# # candi_array = np.dot(shift_array ,candi_array)

#     = candi_array.transpose()  
# main_plot(point_array= candi_array)
















# %%
