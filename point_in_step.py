# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 23:01:25 2022

@author: User
"""

import os
import sys
import time
import requests
import re
import copy
from datetime import datetime
os.chdir(r'C:\Users\User\Desktop\Python\CamDataExtractor')


'''
os.environ['JOB'] = 'patrick-mtkx-md000l-a1d01-panel-multisub'
os.environ['JOB'] = 'patrick-mtkx-md001l-a1d01-panel-multipcb'
os.environ['JOB'] = 'patrick-gucx-pd001j-v1d01-panel-tilted'
os.environ['JOB'] = 'pang-am0002-v1d01-depp'
os.environ['JOB'] = 'patrick-qctx-pc002y-panel-valorsub'
'''
#%%


def parse_sr_info_to_all_step_dict(job, 
                                   sr_step,
                                   all_step_dict):
    
    '''
    job = genClasses.Job(os.environ['JOB'])    
    matrix_info =  genClasses.Matrix(job).getInfo()
    step_object = genClasses.Step(job, 'panel')  
    
    sr_step       : str  -> step name.
    all_step_dict : dict -> a dict keep all sr_step_dict
    
    sr_step = 'sub'
    all_step_dict = {}
    '''
    
    '======================================================'
    standard_step_dict = {'step_name': '',
                         'profile'  : [],
                         'sr_profile_lst'  : [],
                         'child_step' : [],
                         'parent_step': [],
                         'angle' : [0.0],
                         'dx' : [0.0],
                         'dy' : [0.0],
                         'ax' : [0.0],
                         'ay' : [0.0],
                         'nx' : [0.0],
                         'ny' : [0.0]}

    ### get self infomation -> profile###
    #  sr_step = 'panel'
    #  sr_step = 'valor'
    #  sr_step = 'pcb'
    ## get and parse profile ##
    step_prof_info = job.INFO('-t step -e %s/%s -d PROF' % (job.name, sr_step))
    step_profile = parse_profile_info(step_prof_info)
        
    ## get sr_infomation ##
    temp_step_dict = job.DO_INFO('-t step -e %s/%s -d SR' %(job.name, sr_step ))
    #-----------------------------------#
    if temp_step_dict ['gSRstep'] == []: #the input step is the last child step.
        if all_step_dict == {}: #the input parent is 'pcb', no any other step
            sr_step_dict = standard_step_dict
            sr_step_dict['profile']   = step_profile    
            sr_step_dict['step_name'] = sr_step
            sr_step_dict['sr_profile_lst'] = [step_profile] 
        else: #there r parent step in all_step_dict.
        
            sr_step_dict = all_step_dict[sr_step]
            sr_step_dict['profile']   = step_profile
    
            
            sr_profile_lst = [step_profile]
            ref_step = sr_step
            for parent_step in all_step_dict[sr_step]['parent_step']:
#                parent_step = 'valor'
                parent_step_dict = all_step_dict[parent_step]
    
                sr_profile_lst = convert_childstep_into_parentstep(parent_step_dict, ref_step, sr_profile_lst)
                ref_step = parent_step
            sr_step_dict['sr_profile_lst'] = sr_profile_lst 
            
            
        all_step_dict[sr_step] = sr_step_dict
        return all_step_dict
    else:  # the input step have child step
        if sr_step not in all_step_dict: #first run -> the very parent
            #only the very first parent enter this condition.
            sr_step_dict = copy.deepcopy(standard_step_dict)
        else:  #every parent create their child dict with default #standard_step_dict
               #so all child hash by key:sr_step, to get their dict in this condition.
            sr_step_dict = all_step_dict[sr_step]
        #transform temp_step_dict into basic step_dict (child)
        sr_step_dict['step_name'] = sr_step
        sr_step_dict['profile']   = step_profile
        sr_step_dict['sr_profile_lst']   = [step_profile]
        sr_step_dict['child_step']  = temp_step_dict['gSRstep']
        sr_step_dict['angle']  = temp_step_dict['gSRangle']
        sr_step_dict['dx'] = temp_step_dict['gSRdx']
        sr_step_dict['dy'] = temp_step_dict['gSRdy']
        sr_step_dict['ax'] = temp_step_dict['gSRxa']
        sr_step_dict['ay'] = temp_step_dict['gSRya']
        sr_step_dict['nx'] = temp_step_dict['gSRnx']
        sr_step_dict['ny'] = temp_step_dict['gSRny']
        all_step_dict[sr_step] = sr_step_dict
        tmp_sr_child_step_list = [ ]
        for tmp_child_stp in all_step_dict[sr_step]['child_step']:
	         if tmp_child_stp not in tmp_sr_child_step_list:
	             tmp_sr_child_step_list.append(tmp_child_stp)
        
        for child_step_name in tmp_sr_child_step_list: #loop all child step.
        
#            if child_step_name not in all_step_dict: #create child dict in all_step_dict if not exist
            child_step_dict = copy.deepcopy(standard_step_dict)
            all_step_dict[child_step_name] = child_step_dict
            
            
            if sr_step not in all_step_dict[child_step_name]['parent_step']:
                all_step_dict[child_step_name]['parent_step'].append(sr_step)
                relative_parent_lst = all_step_dict[sr_step]['parent_step']
                all_step_dict[child_step_name]['parent_step'].extend(relative_parent_lst )
            all_step_dict = parse_sr_info_to_all_step_dict(job, child_step_name, all_step_dict )

        #step that not last, convert in this part.
        sr_step_dict   = all_step_dict[sr_step]
        sr_profile_lst = sr_step_dict['sr_profile_lst'] 
    
    
        ref_step = sr_step
        for parent_step in sr_step_dict['parent_step']:
            parent_step_dict = all_step_dict[parent_step]
            sr_profile_lst = convert_childstep_into_parentstep(parent_step_dict, ref_step, sr_profile_lst)
            ref_step = parent_step
        sr_step_dict['sr_profile_lst'] = sr_profile_lst 
        all_step_dict[sr_step] = sr_step_dict
        return all_step_dict 
   

def parse_profile_info(step_prof_info):
    ''' 
    Parse input step_prfo_lst 
    Input:
    step_prof_info = 
            ['### Step profile data - panel ###\n',
              '#S P 0\n',
              '#OB -1.1811024 -1.1811024 I\n',
              '#OS -1.1811024 21.1023622\n',s
              '#OS 16.9685039 21.1023622\n',
              '#OS 16.9685039 -1.1811024\n',
              '#OS -1.1811024 -1.1811024\n',
              '#OE\n']
    Output =
            [[-1.1811024, -1.1811024],
              [-1.1811024, 21.1023622],
              [16.9685039, 21.1023622],
              [16.9685039, -1.1811024],
              [-1.1811024, -1.1811024]]
    '''
    profile = []

    for i, line_text in  enumerate(step_prof_info):
        if '#OB' in line_text and 'I' in line_text :
            x = float(line_text.split(' ')[1])
            y = float(line_text.split(' ')[2])
            profile.append([x,y])
        elif '#OS' in line_text:
            x = float(line_text.split(' ')[1])
            y = float(line_text.split(' ')[2])            
            profile.append([x,y])
        elif '#OC' in line_text:
            profile.pop()
            last_line_text = step_prof_info[i-1]
            
            arc_point_lst = get_point_2(last_line_text, line_text,0.01)
#            arc_point_lst.reverse()
            profile.extend(arc_point_lst)
    return profile
    
#---------------#
#    
#line_list =  [['-1.1811024', '-1.1811024'],
#             ['-1.1811024', '21.1023622'],
#             ['16.9685039', '21.1023622'],
#             ['16.9685039', '-1.1811024'],
#             ['-1.1811024', '-1.1811024']]  
#             
#line_list =  [[-1.1811024, -1.1811024],
#             [-1.1811024, 21.1023622],
#             [16.9685039, 21.1023622],
#             [16.9685039, -1.1811024],
#             [-1.1811024, -1.1811024]]
#    
#new_line_list = []
#for line in line_list[:]:
#    new_line_list.append([(line[0][0],line[0][1]),(line[1][0],line[1][1])])
#
#print(new_line_list)    
#    
    
            



#    
#step_prof_info =  ['### Step profile data - pcb ###',
#            '#S P 0',
#            '#OB 10.5708661 0.4724409 I',
#            '#OC 10.492126 0.3937008 10.5708661 0.3937008 N',
#            '#OS 10.492126 0',
#            '#OS 1.1811024 0',
#            '#OS 1.1811024 0.3937008',
#            '#OC 1.1023622 0.4724409 1.1023622 0.3937008 N',
#            '#OS 0 0.4724409',
#            '#OS 0 7.7559055',
#            '#OS 12.6771654 7.7559055',
#            '#OS 12.6771654 0.4724409',
#            '#OS 10.5708661 0.4724409',
#            '#OE']        
#            
#
#profile=parse_profile_info(step_prof_info)   
#main_plot(polygon_lst = [profile])
#
#
#
#
    



def convert_childstep_into_parentstep(parent_step_dict, 
                                      child_step_name, 
                                      child_profile_lst ):

    '''
    To convert specific step into multistep by infomation of step_dict.
    -> child coordinates into parent coordinates
    Input:
        child_step_name : str -> name of child step
        child_profile_lst : [[[],[],...], ->profile1
                             [[],[],...], ->profile2
                             ...] -> profile list
        
        parent_step_dict : dict
            profile : [[],[],...] -> profile point.
            #SUPER IMPORTANT NOTE: 
                the profile in parent_step_dict['profile'] belongs to parent not child.
                ex: parent is 'valor' and child is 'pcb'
                    the value of key 'profile' belongs to valor not pcb.
                    child pofile is another input.
                
            dx, dy  : float -> distance of x, y.
            ax, ay  : float -> anchor point of x, y.
            nx, ny  : int   -> number of step in x, y axis.
            rotation: float -> rotate angle.
        format : {'step_name': 'pcb',
                'profile': [[4, 4], [3, 6], [5, 3], [4, 3], [4, 4]],
                'dx': 2,
                'dy': 1,
                'ax': 0,
                'ay': 0,
                'nx': 2,
                'ny': 3,
                'rotation': 90}
    parent_step_dict
    Output:
        polygon_lst : [ [[x,y],[x,y],[x,y],[x,y],...,[x,y]], -> polygon1  
                        [[x,y],[x,y],[x,y],[x,y],...,[x,y]], -> polygon2
                        ... 
                      ]   
        
    -------------------------------------------------------------------------
    
    parent_step_dict, child_step_name, child_profile_lst = parent_step_dict, ref_step, sr_profile_lst
    parent_step_dict = all_step_dict['valor']
    child_step_name = 'pcb_80'
    child_profile_lst = all_step_dict[child_step_name]['sr_profile_lst']
    '''
    polygon_lst = []
    #parse all keys into variable.
    for index, candi_child_step_name  in enumerate(parent_step_dict['child_step']):
        if child_step_name == candi_child_step_name  :
            pass
        else:
            continue
#    index = parent_step_dict['child_step'].index(child_step_name)

        angle   = parent_step_dict['angle'][index]
        dx= parent_step_dict['dx'][index]
        dy= parent_step_dict['dy'][index]
        ax= parent_step_dict['ax'][index]
        ay= parent_step_dict['ay'][index]
        nx= parent_step_dict['nx'][index]
        ny= parent_step_dict['ny'][index]
        


        #loop nx,y time
        for child_profile in child_profile_lst:
            for x_time in range(nx):
                for y_time in range(ny):
                    datum_point = [ax + dx * x_time, ay + dy * y_time]
                    
                    #each time create new polygon, and add dx or dy on each point.
                    new_polygon = list(map(lambda x:[x[0] + dx * x_time + ax, x[1] + dy * y_time + ay], child_profile))
                    
                    
                    if angle != 0: #need rotate.
                        new_polygon = rotate_polygon(new_polygon, angle = angle, datum = datum_point)  
                    polygon_lst.append(new_polygon)
                    
    return polygon_lst
'''
polygon_lst = convert_childstep_into_parentstep(step_dict)
main_plot(polygon_lst = polygon_lst)        
            
'''        

def rotate_polygon(polygon, angle, datum = [0, 0]):
    '''
    to rotate input polygon.
    Input:
        polygon : [[x,y], [x,y], ....]
        angle   : float -> 0, 90, 180, 270
        datum   : [float, float] -> rotate datum point.
    '''
    # polygon = profile
    rotation_polygon = []
    angle = math.radians(angle)
    for pt in polygon:
        x,y = pt[0],pt[1]
        new_x= (x-datum[0])*math.cos(angle) + (y-datum[1])*math.sin(angle) + datum[0]
        new_y = (y-datum[1])*math.cos(angle) - (x-datum[0])*math.sin(angle) + datum[1]
        rotation_polygon.append([new_x, new_y])
    return rotation_polygon


def rotate_sort_reverse(x,y):
    if x[2] < y[2]:
        return 1
    elif x[2] > y[2]:
        return -1
    else:
        return 0

def rotate_sort(x,y):
    if x[2] < y[2]:
        return -1
    elif x[2] > y[2]:
        return 1
    else:
        return 0

#%% class  SRStepSet


class  SRStepSet:
    
    def __init__(self, all_step_dict):
        '''
        Input:
            all_step_dict : dict -> key: step, value: step_info
            we need only 'sr_profile_lst', which contains all profile of a specific step.
            format:
                key  : valor
                value:{'angle': [0, 0, 0, 0, 0, 0, 0],
                     'ax': [0, 0.9448807, 1.8897614, 2.8346421, 3.7795228, 4.7244035, 5.6692843],
                     'ay': [0, 0, 0, 0, 0, 0, 0],
                     'child_step': ['pcb-l', 'pcb-m', 'pcb-n', 'pcb-p', 'pcb-q', 'pcb-s', 'pcb-r'],
                     'dx': [0, 0, 0, 0, 0, 0, 0],
                     'dy': [0.9448807,
                      0.9448819,
                      0.9448819,
                      0.9448819,
                      0.9448819,
                      0.9448819,
                      0.9448819],
                     'nx': [1, 1, 1, 1, 1, 1, 1],
                     'ny': [14, 14, 14, 14, 14, 14, 14],
                     'parent_step': ['panel'],
                     'profile': [[0.0, 0.0],
                      [0.0, 12.8346291],
                      [6.2204642, 12.8346291],
                      [6.2204642, 0.0],
                      [0.0, 0.0]],
                     'sr_profile_lst': [[[4.7834687, 2.5393783],
                       [4.7834687, 15.3740074],
                       [11.0039329, 15.3740074],
                       [11.0039329, 2.5393783],
                       [4.7834687, 2.5393783]]],
                     'step_name': 'valor'}
        -----------------------------------
        Note:
        method we need:
            0. parse_all_steps_dict(polygon_lst) #for parse?
            1. get_containing_polygon(point)
            2. check_if_point_in_step(point, polygon_name)
            3. point_in_polygon(polygon, point) # base checking method
        '''
        self.all_step_dict = all_step_dict
        self.step_lst = all_step_dict.keys()
        self.polygon_dict = self.parse_all_steps_dict(all_step_dict)
        
    def step_coordinate_conversion(self,target_step, source_step, point_lst):
        '''
        Purpose: to converse position of points in source step to target_step.
        target_step = 'panel'
        source_step = 'pcb'
        point_lst = list -> [[x,y],[x,y],....]
        '''
        point_list_len = len(point_lst)
        all_step_dict = self.all_step_dict
#        point_lst = [[6.3785963, 3.2240002],[6.3785963,4.3161064]]
        child_step_name = source_step
        parent_step_lst = all_step_dict[child_step_name]['parent_step'] # ['sub', 'valor', 'panel']
        for parent_step in parent_step_lst:
            
            
#            point_lst = [[[6.378596300000001, 3.2240002], [6.378596300000001, 4.316106400000001]]]
#            parent_step ='valor'
#            child_step_name ='sub'
            parent_step_dict = all_step_dict[parent_step]
            #parse all keys into variable.

            polygon_lst = []
            for index, candi_child_step_name  in enumerate(parent_step_dict['child_step']):
                if child_step_name == candi_child_step_name  :
                    pass
                else:
                    continue
            #    index = parent_step_dict['child_step'].index(child_step_name)
#                new_point_lst = []
             
                angle   = parent_step_dict['angle'][index]
                dx= parent_step_dict['dx'][index]
                dy= parent_step_dict['dy'][index]
                ax= parent_step_dict['ax'][index]
                ay= parent_step_dict['ay'][index]
                nx= parent_step_dict['nx'][index]
                ny= parent_step_dict['ny'][index]


                datum_point = [ax, ay]
                #loop nx,y time
    #            child_profile = point_lst
#                for sub_point_lst in point_lst:
                for x_time in range(nx):
                    for y_time in range(ny):
                        datum_point = [ax + dx * x_time, ay + dy * y_time]
                        sub_point_lst = copy.deepcopy(point_lst)
                        #each time create new polygon, and add dx or dy on each point.
                        new_polygon = list(map(lambda x:[x[0] + dx * x_time + ax,
                                                         x[1] + dy * y_time + ay ], point_lst))
                        if angle != 0: #need rotate.
                            new_polygon = rotate_polygon(new_polygon, angle = angle, datum = datum_point)
                        polygon_lst.extend(new_polygon)

            child_step_name = parent_step 
            point_lst = copy.deepcopy(polygon_lst)
            if parent_step == target_step:
                break
            
        output_point_list = [] #seperate point list            
        temp_list = []
        i = 0
        for point  in point_lst:
            temp_list.append(point)

            i+=1
            if i == point_list_len:
                output_point_list.append(temp_list)
                temp_list = []
                i = 0
            
        return output_point_list
            
                
        
    def parse_all_steps_dict(self, all_step_dict):
        '''
        Input:
            all_step_dict : dict -> key: step, value: step_info
            we need only 'sr_profile_lst', which contains all profile of a specific step.
            format:
                key  : valor
                value:{'angle': [0, 0, 0, 0, 0, 0, 0],
                     'ax': [0, 0.9448807, 1.8897614, 2.8346421, 3.7795228, 4.7244035, 5.6692843],
                     'ay': [0, 0, 0, 0, 0, 0, 0],
                     'child_step': ['pcb-l', 'pcb-m', 'pcb-n', 'pcb-p', 'pcb-q', 'pcb-s', 'pcb-r'],
                     'dx': [0, 0, 0, 0, 0, 0, 0],
                     'dy': [0.9448807,
                      0.9448819,
                      0.9448819,
                      0.9448819,
                      0.9448819,
                      0.9448819,
                      0.9448819],
                     'nx': [1, 1, 1, 1, 1, 1, 1],
                     'ny': [14, 14, 14, 14, 14, 14, 14],
                     'parent_step': ['panel'],
                     'profile': [[0.0, 0.0],
                      [0.0, 12.8346291],
                      [6.2204642, 12.8346291],
                      [6.2204642, 0.0],
                      [0.0, 0.0]],
                     'sr_profile_lst': [[[4.7834687, 2.5393783],
                       [4.7834687, 15.3740074],
                       [11.0039329, 15.3740074],
                       [11.0039329, 2.5393783],
                       [4.7834687, 2.5393783]]],
                     'step_name': 'valor'}
        -----------------------------------------------------    
        
        return polygon_dict: dict 
    
            standard format:
                {step1 : [[[p1], [p2], ..., [p3], ]],
                 step2 : [[[p1], [p2], ..., [p3], ],   
                          [[p1], [p2], ..., [p3], ],
                          [[p1], [p2], ..., [p3], ]]}
                -> step2 is a multi panel step.
                         ex: pcb or sub.
                step2:[ polygon1,
                        polygon2,
                        polygon3 ]
        '''
        
        polygon_dict = {}
        for step, step_dict in all_step_dict.items():
            polygon_dict[step] = step_dict['sr_profile_lst']
        return polygon_dict   
            
    def get_containing_polygon(self, point):
        '''
        Purpose: To return a list which shows  all step in polygon_dict,
                 that the point is in.
        Input:
            point : [float, float] -> a single point.
            point = [5,7.5]
        return : list of steps : [str, str,...]
                 -> step in self.polygon_dict that contains input point.
        '''
        
        containing_point_step_lst = []
        for step, polygon_lst in self.polygon_dict.items():
            is_in_polygon = False
            for polygon in polygon_lst:
                is_in_polygon = self.point_in_polygon(polygon, point)
                if is_in_polygon :
                    containing_point_step_lst.append(step)
                    break

        

        return containing_point_step_lst
    
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
        if polygon_name not in self.polygon_dict.keys():
            return False #polygon_name not exist.
        
        polygon_lst = self.polygon_dict[polygon_name]
        is_in_step  = False
        for polygon in polygon_lst:
            is_in_polygon = self.point_in_polygon(polygon, point)
            if is_in_polygon :
                is_in_step = True
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


#%%
def main_plot(polygon_dict = None, polygon_lst = None, point_lst = []):
    '''
    this function is only for checking result.
    in other words, this function used only in dev mode.
    import matplotlib.pyplot as plt
    '''
    #檢視
    
    color_lst= ['blue', 'red', 'green', 'purple']
    color_dict = {'panel' : 'black',
                  'valor' : 'green',
                  'sub'   : 'purple',
                  'pcb'   : 'blue'}
    i = 0
    
    plt.figure(figsize=(20, 20))
        
    if point_lst:
        for point in point_lst:
            plt.plot(point[0], point[1], "o", color = 'red')
    if polygon_dict:
        for step, step_dict in polygon_dict.items():   # loop all step
            if step in color_dict.keys():
                color = color_dict[step]
            else:
                color_index = i % len(color_lst)
                color = color_lst[color_index]
            for polygon in step_dict['sr_profile_lst']:# loop all polygon in selected step
                x_lst, y_lst = list(zip(*polygon))
#                plt.plot(x_lst, y_lst, "o", color = 'black',markersize=3)
                plt.plot(x_lst,y_lst,"-",color = color )
                
            i += 1
    if polygon_lst:
        color_index = i % len(color_lst)
        color = color_lst[color_index]
        for polygon in polygon_lst:        # loop all polygon in selected step
            x_lst, y_lst = list(zip(*polygon))
            plt.plot(x_lst, y_lst, "o", color = 'black',markersize=3)
            plt.plot(x_lst,y_lst,"-",color = color )
            
        i += 1
    # plt.xticks(range(-10,10,1),size = 30)
    # plt.yticks(range(-10,10,1),size = 30)

    # axis_max = max(int(max(x_lst))+5,int(max(y_lst))+5)
    # axis_min = min(int(min(x_lst))-5,int(min(y_lst))-5)    
    # plt.xticks(range(axis_min,axis_max,5))
    # plt.yticks(range(axis_min,axis_max,5))
    plt.grid(color='r', linestyle='dotted', linewidth=1)
    plt.gca().set_aspect("equal")
    plt.savefig('test.png',c='c')



#%%




### Step profile data - pcb ###
#S P 0
#OB 10.5708661 0.4724409 I
#OC 10.492126 0.3937008 10.5708661 0.3937008 N
#OS 10.492126 0
#OS 1.1811024 0
#OS 1.1811024 0.3937008
#OC 1.1023622 0.4724409 1.1023622 0.3937008 N
#OS 0 0.4724409
#OS 0 7.7559055
#OS 12.6771654 7.7559055
#OS 12.6771654 0.4724409
#OS 10.5708661 0.4724409
#OE        






#ori_all_step_dict = copy.deepcopy(all_step_dict)



if __name__ == '__main__':
    '''
    os.environ['JOB'] = 'patrick-mtkx-md000l-a1d01-panel-multisub'
    os.environ['JOB'] = 'patrick-mtkx-md001l-a1d01-panel-multipcb'
    os.environ['JOB'] = 'patrick-gucx-pd001j-v1d01-panel-tilted'
    os.environ['JOB'] = 'patrick-qctx-pc002y-panel-valorsub'
    os.environ['JOB'] = 'pang-am0002-v1d01-depp'
    os.environ['JOB'] = 'athr-am0002-v1d01-juri'
    os.environ['JOB'] = 'qctx-ot0007'
    os.environ['JOB'] = 'qctx-pc001w-panel-routcnc'
    
    os.environ['JOB'] = 'tsmc-st007t-a2d01-juri'
    '''
    
    import matplotlib.pyplot as plt

    job = genClasses.Job(os.environ['JOB'])    
    matrix_info =  genClasses.Matrix(job).getInfo()
    step_object = genClasses.Step(job, 'panel')  
    #create input step_dict
    all_step_dict = {}
    sr_step = 'panel'
    all_step_dict = parse_sr_info_to_all_step_dict(job, sr_step, all_step_dict)

    #plot for check.
    main_plot(polygon_dict = all_step_dict, point_lst=new_point_lst)
    
    #create object
    AllStepObj = SRStepSet(all_step_dict)
    
    #=================== test part ===================#
    #Note : 
        #position in AllStepObj is in panel coordinate(the very parent one)
    #test point 
    candi_point = [[2.1591, 0.86119], [2.06496, 0.59055], [2.4164901000000003, 0.8501199999999999], [2.6338600000000003, 0.59055], [2.362205, 0.7874]]
    #function 1. get step that contain test point
    start = time.time()
    for i in range(6000):
        contain_step_lst = AllStepObj.get_containing_polygon(candi_point)
    end = time.time()
    diff = end - start
    
    #funtcion 2. check if test point in specific step.
    is_in_step = AllStepObj.check_if_point_in_step(candi_point, 'pcb')
    #function 3. to convert position of test point from source step into target step
    new_point_lst = AllStepObj.step_coordinate_conversion(target_step = 'panel',
                                                          source_step = 'pcb',
                                                          point_lst = candi_point)
    print(contain_step_lst, is_in_step, new_point_lst)
    tmp_dict = {}
    new_point_lst
    tmp_dict = {'sub':all_step_dict['sub']}
    main_plot(tmp_dict)

#    main()
    feature_dict={'test':candi_point}
    all_step_dict={}
    job_name = 'depp_test'
    r = requests.post("http://10.12.20.149:3113/cam/post_extract_data",
                       data = {"extracted_type": 'plate_thick',
                               "job_name": job_name,
                               "all_step_dict": str(all_step_dict),
                               'feature_dict' : str(feature_dict)} )
       
     


















5