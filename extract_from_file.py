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
import math
os.chdir(r'C:\Users\User\Desktop\Python\CamDataExtractor')

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
sr_step = 'sub'
def get_prof_from_odb(job_name, sr_step):
    
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
            profile.pop()
            last_line_text = profile_line_text[i-1]
            print('oc')
            arc_point_lst = get_point_2(last_line_text, line_text,0.01)
#            arc_point_lst.reverse()
            profile.extend(arc_point_lst)
    return profile
def get_point_2(last_line_text, line_text, cm):

    '''
    目的 : 找所有交點
    input : 
        cm - 每個點的直線距離要幾公分
        layer_name - 層別名
    '''
#    layer_name = 'xray'  
#    cm = 0.01
#    
#    
    strigh_distance = cm * 0.3937     # 1 cm = 0.3937 inch
#
#    #讀取arc資訊
#    arc_feature = step_object.INFO("-t layer  -e {0}/{1}/{2} -d FEATURES -m script,units=inch ".format(step_object.job.name, step_object.name, layer_name))    
#    
#    #順y 逆n 
##        
#    xs = float(arc_feature[1].split()[1]) #起始點
#    ys = float(arc_feature[1].split()[2])
#    xe = float(arc_feature[1].split()[3]) #終點
#    ye = float(arc_feature[1].split()[4])
#    xc = float(arc_feature[1].split()[5]) #圓心
#    yc = float(arc_feature[1].split()[6])    
#    clockwise = str(arc_feature[1].split()[10]) #順逆時針
    
#    last_line_text = '#OS 0.1165349 0.866142 '
#    line_text = '#OC 0.1165349 1.003937 0.1165349 0.9350396 N '
    
    xs = float(last_line_text.split()[1]) #起始點
    ys = float(last_line_text.split()[2])
    xe = float(line_text.split()[1]) #終點
    ye = float(line_text.split()[2])
    xc = float(line_text.split()[3]) #圓心
    yc = float(line_text.split()[4])    
    clockwise = str(line_text.split()[5]) #順逆時針

    #轉換爲極座標角度
    theta_1 = math.atan2(ys-yc,xs-xc)/math.pi*180  #起點
    theta_2 = math.atan2(ye-yc,xe-xc)/math.pi*180  #終點
    
     
    '''step 1. 求起始點到圓心點距離 = 半徑'''
    r = (((xs-xc)**2)+((ys-yc)**2))**0.5
    
    '''step 2. 求弧度與弧長'''
    #整段資訊  
    if xs == xe and ys == ye: #剛好一整個圓
        L = 2 * r * math.pi
        alpha = 360
    else: #為圓弧
        D = (((xs-xe)**2) + ((ys-ye)**2))**0.5 
        #弧度
        if abs(theta_1 - theta_2) == 180:
            alpha = 180
        else:
            cos_alpha = ((r**2) + (r**2) - (D**2)) / (2 * r * r)
            alpha = math.acos(cos_alpha) # is radians
            alpha = math.degrees(alpha) # radians -> degrees
           
        if clockwise == 'Y': #順時針         2 * math.degrees(math.pi) = 360
            if theta_1 >= 0 and theta_2 >= 0:
                if theta_1 > theta_2:
                    alpha = alpha
                else:
                    alpha = 360 - alpha
             
            if theta_1 >= 0 and theta_2 < 0:
                if theta_1 - theta_2 <= 180:
                    alpha = alpha
                else:
                    alpha = 360 - alpha
                    
            if theta_1 < 0 and theta_2 >= 0:
                if 360 + theta_1 - theta_2 <= 180:
                    alpha = alpha
                else:
                    alpha = 360 - alpha    
                    
            if theta_1 < 0 and theta_2 < 0:
                if theta_1 > theta_2:
                    alpha = alpha
                else:
                    alpha = 360 - alpha
        
        if clockwise == 'N': #逆時針
            if theta_1 >= 0 and theta_2 >= 0:
                if theta_1 < theta_2:
                    alpha = alpha
                else:
                    alpha = 360 - alpha
            
            if theta_1 >= 0 and theta_2 < 0:
                if 360 - theta_1 + theta_2 <= 180:
                    alpha = alpha
                else:
                    alpha = 360 - alpha
                    
            if theta_1 < 0 and theta_2 >= 0:
                if abs(theta_1 - theta_2) <= 180:
                    alpha = alpha
                else:
                    alpha = 360 - alpha    
                    
            if theta_1 < 0 and theta_2 < 0:
                if theta_1 < theta_2:
                    alpha = alpha
                else:
                    alpha = 360 - alpha
        #整段弧長L
        L = alpha * r * math.pi / 180
  
    #分段資訊
    #弧度
    cos_theta = ((r**2) + (r**2) - (strigh_distance**2)) / (2 * r * r)
    theta = math.acos(cos_theta) # is radians  
    theta = math.degrees(theta) # radians -> degrees
    
    #每段的弧長
    l = theta * r * math.pi / 180
        
    #總共要分幾段
    m = L / l 
    m = int(m)
        
    '''step 3. 求分段點'''
    point_lst = []
    theta = math.radians(theta)   #degrees -> radians
    if  clockwise == 'Y':
        for i in range (0, m+1):       
            x =  xc - (r * math.sin(theta*i)) 
            y =  yc - (r * math.cos(theta*i))         
            point_lst.append([x, y])       
#        point_lst[i].append(math.atan2(point_lst[i][1]-yc,point_lst[i][0]-xc)/math.pi*180)
    else:
        for i in range (0, m+1):       
            x =  xc - (r * math.sin((-theta)*i)) 
            y =  yc - (r * math.cos((-theta)*i))         
            point_lst.append([x, y])

    '''step 4. 旋轉'''    
    x1 =  xc - (r * math.sin(theta*0)) 
    y1 =  yc - (r * math.cos(theta*0))      
    d = (((x1-xs)**2) + ((y1-ys)**2))**0.5
    cos_delta = ((r**2) + (r**2) - (d**2)) / (2 * r * r)
    delta = math.acos(cos_delta) # 旋轉角度--radian
    
    
    '''---------------------------------------------------------'''
#    step_object.COM('add_pad,attributes=no,x={0},y={1},symbol=r10,polarity=positive,angle=0,mirror=no,nx=1,ny=1,dx=0,dy=0,xscale=1,yscale=1'.format(xs, ys)) 
#    step_object.COM('add_pad,attributes=no,x={0},y={1},symbol=r10,polarity=positive,angle=0,mirror=no,nx=1,ny=1,dx=0,dy=0,xscale=1,yscale=1'.format(x1, y1)) 

    

    #旋轉角度

    theata_original_start = math.atan2(ys-yc,xs-xc)/math.pi*180  #原始起點
    theata_tmp_start = math.atan2(y1-yc,x1-xc)/math.pi*180  #暫時起點
    
    delta = math.degrees(delta) # radians -> degrees      
    
    if clockwise == 'Y': #順時針         2 * math.degrees(math.pi) = 360
        if theata_tmp_start >= 0 and theata_original_start >= 0:
            if theata_tmp_start > theata_original_start:
                delta = delta
            else:
                delta = 360 - delta
         
        if theata_tmp_start >= 0 and theata_original_start < 0:
            if theata_tmp_start - theata_original_start <= 180:
                delta = delta
            else:
                delta = 360 - delta
                
        if theata_tmp_start < 0 and theata_original_start >= 0:
            if 360 + theata_tmp_start - theata_original_start <= 180:
                delta = delta
            else:
                delta = 360 - delta    
                
        if theata_tmp_start < 0 and theata_original_start < 0:
            if theata_tmp_start > theata_original_start:
                delta = delta
            else:
                delta = 360 - delta
    
    if clockwise == 'N': #逆時針
        if theata_original_start >= 0 and theata_tmp_start >= 0:
            if theata_original_start < theata_tmp_start:
                delta = delta
            else:
                delta = 360 - delta
        
        if theata_original_start >= 0 and theata_tmp_start < 0:
            if 360 - theata_original_start + theata_tmp_start <= 180:
                delta = delta
            else:
                delta = 360 - delta
                
        if theata_original_start < 0 and theata_tmp_start >= 0:
            if abs(theata_original_start - theata_tmp_start) <= 180:
                delta = delta
            else:
                delta = 360 - delta    
                
        if theata_original_start < 0 and theata_tmp_start < 0:
            if theata_original_start < theata_tmp_start:
                delta = delta
            else:
                delta = 360 - delta
    
    
    
    delta = math.radians(delta)  
    '''---------------------------------------------------------'''
    
    
    
#    if  clockwise == 'Y':
#        for n in range(len(point_lst)):
#            rx = (point_lst[n][0]-xc) * math.cos(delta) + (point_lst[n][1] - yc) * math.sin(delta) + xc
#            ry = - (point_lst[n][0]-xc) * math.sin(delta) + (point_lst[n][1] - yc) * math.cos(delta) + yc
#            rotated_point_lst.append([rx, ry])
#            rotated_point_lst[n].append(math.atan2(rotated_point_lst[n][1]-yc,rotated_point_lst[n][0]-xc)/math.pi*180)
#    else:
#        for n in range(len(point_lst)):
#            rx = (point_lst[n][0]-xc) * math.cos(delta) - (point_lst[n][1] - yc) * math.sin(delta) + xc
#            ry = (point_lst[n][0]-xc) * math.sin(delta) + (point_lst[n][1] - yc) * math.cos(delta) + yc
#            rotated_point_lst.append([rx, ry])
#            rotated_point_lst[n].append(math.atan2(rotated_point_lst[n][1]-yc,rotated_point_lst[n][0]-xc)/math.pi*180)
    

    rotated_point_lst = []
    for n in range(len(point_lst)):
        rx = (point_lst[n][0]-xc) * math.cos(delta) + (point_lst[n][1] - yc) * math.sin(delta) + xc
        ry = - (point_lst[n][0]-xc) * math.sin(delta) + (point_lst[n][1] - yc) * math.cos(delta) + yc
        rotated_point_lst.append([rx, ry])
        rotated_point_lst[n].append(math.atan2(rotated_point_lst[n][1]-yc,rotated_point_lst[n][0]-xc)/math.pi*180)
   
    #排除極接近端點的
    for n in range(len(rotated_point_lst)):
        rotated_point_lst[n][2] = round(rotated_point_lst[n][2],3)

            
            
#    rotated_point_lst.append([xe, ye])

    rotated_point_lst_pos = []
    rotated_point_lst_neg = []

    for m in range(len(rotated_point_lst)):
        if rotated_point_lst[m][2] >= 0:
            rotated_point_lst_pos.append(rotated_point_lst[m])
        else:
            rotated_point_lst_neg.append(rotated_point_lst[m])
            
    
    
#    rotated_point_lst_pos = sorted(rotated_point_lst_pos, key = lambda s : s[2])
#    rotated_point_lst_neg = sorted(rotated_point_lst_neg, key = lambda s : s[2])  #由小到大
#    
#    rotated_point_lst_pos = sorted(rotated_point_lst_pos, key = lambda s : s[2], reverse = True)
#    rotated_point_lst_neg = sorted(rotated_point_lst_neg, key = lambda s : s[2], reverse = True)  #由大到小
#    

    final_sorted_rotated_point = []
    if clockwise == 'Y':
        rotated_point_lst_pos = sorted(rotated_point_lst_pos, key = lambda s : s[2], reverse = True)
        rotated_point_lst_neg = sorted(rotated_point_lst_neg, key = lambda s : s[2], reverse = True)  #由大到小
        
        if theta_1 >= 0:
            final_sorted_rotated_point.extend(rotated_point_lst_pos)
            final_sorted_rotated_point.extend(rotated_point_lst_neg)
        else:
            final_sorted_rotated_point.extend(rotated_point_lst_neg)
            final_sorted_rotated_point.extend(rotated_point_lst_pos)
    else:
        
        rotated_point_lst_pos = sorted(rotated_point_lst_pos, key = lambda s : s[2])
        rotated_point_lst_neg = sorted(rotated_point_lst_neg, key = lambda s : s[2])  #由小到大
        
        if theta_1 >= 0:
            final_sorted_rotated_point.extend(rotated_point_lst_pos)
            final_sorted_rotated_point.extend(rotated_point_lst_neg)
        else:
            final_sorted_rotated_point.extend(rotated_point_lst_neg)
            final_sorted_rotated_point.extend(rotated_point_lst_pos)

    final_sorted_rotated_point.append([xe, ye, theta_2])

    return final_sorted_rotated_point


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
        dx = parent_step_dict['dx'][index]
        dy = parent_step_dict['dy'][index]
        ax = parent_step_dict['ax'][index]
        ay = parent_step_dict['ay'][index]
        nx = parent_step_dict['nx'][index]
        ny = parent_step_dict['ny'][index]
        

        datum_point = [ax, ay]
        #loop nx,y time
        
        for child_profile in child_profile_lst:
            for x_time in range(nx):
                for y_time in range(ny):
                    #each time create new polygon, and add dx or dy on each point.
                    new_polygon = list(map(lambda x:[x[0] + dx * x_time + datum_point[0], x[1] + dy * y_time + + datum_point[1]], child_profile))
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
    '''
    
    '======================================================'
    standard_step_dict = {'step_name': '',
                         'profile'  : [],
                         'sr_profile_lst'  : [],
                         'child_step' : [],
                         'parent_step': [],
                         'angle' : [float],
                         'dx' : [float],
                         'dy' : [float],
                         'ax' : [float],
                         'ay' : [float],
                         'nx' : [float],
                         'ny' : [float]}

    ### get self infomation -> profile###
    #sr_step = 'panel'
    ## get and parse profile ##
#    step_prof_info = job.INFO('-t step -e {0}/{1} -d PROF'.format(job.name, sr_step))
#    step_profile = parse_profile_info(step_prof_info)
        
    step_profile = get_prof_from_odb(job_name, sr_step)
    
    ## get sr_infomation ##
#    temp_step_dict = job.DO_INFO('-t step -e {0}/{1} -d SR'.format(job.name, sr_step ))
    temp_step_dict = get_sr_dict_from_odb(job_name, sr_step)
    
    
    #-----------------------------------#
    if temp_step_dict ['gSRstep'] == []: #the input step is the last child step.
        sr_step_dict = all_step_dict[sr_step]
        sr_step_dict['profile']   = step_profile

        
        sr_profile_lst = [step_profile]
        ref_step = sr_step
        for parent_step in all_step_dict[sr_step]['parent_step']:
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
        for child_step_name in set(all_step_dict[sr_step]['child_step']): #loop all child step.

#            if child_step_name not in all_step_dict: #create child dict in all_step_dict if not exist
            child_step_dict = copy.deepcopy(standard_step_dict)
            all_step_dict[child_step_name] = child_step_dict
            
            
            if sr_step not in all_step_dict[child_step_name]['parent_step']:
                all_step_dict[child_step_name]['parent_step'].append(sr_step)
                relative_parent_lst = all_step_dict[sr_step]['parent_step']
                all_step_dict[child_step_name]['parent_step'].extend(relative_parent_lst )
            all_step_dict = parse_sr_info_to_all_step_dict(job_name, child_step_name, all_step_dict )

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



#%%plotter

import matplotlib.pyplot as plt
def main_plot(polygon_dict = None, polygon_lst = None, point_lst = []):
    '''
    this function is only for checking result.
    in other words, this function used only in dev mode.
    '''
    #檢視
    
    
    color_lst= ['blue', 'red', 'green', 'purple']
    i = 0
    
    plt.figure(figsize=(20, 20))
        
    if point_lst:
        for point in point_lst:
            plt.plot(point[0], point[1], "o", color = 'red')
    if polygon_dict:
        for step, polygon_lst_in_dict in polygon_dict.items():   # loop all step
            color_index = i % len(color_lst)
            color = color_lst[color_index]
            for polygon in polygon_lst_in_dict['sr_profile_lst']:# loop all polygon in selected step
                x_lst, y_lst = zip(*polygon)
#                plt.plot(x_lst, y_lst, "o", color = 'black',markersize=3)
                plt.plot(x_lst,y_lst,"-",color = color )
                
            i += 1
    if polygon_lst:
        color_index = i % len(color_lst)
        color = color_lst[color_index]
        for polygon in polygon_lst:        # loop all polygon in selected step
            x_lst, y_lst = zip(*polygon)
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
#%%

def main(job_name):
    print('hi')
   
    ori_all_step_dict = {}
    sr_step = 'panel'
    ori_all_step_dict = parse_sr_info_to_all_step_dict(job_name, sr_step, ori_all_step_dict )

    pass
#%%
if __name__ == "__main__":
    print("****************************************")
    print('************Program execute*************')
    print("****************************************")


    '''
    os.environ['JOB'] = 'MTKX-OT001C-A1D0'.lower()
    os.environ['JOB'] = 'pang-am0002-v1d01-juri'
    
    '''
    job_name = 'pang-am0008-xxxx'
    ori_all_step_dict = {}
    sr_step = 'panel'
    s = time.time()
    ori_all_step_dict = parse_sr_info_to_all_step_dict(job_name, sr_step, ori_all_step_dict )
    print(time.time() -s )
    
    s = time.time()
    n = 100000
    candi_list = [[x,y] for x , y in zip(range(n),range(n))]
    
    AllStepObj = SRStepSet(ori_all_step_dict )
    
    
    new_point_lst = AllStepObj.step_coordinate_conversion(target_step = 'sub',
                                                          source_step = 'pcb',
                                                          point_lst = candi_list )
    
    print(time.time() -s )
    
    main_plot(ori_all_step_dict )
    
    all_step_dict = {}
    sr_step = 'panel'
    all_step_dict = parse_sr_info_to_all_step_dict(job_name, sr_step, all_step_dict)
    #create object
    AllStepObj = SRStepSet(all_step_dict)
    
    
    # main_plot(ori_all_step_dict ,polygon_lst = new_point_lst)
    new_point_lst
    
    
    
    
    
    
    
    
    