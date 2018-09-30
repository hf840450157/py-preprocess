# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:49:22 2016

@author: 贾瑶
"""

import numpy as np
import os
import tarfile
import cv2

class Data_info():
    def __init__(self, filename):
        self.filename = filename
        self.nodes_names = None
        self.process_direction = -1
        self.src = None
        self.dst = None
        self.ColorType = -1
        self.begin_num = -1
        self.end_num = -1
        self.is_projection = -1
        self.thick_projection = -1
        self.is_resampling = -1
        self.reciprocal_scale = None
        self.frame_info_init = None
        self.frame_info = None
        self.is_build_file = 1
        self.redundancy_pixels = 0
        self.is_reReverse = 1
        self.trs_about = [0,0]
        self.changeSize = [1.0,1.0]
        self.xWidth = -1
        self.yWidth = -1
        self.maxFrameInfo = []
        
        #######
        self.dataType = -1
        self.pre_frame = None
        self.post_frame = None
        self.Img_height0 = -1
        self.Img_width0 = -1
        self.Img_height = -1
        self.Img_width = -1
        self.re_width = -1
        self.re_height = -1
        

        
        self.get_info()
        self.adj_info()
    
    def get_info(self):
        data_file = open(self.filename, 'r')
        i = 0
        for line in data_file:
            if i == 2:
                self.process_direction = int(line.strip())
            if i == 0:
                self.src = line.strip()
            if i == 1:
                self.dst = line.strip()
            if i == 3:
                self.ColorType = int(line.strip())
            if i == 5:
                zRange = [int(k) for k in line.strip().split(',')]
                self.begin_num = zRange[0]
                self.end_num = zRange[1]
            if i == 10:
                self.is_projection = bool(int(line.strip()))
            if i == 11:
                self.thick_projection = int(line.strip())
            if i == 12:
                self.is_resampling = bool(int(line.strip()))
            if i == 13:
                self.reciprocal_scale = [float(j) for j in line.strip().split(',')]
            if i == 6:
                self.frame_info_init = [int(k) for k in line.strip().split(',')]
            if i == 7:
                self.redundancy_pixels = int(line.strip())
            if i == 4:
                self.is_reReverse = int(line.strip())
            if i == 9:
                self.trs_about = [int(k) for k in line.strip().split(',')]
            if i == 8:
                self.changeSize = [float(k) for k in line.strip().split(',')]
            if i == 14:
                self.yWidth = int(line.strip().split(',')[0])
                self.xWidth = int(line.strip().split(',')[1])
            i += 1
        data_file.close()
        frame_file = open('mat.txt')
        frameInfo = []
        for line in frame_file:
            temp_frame = []
            for j in line.strip().split(','):
                temp_frame.append(int(j))
            frameInfo.append(temp_frame)
        frame_file.close()
        
        temp_frame_info = np.array(frameInfo)
        temp = temp_frame_info[:,2].copy()
        temp_frame_info[:,2] = temp_frame_info[:,3]
        temp_frame_info[:,3] = temp_frame_info[:,4]
        temp_frame_info[:,4] = temp
        self.frame_info = temp_frame_info
        i = 0
        for i in range(np.size(self.frame_info,0)-1):
            self.frame_info[i+1,1:3] += self.frame_info[i,1:3]
        self.get_parameter()

    def get_parameter(self):
        self._separator = '/'
        tempStr = [NameStr for NameStr in self.dst.split('/')]
        self.ddst = "/dev/shm/dtep"+"%04d"%(0)+NameStr+tempStr[len(tempStr)-2]+tempStr[len(tempStr)-1]
        #self.ddst = "W:/yjia/python/flowwork-20160507/"+"%04d"%(rank)+tempStr[len(tempStr)-2] + tempStr[len(tempStr)-1]
        if os.path.exists(self.ddst):
            os.system("rm -rf "+ self.ddst)
        os.mkdir(self.ddst)
        xStr = "%d"%(self.frame_info_init[0])#40
        yStr = "%d"%(self.frame_info_init[2])#31
        #obtain xWith/yWith dataType of detar image
        i = self.begin_num
        while 1:
            numStr = "%05d"%(i)
            #print numStr
            dfile_name = self.src + '/' + numStr + '.tar'
            try:
                tar = tarfile.open(dfile_name)
            except IOError:
                i = i+1
                continue
            else:
                try:
                    names = tar.getnames()
                    for name in names:
                        tar.extract(name, self.ddst + self._separator)
                finally:
                    #print "close"
                    tar.close()
            nameList = os.listdir(self.ddst + self._separator + numStr + self._separator)
            temname = nameList[1].split('.')
            self.post_frame = '.'+temname[len(temname)-1]
            temname = nameList[1].split('_')
            ssssi = 1
            if temname[1] == '':
                ssssi = 2
            temname = nameList[1].split(temname[ssssi])
            self.pre_frame = temname[0]
            filename = self.ddst + self._separator + numStr + self._separator + self.pre_frame + numStr +"_" + xStr + "_" + yStr + self.post_frame
            subimg = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
            if subimg == None:
                i = i+1
            else:
                break
        if((not self.xWidth == subimg.shape[0]) or (not self.yWidth == subimg.shape[1])):
            print "image size information is ood"
        self.xWidth = subimg.shape[0]
        self.yWidth = subimg.shape[1]
         
        if type(subimg[0,0]) == np.uint8:
            self.dataType = np.uint8
        else:
            self.dataType = np.uint16
        if os.path.exists(self.ddst):
            os.system("rm -rf "+ self.ddst)
        
    ##############
    def adj_info(self):
        print self.frame_info
        self.yWidth = self.yWidth-self.redundancy_pixels
        self.maxFrameInfo.extend(list(np.min(self.frame_info[:,1:2],0)))
        self.maxFrameInfo.extend(list(np.max(self.frame_info[:,2:3],0)))
        a = np.array([self.yWidth,self.xWidth])
        #print self.frame_info[:,3:5]
        b = (self.frame_info[:,3:5]-np.array([self.frame_info_init[0]-1,self.frame_info_init[2]-1]))*a
        b[:,0] = b[:,0]-self.frame_info[:,2]
        b[:,1] = b[:,1]+self.frame_info[:,1]
        self.maxFrameInfo.extend(np.max(b,0))
        
        self.xtransMax = self.maxFrameInfo[1]
        self.ytransMax = self.maxFrameInfo[0]
        self.xMaxEnd = self.maxFrameInfo[2]
        self.yMaxEnd = self.maxFrameInfo[3]
#        print self.Img_height0
#        print self.yMaxEnd
#        print self.ytransMax
#        print self.Img_width0
#        print self.xMaxEnd
#        print self.xtransMax

        if self.ytransMax>=0:
            self.ytransMax = 0
        if self.xtransMax<=0:
            self.xtransMax = 0
            
        self.Img_height0 = self.yMaxEnd-self.ytransMax
        self.Img_width0 = self.xMaxEnd+self.xtransMax
        
        self.Img_height = self.Img_height0
        self.Img_width = self.Img_width0

        if not self.changeSize[0] == self.changeSize[1] == 1:
#            if self.changeSize[0] == 0 or self.changeSize[1] == 0:
#                print 'parameter "changeSize" is not right'
#                return False
#            if not self.redundancy_pixels == 0:
#                print 'parameter "changeSize " and "redundancy_pixels" is conflictive, cant be set simultaneously'
#                return False
            self.Img_width = int(round(self.Img_width/self.changeSize[0]))
            self.Img_height = int(round(self.Img_height/self.changeSize[1]))
        
        if self.is_resampling == True:
            self.re_width = int(round(self.Img_width /self.reciprocal_scale[0]))
            self.re_height = int(round(self.Img_height / self.reciprocal_scale[1]))
            
        return True
        
                    
                
                
        