import os
import tarfile
import cv2
import numpy as np
import copy

class Preprocess():
    def __init__(self, filename,n):
        self.data = Data_info(filename)
        self.process_num = n
        self._separator = '/'
        self.tasks = None
        
    def isExist_file(self):
        if os.path.exists(self.data.dst):
            return True
        else:
            return False
    
    def allocate_task(self):
        begin_num = self.data.begin_num
        end_num = self.data.end_num
        taskType = self.data.taskType
        process_num = self.process_num
        thickness = self.data.thick_projection
        self.tasks = [[] for iii in range(process_num)]
        if taskType == 1 and self.data.thick_projection <= 0:
            print 'information about projection is incomplete or wrong'
            return False
        if taskType != 1:
            thickness = 5
        num_each_round = thickness
        index = 0
        set_end = False
        while index < end_num-begin_num+1:
            for i in xrange(process_num):
                tmp_task_set = []
                for k in xrange(num_each_round):               
                    tmp_task_set.append(begin_num + index)
                    if begin_num + index == end_num:
                        set_end = True
                        index += 1
                        break
                    index += 1
                self.tasks[i].extend(tmp_task_set)
                if set_end:
                    break
        return True
    
    def get_task_set(self, r):
        if self.allocate_task():
            return self.tasks[r]
        else:
            return []
    
    def do_projection(self, task):
        if len(task) == 0:
            return
        if self.data.thick_projection == -1 or self.data.thick_projection == 0:
            print 'information about projection is incomplete or wrong'
            return
        src = self.data.src
        dst = self.data.dst
        readPre = self.data.pre_frame
        readPost = self.data.post_frame
        writePre = "test_"
        wirtePost = "_pro.tif"
        blockNum = self.data.thick_projection
        sA = task
        for j in range(0,len(sA), blockNum):
            s = j
            e = j + blockNum - 1    
            if e > len(sA)-1:
                e = len(sA)-1
            #print "process begin: %05d  end : %05d"%(sA[s], sA[e])
            sS = "%05d"%(sA[s])
            eS = "%05d"%(sA[e])
            for i in range(s, e+1):
                n = sA[i]
                numStr = "%05d"%(n)
                filename = src + self._separator + readPre + numStr + readPost
                print readPre + numStr + readPost
                if i == s:
                    img1 = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
                    if self.data.image_depth == 8:
                        img3 = np.zeros(img1.shape,np.uint8)
                    elif self.data.image_depth == 16:
                        img3 = np.zeros(img1.shape,np.uint16)
                else:
                    img1 = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
                img3 = np.maximum(img1,img3)
            #print dst + self._separator + writePre + sS + "-" + eS + wirtePost
            cv2.imwrite(dst + self._separator +writePre+sS+"-"+eS+wirtePost,img3)
            
    def do_resample(self, task):
        if len(task) == 0:
            return
        if self.data.reciprocal_scale == None:
            print 'information about resample is incomplete or wrong'
            return
        src = self.data.src
        readPre = self.data.pre_frame
        readPost = self.data.post_frame
        dst = self.data.dst
        x_ratio = 1.0 / self.data.reciprocal_scale[0]
        y_ratio = 1.0 / self.data.reciprocal_scale[1]
        z_ratio_rec = self.data.reciprocal_scale[2]
        numStr = "%05d"%(task[0])
        filename = src + self._separator + readPre + numStr + readPost
        tepimg = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
        raw_width = tepimg.shape[1]
        raw_height = tepimg.shape[0]
        re_width = int(raw_width * x_ratio)
        re_height = int(raw_height * y_ratio)
        for z in range(len(task)):
            s = task[z]
            if (s % z_ratio_rec) == 0:
                filename = src + self._separator + readPre + "%05d"%(s) + readPost
                print readPre + "%05d"%(s) + readPost
                img = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
                re_image = cv2.resize(img, (re_width,re_height))
                cv2.imwrite(dst + self._separator + "test_" + "%05d"%(s) + '_res.tif', re_image)

    def do_crop(self, task):
        if len(task) == 0:
            return
        src = self.data.src
        readPre = self.data.pre_frame
        readPost = self.data.post_frame
        dst = self.data.dst
        x = self.data.crop_range[0]
        y = self.data.crop_range[1]
        w = self.data.crop_range[2]
        h = self.data.crop_range[3]
        for z in range(len(task)):
            s = task[z]
            numStr = "%05d"%(s)
            filename = src + self._separator + readPre + numStr + readPost
            savedName = dst + self._separator + 'test' + numStr + "_x%d_y%d_w%d_h%d"%(x,y,w,h) + '.tif'
            print readPre + numStr + readPost
            origImg = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
            tmpImg = origImg[y:y+h,x:x+w]
            cv2.imwrite(savedName,tmpImg)
    
    def do_translation(self, task):
        if len(task) == 0:
            return
        src = self.data.src
        readPre = self.data.pre_frame
        readPost = self.data.post_frame
        dst = self.data.dst
        if self.data.trs_about[0] == self.data.trs_about[1] == 0:
            return
        vertical = 1
        if self.data.trs_about[1] < 0:
            vertical = -1
        vp = vertical*self.data.trs_about[1]
        horizontal = 1
        if self.data.trs_about[0]<0:
            horizontal = -1
        hp = horizontal*self.data.trs_about[0]
        
        numStr = "%05d"%(task[0])
        filename = src + self._separator + readPre + numStr + readPost
        tepimg = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
        raw_width = tepimg.shape[0]
        raw_height = tepimg.shape[1]
        if self.data.image_depth == 8:
            tmpImg = np.zeros((raw_width,raw_height),np.uint8)
        elif self.data.image_depth == 16:
            tmpImg = np.zeros((raw_width,raw_height),np.uint16)
        for z in range(len(task)):
            s = task[z]
            numStr = "%05d"%(s)
            filename = src + self._separator + readPre + numStr + readPost
            savedName = dst + self._separator + readPre + numStr + '_tran.tif'
            print readPre + numStr + readPost
            origImg = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
            if vertical == 0:
                if horizontal == 1:
                    tmpImg[:,0:raw_height-hp] = origImg[:,hp:raw_height]
                if horizontal == -1:
                    tmpImg[:,hp:raw_height] = origImg[:,0:raw_height-hp]
            if horizontal == 0:
                if vertical == 1:
                    tmpImg[0:raw_width-vp,:] = origImg[vp:raw_width,:]
                if vertical == -1:
                    tmpImg[vp:raw_width,:] = origImg[0:raw_width-vp,:]
            if vertical == 1:
                if horizontal == 1:
                    tmpImg[0:raw_width-vp,0:raw_height-hp] = origImg[vp:raw_width,hp:raw_height]
                if horizontal == -1:
                    tmpImg[0:raw_width-vp,hp:raw_height] = origImg[vp:raw_width,0:raw_height-hp]
            if vertical == -1:
                if horizontal == 1:
                    tmpImg[vp:raw_width,0:raw_height-hp] = origImg[0:raw_width-vp,hp:raw_height]
                if horizontal == -1:
                    tmpImg[vp:raw_width,hp:raw_height] = origImg[0:raw_width-vp,0:raw_height-hp]                
            cv2.imwrite(savedName,tmpImg)

            
    def do_Img16to8(self, task):
        if len(task) == 0:
            return
        src = self.data.src
        readPre = self.data.pre_frame
        readPost = self.data.post_frame
        dst = self.data.dst
        for z in range(len(task)+1):
            s = task[z]-1
            numStr = "%05d"%(s)
            filename = src + readPre + numStr + readPost
            saveName = dst + readPre + numStr + readPost
            if os.path.exists(saveName):
                continue
            print " processing : " + readPre + numStr + readPost
            origImg = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
            mi = np.min(origImg)
            ma = np.max(origImg)
            scale = 256.0 / (ma - mi + 1)
            origImg = np.uint8((origImg - mi + 1) * scale)
            cv2.imwrite(saveName, origImg)
            #print saveName

class Data_info():
    def __init__(self, filename):
        self.filename = filename
        self.taskType = -1

        self.src = None
        self.dst = None
        self.image_depth = -1
        self.zRange = None   
        self.begin_num = -1
        self.end_num = -1

        self.thick_projection = -1
        self.reciprocal_scale = None
        self.crop_range = None
        self.trs_about = None
        self.pre_frame = None
        self.post_frame = None

        self.get_info()
        self.getName()
    
    def get_info(self):
        data_file = open(self.filename, 'r')
        i =  0 
        for line in data_file:
            if i == 0:
                self.taskType = int(line.strip())
                i+=1
                continue
            if  i==5:
                self.thick_projection = int(line.strip())
                i+=1
                continue
            if i == 6:
                self.reciprocal_scale = [float(j) for j in line.strip().split(',')]
                i+=1
                continue
            if i == 7:
                self.crop_range = [int(h) for h in line.strip().split(',')]
                i+=1
                continue
            if i == 8:
                self.trs_about = [int(w) for w in line.strip().split(',')]
                i+=1
                continue
            if i == 3:
                self.image_depth = int(line.strip())
                i+=1
                continue
            if i == 1:
                self.src = line.strip()
                i+=1
                continue
            if i == 2:
                self.dst = line.strip()
                i+=1
                continue
            if i == 4:
                self.zRange = [int(w) for w in line.strip().split(',')]
                self.begin_num = self.zRange[0]
                self.end_num = self.zRange[1]
                i+=1
                continue
            i += 1
        data_file.close()
    def getName(self):
        nameList = os.listdir(self.src)
        temname = nameList[1].split('_')
        self.post_frame = '_'+temname[len(temname)-1]
        self.pre_frame = temname[0]+'_'
        
        
        
        
        
        
        