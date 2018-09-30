import os
import tarfile
import cv2
import numpy as np


class Preprocess():
    def __init__(self, filename, mpisize):
        self.data = Data_info(filename)
        self._separator = '/'
        self.montage_file = self.data.dst + self._separator  + 'montage'
        self.projection_file = self.data.dst + self._separator  + 'projection'
        self.resample_file = self.data.dst + self._separator  + 'resample'
        self.tasks = None
        self.data.process_num = mpisize
        
        self.process_direction = self.data.process_direction
        self.pre_frame = None
        self.post_frame = None
        self.xBeg = self.data.frame_info[0]
        self.xEnd = self.data.frame_info[1]
        self.yBeg = self.data.frame_info[2]
        self.yEnd = self.data.frame_info[3]
        #staff about detar        
        self.dfile_pre = ''
        self.dfile_post = ".tar"
        
        self.writePre = "test_"
        self.writePost = "_mon.tif"
        #staff about projection        
        self.pwritePre = "test_"
        self.pwirtePost = "_pro.tif"
        
        self.ddst = None
        
        self.xWidth = -1
        self.yWidth = -1
        self.raw_width = -1
        self.raw_height = -1  
        
        self.img = None
        self.subimg1 = None
        self.proimg = None
        
        self.re_width = -1
        self.re_height = -1
        self.z_ratio_rec = -1.0
    
    def make_file(self):
        if os.path.exists(self.data.dst):
            #os.mkdir(detar_file_name)
            if not os.path.exists(self.montage_file):
                os.mkdir(self.montage_file)
            if self.data.is_projection and (not os.path.exists(self.projection_file)):
                os.mkdir(self.projection_file)
            if self.data.is_resampling and (not os.path.exists(self.resample_file)):
                os.mkdir(self.resample_file)
            return True
        else:
            return False
    
    def allocate_task(self):
        begin_num = self.data.zRange[0]
        end_num = self.data.zRange[1]
        process_num = self.data.process_num
        thickness = self.data.thick_projection
        self.tasks = [[] for iii in range(process_num)]
        if (not self.data.is_projection) or (not thickness):
            thickness = 5
        num_each_round = thickness
        index = 0
        set_end = False
        while index <= end_num-begin_num:
            for i in xrange(process_num):
                tmp_task_set = []
                for k in xrange(num_each_round):               
                    tmp_task_set.append(begin_num + index)
                    index += 1
                    if begin_num + index == end_num + 1:
                        set_end = True
                        break
                self.tasks[i].extend(tmp_task_set)
                if set_end:
                    break
        return True

    def get_task_set(self, r):
        self.allocate_task()
        return self.tasks[r]
        
    def set_parameter(self, task, rank):
        tempStr = [NameStr for NameStr in self.data.dst.split('/')]
        self.ddst = "/dev/shm/dtep"+"%04d"%(rank)+NameStr+tempStr[len(tempStr)-2]+tempStr[len(tempStr)-1]
        #self.ddst = "W:/yjia/python/flowwork-20160507/"+"%04d"%(rank)+tempStr[len(tempStr)-2] + tempStr[len(tempStr)-1]
        if os.path.exists(self.ddst):
            os.system("rm -rf "+ self.ddst)
        os.mkdir(self.ddst)
        
        
        xStr = "%d"%(40)
        yStr = "%d"%(31)
        #obtain xWith/yWith of detar image
        
        i = 0
        while 1:
            numStr = "%05d"%(task[i])
            dfile_name = self.data.src + self._separator + self.dfile_pre + numStr + self.dfile_post
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
        self.xWidth = subimg.shape[0]
        #yWidth1 = subimg.shape[1]
        self.yWidth = subimg.shape[1]-self.data.redundancy_pixels
        self.raw_height = self.xWidth*(self.yEnd-self.yBeg+1)
        self.raw_width = self.yWidth*(self.xEnd-self.xBeg+1)
        
        if self.data.image_depth == 8:
            self.img = np.zeros((self.raw_height,self.raw_width),np.uint8)
            self.subimg1 = np.zeros((self.xWidth,self.yWidth),np.uint8)
        elif self.data.image_depth == 16:
            self.img = np.zeros((self.raw_height,self.raw_width),np.uint16)
            self.subimg1 = np.zeros((self.xWidth,self.yWidth),np.uint16)

        tddir = self.ddst + self._separator + numStr
        os.system("rm -rf "+ tddir)

        if not self.data.changeSize[0] == self.data.changeSize[1] == 1:
            if self.data.changeSize[0] == 0 or self.data.changeSize[1] == 0:
                print 'parameter "changeSize" is not right'
                return False
            if not self.data.redundancy_pixels == 0:
                print 'parameter "changeSize " and "redundancy_pixels" is conflictive, cant be set simultaneously'
                return False
            self.raw_width = int(self.raw_width/self.data.changeSize[0])
            self.raw_height = int(self.raw_height/self.data.changeSize[1])
        
        self.z_ratio_rec = self.data.reciprocal_scale[2]
        if self.data.is_resampling == True:
            x_ratio = 1.0 / self.data.reciprocal_scale[0]
            y_ratio = 1.0 / self.data.reciprocal_scale[1]
            self.re_width = int(self.raw_width * x_ratio)
            self.re_height = int(self.raw_height * y_ratio)
            
        if self.data.is_projection == True:
            if self.data.image_depth == 8:
                self.proimg = np.zeros((self.raw_height,self.raw_width),np.uint8)
            elif self.data.image_depth == 16:
                self.proimg = np.zeros((self.raw_height,self.raw_width),np.uint16)
        return True

    def do_montage(self, task, rank):
        if len(task) == 0:
            return
        errorMontage = 0
        dtarerror = 0     
        projecting = 0
        for z in range(len(task)):
            s = task[z]
            numStr = "%05d"%(s)
            dfile_name = self.data.src + self._separator + self.dfile_pre + numStr + self.dfile_post
            if os.path.exists(dfile_name):
                try:
                    tar = tarfile.open(dfile_name)
                except IOError, e:
                    print e
                    edir = self.data.dst + self._separator + "keyError.txt"
                    file_object = open(edir, 'a+')
                    all_the_txt = numStr + "error\n" + e +'\n'
                    file_object.write(all_the_txt)
                    file_object.close( )
                    projecting = projecting+1
                    continue
                else:
                    try:
                        names = tar.getnames()
                        for name in names:
                            tar.extract(name, self.ddst + self._separator)
                    finally:
                        tar.close()
            else:
                dtarerror = 1
            for i in xrange(self.yBeg,self.yEnd+1):
                for j in xrange(self.xBeg,self.xEnd+1):
                    yStr = "%d"%(i)
                    xStr = "%d"%(j)
                    filename = self.ddst + self._separator + numStr + self._separator + self.pre_frame + numStr +"_" + xStr + "_" + yStr + self.post_frame   
                    subimg = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED)
                    if subimg == None or dtarerror == 1:
                        subimg = self.subimg1
                        if errorMontage == 0:
                            errorMontage = 1
                            edir = self.data.dst + self._separator + "error.txt"
                            file_object = open(edir, 'a+')
                            all_the_txt = numStr + "error\n"
                            file_object.write(all_the_txt)
                            file_object.close( )
                    if i == 31:
                        subimg[0:131,:] = 0
                    if self.data.is_reReverse == 1:
                        self.img[(i-self.yBeg)*self.xWidth:(i+1-self.yBeg)*self.xWidth,(j-self.xBeg)*self.yWidth:(j+1-self.xBeg)*self.yWidth] = subimg[0:self.xWidth,0:self.yWidth]
                    if self.data.is_reReverse == 0:
                        if self.process_direction == 1:
                            subimg = subimg[:,::-1]
                            self.img[(i-self.yBeg)*self.xWidth:(i+1-self.yBeg)*self.xWidth,(j-self.xBeg)*self.yWidth:(j+1-self.xBeg)*self.yWidth] = subimg[0:self.xWidth,0:self.yWidth]
                        else:
                            if (j-40)%2 == 0:
                                subimg = subimg[:,::-1]
                                self.img[(i-self.yBeg)*self.xWidth:(i+1-self.yBeg)*self.xWidth,(j-self.xBeg)*self.yWidth:(j+1-self.xBeg)*self.yWidth] = subimg[0:self.xWidth,0:self.yWidth]
                            else:
                                if i == 31:
                                    subimg = subimg[::-1,::-1]
                                    self.img[(self.yEnd-i)*self.xWidth+128:(self.yEnd+1-i)*self.xWidth,(j-self.xBeg)*self.yWidth:(j+1-self.xBeg)*self.yWidth] = subimg[0:self.xWidth-127,0:self.yWidth]
                                else:
                                    subimg = subimg[::-1,::-1]
                                    self.img[(self.yEnd-i)*self.xWidth+128:(self.yEnd+1-i)*self.xWidth+128,(j-self.xBeg)*self.yWidth:(j+1-self.xBeg)*self.yWidth] = subimg[0:self.xWidth,0:self.yWidth]
            #img[600:xWidth*(yEnd-yBeg+1),:] = img[0:xWidth*(yEnd-yBeg+1)-600,:]
            if self.data.changeSize[0] == self.data.changeSize[1] == 1:
                image = self.img
            else:
                image = cv2.resize(self.img, (self.raw_width, self.raw_height))
            
            if not self.data.trs_about[0] == self.data.trs_about[0] == 0:
                image = self.do_translation(image)
            print self.writePre + numStr + self.writePost
            cv2.imwrite(self.montage_file + self._separator + self.writePre + numStr + self.writePost,image)

            if self.data.is_resampling == True and (((s - task[0]) % self.z_ratio_rec) == 0):
                re_image = cv2.resize(image, (self.re_width, self.re_height))
                cv2.imwrite(self.resample_file + self._separator + "test_" + numStr + '_res.tif', re_image)

            tddir = self.ddst + self._separator + numStr
            os.system("rm -rf "+tddir)
            
            if self.data.is_projection == True:
                self.proimg = np.maximum(image,self.proimg)
                projecting = projecting + 1
                if projecting == self.data.thick_projection:
                    pfilename = self.projection_file + self._separator + self.pwritePre + "%05d"%(s-projecting+1) + "-" + "%05d"%(s) + self.pwirtePost
#                    print pfilename
                    cv2.imwrite(pfilename,self.proimg)
                    if self.data.image_depth == 8:
                        self.proimg = np.zeros((self.raw_height,self.raw_width),np.uint8)
                    elif self.data.image_depth == 16:
                        self.proimg = np.zeros((self.raw_height,self.raw_width),np.uint16)
                    projecting = 0
                elif z == len(task)-1:
                    pfilename = self.projection_file + self._separator + self.pwritePre + "%05d"%(s-projecting+1) + "-" + "%05d"%(s) + self.pwirtePost
#                    print pfilename
                    cv2.imwrite(pfilename,self.proimg)
                    if self.data.image_depth == 8:
                        self.proimg = np.zeros((self.raw_height,self.raw_width),np.uint8)
                    elif self.data.image_depth == 16:
                        self.proimg = np.zeros((self.raw_height,self.raw_width),np.uint16)
                    projecting = 0
            errorMontage = 0
            dtarerror = 0
            
    def do_translation(self, origImg):
        vertical = 0
        horizontal = 0
        vp = self.data.trs_about[1]
        if not vp == 0:
            vertical = 1
        if vp < 0:
            vertical = -1
            vp = -vp
        hp = self.data.trs_about[0]
        if not hp == 0:
            horizontal = 1
        if hp < 0 :
            horizontal = -1
            hp = -hp
        if vertical == horizontal == 0:
            return
        raw_width = self.raw_width
        raw_height = self.raw_height
        if self.data.image_depth == 8:
            tmpImg = np.zeros((raw_height,raw_width),np.uint8)
        elif self.data.image_depth == 16:
            tmpImg = np.zeros((raw_height,raw_width),np.uint16)
            
        if horizontal == 1:
            tmpImg[:,0:raw_width-hp] = origImg[:,hp:raw_width]
        if horizontal == -1:
            tmpImg[:,hp:raw_width] = origImg[:,0:raw_width-hp]

        if vertical == 1:
            tmpImg[0:raw_height-vp,:] = tmpImg[vp:raw_height,:]
            tmpImg[raw_height-vp:raw_height,:] = 0
        if vertical == -1:
            tmpImg[vp:raw_height,:] = tmpImg[0:raw_height-vp,:]
            tmpImg[0:vp,:] = 0
        return tmpImg


class Data_info():
    def __init__(self, filename):
        self.filename = filename
        self.process_direction = -1
        self.src = None
        self.dst = None
        self.image_depth = -1
        self.zRange = None
        self.is_projection = -1
        self.thick_projection = -1
        self.is_resampling = -1
        self.reciprocal_scale = None
        self.frame_info = None
        self.is_build_file = 1
        self.redundancy_pixels = 0
        self.is_reReverse = 1
        self.trs_about = [0,0]
        self.changeSize = [1.0,1.0]
        self.get_info()
    
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
                self.image_depth = int(line.strip())
            if i == 5:
                self.zRange = [int(k) for k in line.strip().split(',')]
            if i == 10:
                self.is_projection = bool(int(line.strip()))
            if i == 11:
                self.thick_projection = int(line.strip())
            if i == 12:
                self.is_resampling = bool(int(line.strip()))
            if i == 13:
                self.reciprocal_scale = [float(j) for j in line.strip().split(',')]
            if i == 6:
                self.frame_info = [int(k) for k in line.strip().split(',')]
            if i == 7:
                self.redundancy_pixels = int(line.strip())
            if i == 4:
                self.is_reReverse = int(line.strip())
            if i == 9:
                self.trs_about = [int(k) for k in line.strip().split(',')]
            if i == 8:
                self.changeSize = [float(k) for k in line.strip().split(',')]
            i += 1
        data_file.close()
