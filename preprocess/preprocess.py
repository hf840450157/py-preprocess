import os
import tarfile
import cv2
import numpy as np

class Preprocess():
    def __init__(self, Data_info, mpisize):
        self.data = Data_info
        self._separator = '/'
        self.montage_file = self.data.dst + self._separator + 'montage'
        self.projection_file = self.data.dst + self._separator  + 'projection'
        self.projection_jpg = self.data.dst + self._separator  + 'projection_jpg'
        self.resample_file = self.data.dst + self._separator  + 'resample'
        self.tasks = None
        self.data.process_num = mpisize
        
        #staff about detar 
        self.ddst = None
        self.dfile_pre = ''
        self.dfile_post = ".tar"
        
        self.writePre = "test_"
        self.writePost = "_mon.tif"
        
        #staff about projection        
        self.pwritePre = "test_"
        self.pwirtePost = "_pro.tif" 
        self.rewritePre = "test_"
        self.rewirtePost = "_res.tif"
        
        self.xBeg = self.data.frame_info_init[0]#40
        self.yBeg = self.data.frame_info_init[2]#31
        self.xEnd = -1
        self.yEnd = -1
        self.dataType = self.data.dataType
        self.post_frame = self.data.post_frame
        self.pre_frame = self.data.pre_frame
        #imageSize about
        #tile size
        self.xWidth = self.data.xWidth
        self.yWidth = self.data.yWidth
        #result image size
        self.Img_width0 = self.data.Img_width0
        self.Img_height0 = self.data.Img_height0
        self.Img_width = self.data.Img_width
        self.Img_height = self.data.Img_height 
        self.re_width = self.data.re_width
        self.re_height = self.data.re_height
        self.z_ratio_rec = self.data.reciprocal_scale[2]
        
        self.img = None
        self.subimg1 = None #black image
        self.proimg = None
        
        
    
    def make_file(self):
        if os.path.exists(self.data.dst):
            #os.mkdir(detar_file_name)
            if not os.path.exists(self.montage_file):
                os.mkdir(self.montage_file)
            if self.data.is_projection and (not os.path.exists(self.projection_file)):
                os.mkdir(self.projection_file)
                os.mkdir(self.projection_jpg)
            if self.data.is_resampling and (not os.path.exists(self.resample_file)):
                os.mkdir(self.resample_file)
            return True
        else:
            return False
    
    def allocate_task(self):
        begin_num = self.data.begin_num
        end_num = self.data.end_num
        #print begin_num,end_num
        process_num = self.data.process_num
        thickness = self.data.thick_projection
        frameInfo = self.data.frame_info
        self.tasks = [[] for iii in range(process_num)]
        num_each_round = thickness
        if num_each_round == 0:
            num_each_round = 1
        starF = 0
        for starF in range(np.size(frameInfo,0)):
            if frameInfo[starF,0]<=begin_num and frameInfo[starF+1,0]>begin_num:
                break;
        index = 0
        set_end = False
        if end_num-begin_num < 0:
            return False
        while index <= end_num-begin_num:
            for i in xrange(process_num):
                tmp_task_set = []
                for k in xrange(num_each_round):
                    tmp_task = []

                    tmp_task.append(begin_num + index)
                    tmp_task.extend(list(frameInfo[starF,1:5])) 
                    tmp_task_set.append(tmp_task)
                    index += 1
                    if begin_num + index == end_num + 1:
                        set_end = True
                        break
                    if frameInfo[starF+1,0] == begin_num + index :
                        starF += 1;
                self.tasks[i].extend(tmp_task_set)
                if set_end:
                    break
        return True

    def get_task_set(self, r):
        if self.allocate_task():
            return self.tasks[r]
        else:
            return []
        
    def set_parameter(self, task, rank):
        tempStr = [NameStr for NameStr in self.data.dst.split('/')]
        self.ddst = "/dev/shm/dtep"+"%04d"%(rank)+NameStr+tempStr[len(tempStr)-2]+tempStr[len(tempStr)-1]
        #self.ddst = "W:/yjia/python/flowwork-20160507/"+"%04d"%(rank)+tempStr[len(tempStr)-2] + tempStr[len(tempStr)-1]
        if os.path.exists(self.ddst):
            os.system("rm -rf "+ self.ddst)
        os.mkdir(self.ddst)
        self.subimg1 = np.zeros((self.xWidth,self.yWidth),self.dataType)
        if self.data.is_projection == True:
            self.proimg = np.zeros((self.Img_height,self.Img_width),self.dataType)
        return True
    

    def do_montage(self, task, rank):
        if len(task) == 0:
            return
        errorMontage = 0
        dtarerror = 0     
        projecting = 0
        for z in range(len(task)):
            s = task[z]
            numStr = "%05d"%(s[0])
            self.xEnd = int(s[3])
            self.yEnd = int(s[4])
            transY = -self.data.ytransMax+s[1]
            transX = self.data.xtransMax-s[2]
            self.img = np.zeros((self.Img_height0,self.Img_width0),self.dataType)

            dfile_name = self.data.src + self._separator + self.dfile_pre + numStr + self.dfile_post
            if os.path.exists(dfile_name):
                try:
                    tar = tarfile.open(dfile_name)
                except IOError, e:
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
                    if i == self.yBeg:
                        subimg[0:131,:] = 0
                    if self.data.is_reReverse == 1:
                        subimg = subimg[:,::-1]
                        self.img[(i-self.yBeg)*self.xWidth:(i+1-self.yBeg)*self.xWidth,(j-self.xBeg)*self.yWidth:(j+1-self.xBeg)*self.yWidth] = subimg[0:self.xWidth,0:self.yWidth]
                    if self.data.is_reReverse == 0:
                        if self.data.process_direction == 1:
                            self.img[transY+(i-self.yBeg)*self.xWidth:transY+(i+1-self.yBeg)*self.xWidth,self.Img_width0-transX-(j+1-self.xBeg)*self.yWidth:self.Img_width0-transX-(j-self.xBeg)*self.yWidth] = subimg[0:self.xWidth,0:self.yWidth]
                        else:
                            if (j-40)%2 == 0:
                                self.img[transY+(i-self.yBeg)*self.xWidth:transY+(i+1-self.yBeg)*self.xWidth,self.Img_width0-transX-(j+1-self.xBeg)*self.yWidth:self.Img_width0-transX-(j-self.xBeg)*self.yWidth] = subimg[0:self.xWidth,0:self.yWidth]
                            else:
                                if i == 31:
                                    subimg = subimg[::-1,:]
                                    self.img[transY+(self.yEnd-i)*self.xWidth+128:transY+(self.yEnd+1-i)*self.xWidth,self.Img_width0-transX-(j+1-self.xBeg)*self.yWidth:self.Img_width0-transX-(j-self.xBeg)*self.yWidth] = subimg[0:self.xWidth-127,0:self.yWidth]
                                else:
                                    subimg = subimg[::-1,:]
                                    self.img[transY+(self.yEnd-i)*self.xWidth+128:transY+(self.yEnd+1-i)*self.xWidth+128,self.Img_width0-transX-(j+1-self.xBeg)*self.yWidth:self.Img_width0-transX-(j-self.xBeg)*self.yWidth] = subimg[0:self.xWidth,0:self.yWidth]
            
            if self.data.changeSize[0] == self.data.changeSize[1] == 1:
                image = self.img
            else:
                image = cv2.resize(self.img, (self.Img_width, self.Img_height))
            if not self.data.trs_about[0] == self.data.trs_about[0] == 0:
                image = self.do_translation(image)
            print self.writePre + numStr + self.writePost
            cv2.imwrite(self.montage_file + self._separator + self.writePre + numStr + self.writePost,image)
            

            if self.data.is_resampling == True and ((s[0] % self.z_ratio_rec) == 0):
                re_image = cv2.resize(image,(self.re_width,self.re_height))
                cv2.imwrite(self.resample_file + self._separator + self.rewritePre + numStr + self.rewirtePost, re_image)

            tddir = self.ddst + self._separator + numStr
            os.system("rm -rf "+tddir)
            
            if self.data.is_projection == True:
                if errorMontage == 0:
                    self.proimg = np.maximum(image,self.proimg)
                projecting = projecting + 1
                if projecting == self.data.thick_projection or z == len(task)-1:
                    pfilename = self.projection_file + self._separator + self.pwritePre + "%05d"%(s[0]-projecting+1) + "-" + "%05d"%(s[0]) + self.pwirtePost
                    pjpgname = self.projection_jpg + self._separator + self.pwritePre + "%05d"%(s[0]-projecting+1) + "-" + "%05d"%(s[0]) + ".jpg"
                    #print pfilename
                    cv2.imwrite(pfilename,self.proimg)
                    if self.dataType == np.uint16:
                        cv2.imwrite(pjpgname,np.uint8(self.proimg*(255.0/(2**12-1))))
                    self.proimg = np.zeros((self.Img_height,self.Img_width),self.dataType)
                    projecting = 0
            errorMontage = 0
            dtarerror = 0
            
    def do_translation(self, origImg):
        vertical = 0
        horizontal = 0
        vp = self.data.trs_about[0]
        if not vp == 0:
            vertical = 1
        if vp < 0:
            vertical = -1
            vp = -vp
        hp = self.data.trs_about[1]
        if not hp == 0:
            horizontal = 1
        if hp < 0 :
            horizontal = -1
            hp = -hp
        
        if vertical == horizontal == 0:
            return
        raw_width = self.Img_width
        raw_height = self.Img_height
        tmpImg = np.zeros((raw_height,raw_width),self.dataType)
            
        if horizontal == 1:
            tmpImg[:,0:raw_width-hp] = origImg[:,hp:raw_width]
        if horizontal == -1:
            tmpImg[:,hp:raw_width] = origImg[:,0:raw_width-hp]
        if horizontal == 0:
            tmpImg = origImg

        if vertical == 1:
            tmpImg[0:raw_height-vp,:] = tmpImg[vp:raw_height,:]
            tmpImg[raw_height-vp:raw_height,:] = 0
        if vertical == -1:
            tmpImg[vp:raw_height,:] = tmpImg[0:raw_height-vp,:]
            tmpImg[0:vp,:] = 0    
        return tmpImg



