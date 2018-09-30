import sys
import preprocess as pp
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpiSize = comm.Get_size()
name = MPI.Get_processor_name()
pre_process = pp.Preprocess('data.txt',mpiSize)
data_info = pre_process.data
terminate_flag = None
print data_info.taskType
print data_info.thick_projection
print data_info.begin_num
print data_info.dst

if data_info.taskType > 7 or data_info.taskType < 1:
    print "none type of task  is specified."    
    sys.exit()
    
if rank == 0:    
    if not pre_process.isExist_file():
        print "The target file does not exists."
        terminate_flag = True
  
terminate_flag = comm.bcast(terminate_flag,root=0)
if terminate_flag == True:
    print "the end: ",rank
    sys.exit()

task_set = pre_process.get_task_set(rank)
if len(task_set) == 0:
    sys.exit()

if data_info.taskType == 1:
    pre_process.do_projection(task_set)
if data_info.taskType == 2:
    pre_process.do_resample(task_set)
if data_info.taskType == 3:
    pre_process.do_crop(task_set)
if data_info.taskType == 4:
    pre_process.do_translation(task_set)

print "I'm done by rank", rank, name



    


