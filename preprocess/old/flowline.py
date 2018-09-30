import os
import preprocess as pp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpiSize = comm.Get_size()
name = MPI.Get_processor_name()
pre_process = pp.Preprocess('data.txt',mpiSize)
data_info = pre_process.data
terminate_flag = None

#print pre_process.data.is_build_file

if pre_process.data.is_build_file == True:
    if rank == 0:
    	if not pre_process.make_file():
            print "The target file does not exists."
            terminate_flag = True
terminate_flag = comm.bcast(terminate_flag,root=0)
if terminate_flag == True:
    print "the end: ",rank
    exit()

task_set = pre_process.get_task_set(rank)
if len(task_set) == 0:
    exit()
try:
    if not pre_process.set_parameter(task_set,rank):
        exit()
    pre_process.do_montage(task_set,rank)
finally:
    os.system("rm -rf "+ pre_process.ddst)

print "I'm done by rank", rank, name