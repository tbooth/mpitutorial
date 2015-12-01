#!/usr/bin/env python

"""
 On the intranet page for the Cirrus cluster there is a simple demo app called
 Snowfall_cluster.r that generates 100000 numbers using the formula
 y <- x + rnorm(1).  It farms the task out to nodes on the cluster.  This is
 a Python reimplementation using mpi4py.

 Warning - raw MPI is hard and does not provide the convenience of things like
 sfClusterApplyLB.  To have any idea what is happening, read up on
  http://mpitutorial.com/beginner-mpi-tutorial/ first.

 This should work on Cirrus, but you may need to build mpi4py yourself.

 If this runs faster or slower than the R version, don't read anything into it. By
 far the largest amount of time in the R version is spent writing debugging info
 to the shared log file!
"""

# This makes the code compatible with Python 2.6/2.7 as well as 3.4
from __future__ import division, print_function

from mpi4py import MPI
import sys
# Docs for NumPy all use 'np' as the import name.  Do not confuse with -np arg to mpirun
# which is captured as 'size' below.
import numpy as np
import atexit
from time import time, localtime

# Fire up MPI and see which worker the script is running as.
comm = MPI.COMM_WORLD
size = comm.Get_size()  # How many workers??
rank = comm.Get_rank()  # Which one am I??

# At the moment, this won't work at all if you have only one thread.
assert (size > 1), "This won't do anything if there are no worker nodes.  Run with mpirun -np [>=2]."

# Make at least a million numbers, otherwise it's just too fast.
# Ten million takes about 30 sec, but virtually all of that is spent outputting
# the data to the file.
# Note that with the master/slave model we don't need to set these numbers in global scope;
# only the rank 0 process needs to know them.
# Also there is no need for nums_wanted to be an exact multiple of batch_size.
nums_wanted = 1000064
batch_size = 10000

# Function that will run on the main thread...
def main_thread():

    # The old R example used the number 5 as the mean of the distribution, but if setting
    # a constant there is no need to pass it around with MPI - it could just be a constant.
    # I'll be slightly more dynamic and use either the first command line arg or else the
    # current day of the month (1 to 31).
    dist_mean = 0
    if(sys.argv[1:]):
        dist_mean = float( sys.argv[1] )
    else:
        dist_mean = float( localtime().tm_mday )

    # Using numpy-encapsulated values is good practise with MPI as they can be passed
    # efficiently between worker nodes.
    # But do I need to do it for single values?
    #     dist_mean = numpy.array(dist_mean)
    # In this case it seems easier to pass out jobs as Python objects and fetch the results
    # as NumPy arrays.

    # We'll spit out the numbers to a file
    outfile = open("random_%i_nums.txt" % nums_wanted, 'w')
    print("Picking %d numbers from rnorm distribution about %i." % (nums_wanted, dist_mean))

    # We are using a set size pool of workers.  It is possible to start workers
    # dynamically with MPI2, but we won't do that here.

    # On the main thread, run a loop doling out tasks to the workers.
    # Because this give a new task to each worker as soon as it becomes
    # free it is similar to SFClusterApplyLB in Snowfall.
    workers_avail = list(range(1, size))
    nums_out = 0  #How many numbers have I asked to generate?
    nums_in = 0   #How many have I actually received?
    while nums_in < nums_wanted:
        #While there are free workers, dole out jobs.  Tell the worker what
        #the mean is and how many values it should generate.
        #If I ask a worker to generate 0 values it will exit.
        while workers_avail:
            count = min(batch_size, (nums_wanted - nums_out))

            # Farm out a block to the next avail worker
            comm.send( {'mean':dist_mean, 'count':count}, dest=workers_avail.pop() )
            # And add this count to the tally of numbers requested
            nums_out += count

        # When all workers are fed, wait to recv from any worker.  Here the return
        # vals are NumPy objects of max length batch_size.  Note that here I don't care about
        # the return order.  If I did, I'd need to have a buffer to keep any out-of-order results
        # in and things would get a bit more complicated.
        # I do need to pre-allocate status and buffer objects to capture the received data.
        recv_status = MPI.Status()
        recv_buf = np.empty(batch_size, 'float')
        comm.Recv(recv_buf, source=MPI.ANY_SOURCE, status=recv_status)

        #add the worker that replied back to the list
        workers_avail.append(recv_status.Get_source())
        #See how many values I got back.  I could set the type to MPI.DOUBLE
        #but since I'm getting mpi4py to infer it above I feel I should follow the same logic.
        recv_buf_size = recv_status.Get_count(MPI.__TypeDict__[recv_buf.dtype.char])
        #Trim the array if it is shorter than batch_size.
        recv_buf = recv_buf[:recv_buf_size]

        print("Worker %d sent me %d numbers." % (recv_status.Get_source(), len(recv_buf)) )

        #Save out the result
        # Note that in the R example the workers print the results, but doing it this
        # way will work even if the worker nodes don't have access to the shared FS.
        # And I'd still need to send an empty message back to the root process from
        # each worker to signal readiness for another batch even if I didn't receive
        # the resutls back.
        for i in recv_buf:
            print(i, file=outfile)

        #Add what we just got to the count of numbers received.
        nums_in += recv_buf_size

    print("Got %d numbers. Yay." % nums_in )

    #At this point the last worker we just got a reply from will be waiting for work.
    #In the case where (nums_wanter / batch_size) < size we could have many workers
    #waiting.  In any case, clean 'em up.
    for w in workers_avail:
        comm.send( {'mean':0.0, 'count':0}, dest=w )

    #Sync at this point to be sure all the workers really exited the loop.
    comm.Barrier()

    print("Done.  Closing file %s." % outfile.name)
    outfile.close()

# Function that will run on each worker thread...
def worker_thread():
    #Receive instructions from the root node until we run out of jobs.
    while True:
        #Very simple receive from the root.
        job = comm.recv()

        #A zero-length job tells us we're done.
        if not job['count']:
            break

        #Standard NumPy call
        data_to_send = np.random.normal(loc=job['mean'], size=job['count'])

        #Send with a capital 'S' for NumPy arrays.
        comm.Send(data_to_send, dest=0)

    #If I have a barrier in the main thread I need one here too.
    comm.Barrier()


if rank==0:
    try:
        main_thread()
    except:
        # On all main thread, tear down MPI.  This produces a big grumpy error
        # but does avoid hanging the process.
        atexit.register(MPI.COMM_WORLD.Abort, 1)
        raise
else:
    worker_thread()
