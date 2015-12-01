#!/usr/bin/env python

"""
 On the intranet page for the Cirrus cluster there is a simple demo app called
 Snowfall_cluster.r that generates 100000 numbers using the formula
 y <- x + rnorm(1).  It farms the task out to nodes on the cluster.  This is
 a similar thing in Python using mpi4py.

 This version tackles the problem by doing an equal split across all worker threads.
 This is conceptually simple but notice the number of 'if rank==0' switches that
 I need to scatter into my code which starts to get messy.  Also we have variables
 like outfile that exist in all threads but are only ever used in the root thread.
 And we need to have everything in memory at once, which might be an issue.
 See also mpi_random_apply.py which is closer to the Snowfall "apply" paradigm.

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

# This can run with 1 thread, but it's a bit silly.
if not (size > 1):
    print("To actually make use of MPI, run this with mpirun -np [>=2].")

# Make at least a million numbers, otherwise it's just too fast.
# Ten million takes about 30 sec, but virtually all of that is spent outputting
# the data to the file.
nums_wanted = 1000064

def main():
    # With the scatter/gather model we need to ensure that nums_wanted is an exact multiple
    # of size.  Or else it complicates things.
    # Here we correct the value on all the nodes but only print a message from the root.
    nums_to_make = nums_wanted
    if(nums_wanted % size):
        if rank==0:
            print("The output size %d does not divide exactly by %d.  Increasing it to %d."
                    % (nums_wanted, size, nums_wanted + size - nums_wanted % size) )
        nums_to_make += size - nums_wanted % size

    # The old R example used the number 5 as the mean of the distribution, but if setting
    # a constant there is no need to pass it around with MPI - it could just be a constant.
    # I'll be slightly more dynamic and use either the first command line arg or else the
    # current day of the month (1 to 31).
    dist_params = {}
    if rank==0:
        if(sys.argv[1:]):
            dist_params['mean'] = float( sys.argv[1] )
        else:
            dist_params['mean'] = float( localtime().tm_mday )

    # Let's assume the value I just got for dist_mean was only available on the root node.
    # I can broadcast it out very simply.
    dist_params = comm.bcast(dist_params)
    assert 'mean' in dist_params

    # We'll spit out the numbers to a file.
    # Again, only the root node will open the file.
    outfile = None
    if rank==0:
        outfile = open("random_%i_nums.txt" % nums_wanted, 'w')
        print("Picking %d numbers from rnorm distribution about %i." % (nums_to_make, dist_params['mean']))

    # First step in a Scatter/Gather is normally to scatter the job across threads.
    # But here there is nothing to scatter.
    # I guess instead of broadcasting the mean I could could scatter an array of means
    # but it's silly.
    if 'mode' == 'silly':
        all_numbers = np.zeros(nums_to_make, 'float') + dist_params['mean']
        sub_numbers = np.empty(nums_to_make // size, 'float')
        comm.Scatter(all_numbers, sub_numbers)

    #Since all threads now know the mean, just generate a bunch of numbers like so.
    sub_numbers = np.random.normal(loc=dist_params['mean'], size=nums_to_make // size)

    #And gather the results in a big array, which I do need to allocate first but only
    #in the root thread.
    all_numbers = None
    if rank==0:
        all_numbers = np.empty(nums_to_make, 'float')

    comm.Gather(sub_numbers, all_numbers)

    # On the main thread, print out all the items in the big array.
    if rank==0:
        print("Got %d numbers. Outputting %d of them." % (len(all_numbers), nums_wanted) )
        for i in all_numbers[:nums_wanted]:
            print(i, file=outfile)

    #Sync at this point to be sure all the workers are in the same place.
    comm.Barrier()

    if rank==0:
        print("Done.  Closing file %s." % outfile.name)
        outfile.close()

try:
    main()
except:
    # On all main thread errors, tear down MPI.  This produces a big grumpy error
    # but does avoid hanging the process.
    # Unfortunately with Python3 my stack traces don't seem to be appearing ?!
    if rank==0:
        atexit.register(MPI.COMM_WORLD.Abort, 1)
    raise
