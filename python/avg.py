#!/usr/bin/python
"""
Author: Wes Kendall + Tim Booth
Copyright 2012 www.mpitutorial.com
This code is a Python translation of one of the tutorials on mpitutorial.com. Feel
free to modify it for your own use. Any distribution of the code must
either provide a link to www.mpitutorial.com or keep this header in tact.

Program that computes the average of an array of elements in parallel using
MPI_Scatter and MPI_Gather.

The original function names have been kept, though in fact most of the stats calcuations
can be deferred to NumPy so the functions create_rand_nums and compute_avg are minimal.

Run with, eg.:
$ mpirun -n 4 python avg.py 100

Expected result is about 0.5.  The two results should not mismatch at all because the
underlying precision is greater than what is printed out.
"""

"""
Useful links:
    http://mpi4py.scipy.org/docs/usrman/tutorial.html
    http://mpi4py.scipy.org/docs/apiref/mpi4py.MPI.Comm-class.html
    http://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node34.html
    http://stackoverflow.com/questions/21088420/mpi4py-send-recv-with-tag
"""

# This makes the code compatible with Python 2.6/2.7 as well as 3.4
from __future__ import division, print_function

from sys import stdout, stderr, argv, exit
import numpy as np
import atexit
from time import time
from mpi4py import MPI

# Creates an array of random numbers. Each number has a value from 0 - 1
def create_rand_nums(num_elements):
    # In the C version this used malloc(...) and a loop to populate the array,
    # but we will call out directly to NumPy.  Very simple.
    # Note than this function only ever gives out floats.
    return np.random.random(num_elements)


# Computes the average of an array of numbers
def compute_avg(array, num_elements=None):
    # In the C version we had to specify the array length explicitly.
    # In Python we know how long our lists are.
    # Again, use NumPy to do the calculation.  Again very simple.
    return array.mean()


def main():
    # argv is imported above so does not need to be passed in to main()
    if (len(argv) != 2):
        print("Usage: avg num_elements_per_proc", file=stderr)
        exit(1)

    num_elements_per_proc = int(argv[1])
    # Seed the random number generator.
    # Numpy does this for us, but we can still seed from time() if we want.
    #  np.random.seed([time()])

    # Fire up MPI and see what's what.
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()  # How many workers??
    world_rank = comm.Get_rank()  # Which one am I??

    # Create a random array of elements on the root process. Its total
    # size will be the number of elements per process times the number
    # of processes
    rand_nums = None
    if (world_rank == 0):
        rand_nums = create_rand_nums(num_elements_per_proc * world_size);

    # Scatter the random numbers from the root process to all processes in
    # the MPI world
    # In the C version we have to malloc() the buffer in each worker first, and
    # with NumPy we have to do the equivalent using empty()
    # Hopever, we can omit most of the parameters to Scatter() because Python can
    # work them out.
    # Note use of 'Scatter' not 'scatter' as we're using NumPy arrays.
    sub_rand_nums = np.empty(num_elements_per_proc, 'float')
    comm.Scatter(rand_nums, sub_rand_nums)

    """ The C code here was:
    float *sub_rand_nums = (float *)malloc(sizeof(float) * num_elements_per_proc);
    assert(sub_rand_nums != NULL);
    MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT, sub_rand_nums,
              num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
    """

    # Now all workers, including node 0, have a subset of the numbers to work on.
    # Compute the average of our subset
    sub_avg = compute_avg(sub_rand_nums)

    # Gather all partial averages down to the root process.  Again I need to
    # pre-allocate a NumPy array to hold the results.
    sub_avgs = None
    if (world_rank == 0):
        sub_avgs = np.empty(world_size, 'float')

    comm.Gather(sub_avg, sub_avgs)

    """ In the C code we had to malloc() sub_avgs and then do:
    MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    """

    # Now that we have all of the partial averages gethered on the root, compute the
    # total average of all numbers. Since we are assuming each process computed
    # an average across an equal amount of elements, a mean of means will
    # produce the correct answer.
    if (world_rank == 0):
        print("The %d workers produced these averages:\n %s" % (world_size, repr(sub_avgs)))

        avg = compute_avg(sub_avgs);
        print("Avg of all elements is therefore %f" % avg)

        # Compute the average across the original data for comparison
        original_data_avg = compute_avg(rand_nums, num_elements_per_proc * world_size)
        print("Avg computed across original data is %f" % original_data_avg)

    # Clean up
    # Python frees memory and shuts down MPI for us, but we can still do a sync
    # if we like.
    comm.Barrier()

try:
    main()
except:
    # On all errors, tear down MPI.  This produces a big grumpy error
    # but does avoid hanging the process.
    atexit.register(MPI.COMM_WORLD.Abort)
    raise
