#!/usr/bin/env python

# On the intranet page for the Cirrus cluster there is a simple demo app called
# Snowfall_cluster.r that generates 100000 numbers using the formula
# y <- x + rnorm(1).  It farms the task out to nodes on the cluster.

# I'm making a Python re-implementation.  First up, here is a regular non-MPI
# version to test that the actual number generation works and demonstrate NumPy

# This makes the code compatible with Python 2.6/2.7
from __future__ import division, print_function

import sys
import numpy as np
from time import localtime

# Make 10 million numbers, otherwise it's just too fast.
# This takes about 30 sec, but virtually all of that is spent outputting
# the data to the file.
nums_wanted = 10000000
batch_size = 10000

# If nums_wanted does not divide exactly by the batch size my algorithm won't work.
# I could address this but I'll keep it simple just now.
assert(nums_wanted % batch_size == 0)

# The old R example used the number 5 as the mean of the distribution.
# I'll be slightly more dynamic and use either the first command line arg or else the
# current day of the month (1 to 31).
dist_mean = 0
if(sys.argv[1:]):
    dist_mean = int(sys.argv[1])
else:
    dist_mean = localtime().tm_mday

# Using numpy-encapsulated values is good practise with MPI as they can be passed
# efficiently between worker nodes.  Here we make a 0-dimensional array, which is
# just an integer wrapped as a NumPy object.
dist_mean = np.array(dist_mean, 'float')

# Tell the user what we are up to...
print("Calculating rnorm distribution size %i about %f." % (nums_wanted, dist_mean))

# We'll spit out the numbers to a file
outfile = open("random_%i_nums.txt" % nums_wanted, 'w')

# Lets' pretend we can't have all 10000000 values in memory at once, so we are generating
# them in batches.

nums_generated = 0
while nums_generated < nums_wanted:
    # NumPy makes it very easy to generate a batch of things at once,
    # in this case a random sampling from a normal distribution about
    # a given mean.
    res = np.random.normal(loc=dist_mean, size=batch_size)

    # And I can print the partial result like this.
    # I'm not bothering to use np.nditer(res) because for this 1D array it's redundant.
    for i in res:
        print(i, file=outfile)

    nums_generated += batch_size

print("Done.  Closing file %s." % outfile.name)
outfile.close()
