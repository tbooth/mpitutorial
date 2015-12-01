The file avg.py contains a Python script that makes use of the mpi4py
libraries scatter/gather to split a job across multiple nodes.
In this case the job is the rather pointless task of finding the average
of a bunch of random numbers, but it's a good way to test that the library
is working.  If for some reason you need to build MPI4Py yourself here
are some notes.

1) Activate the GCC compiler for MPI.  The ICC compiler will not produce
   a working version of the module (at least not on my system).
   Note that these commands only apply to systems using the "modules" system
   which is not most Linux systems.

$ module rm openmpi/intel
$ module load openmpi/gcc

2) Download and unpack the mpi4pi source

$ wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-1.3.1.tar.gz
$ tar -xvaf mpi4py-1.3.1.tar.gz

3) Build it

$ ( cd mpi4py-1.3.1 ; python setup.py build )

4) If that worked, move the result to your personal library.

$ mkdir -p ~/python_libs
$ mv mpi4py-1.3.1/build/lib.linux-*/* ~/python_libs

5) Ensure the module can be seen by Python.  You must put this in your
   .bashrc, as well as running it, so that the module can be seen by worker
   processes on every node.

$ export PYTHONPATH="$HOME/python_libs"

6) Now try out avg.py.  As it runs very fast we can bypass the scheduler and
   run it right off the head node for initial testing.

$ mpirun -np 10 python avg.py 100

or even (do not do this with long-running code!!!)...

$ mynodes=`echo node0{20,20,19,19,18,18,17,17} | tr ' ' ,`
$ mpirun -H $mynodes python avg.py 100

You can now remove the mpi4py source directory and the .tar.gz file to keep things
tidy.  To find out more about MPI, look at http://mpitutorial.com.  You can
also find the docs for mpi4py at http://mpi4py.scipy.org but be warned they
are 'very' uninformative for new users and assume that you are already
familiar with both MPI and NumPy before you start, and that you can make some
educated guesses about the mpi4py routines based on the MPI C documentation.

You might also want to look at the mpi_random_apply.py and mpi_random_gather.py
scripts that demonstrate two distinct strategies for splitting a large job
across multiple MPI workers.  In this case it's the old chestnut of generating
a normal distribution.

Happy hacking!

Tim Booth, Sept 2015.
