"""
    mpi.py : Parallelization routines and utilities

    This file is a part of fedrl.
"""

from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
version = ".".join(map(str, MPI.get_vendor()[1]))
vendor = MPI.get_vendor()[0]

if size < 2:
    if rank == 0:
        print('ERROR: must be run with at least 2 MPI processes')

    sys.exit(-1)

if rank == 0:
    print(f'MPI initialized with comm size {size}, backend {vendor} {version}')
