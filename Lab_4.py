from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

size = comm.Get_size()

def floyd(mat):
# Number of rows per thread
    num_rows_per_threads = len(mat) / size
# Number of threads per row
    threads_per_row = size / len(mat)
# Where to start for that thread`
    start = int(num_rows_per_threads * rank)
# Where to end for that thread
    end = int(num_rows_per_threads * (rank + 1))

    for k in range(len(mat)):
        row_owner = (threads_per_row * k)
        mat[k] = comm.bcast(mat[k], root=row_owner)
        for x in range(start, end):
            for y in range(len(mat)):
                mat[x][y] = min(mat[x][y], mat[x][k] + mat[k][y])
    if rank == 0:
        for k in range(end, len(mat)):
            row_owner = (threads_per_row * k)
            mat[k] = comm.recv(source=row_owner, tag=k)
    else:
        for k in range(start, end):
            comm.send(mat[k], dest=0, tag=k)
    return mat

def read_file(doc):
    f_doc = open(doc, 'r')
    lines_doc = f_doc.readlines()
    for i in range(len(lines_doc)):
        lines_doc[i] = lines_doc[i].split()
    return lines_doc

def main():
    matrix_doc = read_file("fwTest.txt")
    start = time.time()
    floyd(matrix_doc)
    print(time.time() - start)

main()
