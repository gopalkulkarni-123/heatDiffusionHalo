#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstring>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Build a 2D Cartesian grid
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);  // Px, Py
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    if (cart_comm == MPI_COMM_NULL) {
        if (world_rank == 0) std::cerr << "Failed to create Cartesian communicator\n";
        MPI_Finalize();
        return 1;
    }

    int cart_rank = 0;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    int up = MPI_PROC_NULL, down = MPI_PROC_NULL, left = MPI_PROC_NULL, right = MPI_PROC_NULL;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);     // vertical
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);  // horizontal

    // Local interior size
    const int nx = 8;
    const int ny = 8;
    const int Nx = nx + 2; // with halos
    const int Ny = ny + 2;

    // Local grids (stack arrays)
    double grid[Nx][Ny];
    double newGrid[Nx][Ny];

    // Init to zero
    for (int i = 0; i < Nx; i++)
        for (int j = 0; j < Ny; j++)
            grid[i][j] = 0.0;

    // Apply physical boundaries once before iterations
    if (up == MPI_PROC_NULL)    for (int j = 0; j < Ny; ++j) grid[0][j]      = 0.0;
    if (down == MPI_PROC_NULL)  for (int j = 0; j < Ny; ++j) grid[Nx - 1][j] = 0.0;
    if (left == MPI_PROC_NULL)  for (int i = 0; i < Nx; ++i) grid[i][0]      = 100.0;
    if (right == MPI_PROC_NULL) for (int i = 0; i < Nx; ++i) grid[i][Ny - 1] = 100.0;

    // Iterations
    const int iters = 50;

    // Buffers for halo exchange
    double sendUp[ny], sendDown[ny], sendLeft[nx], sendRight[nx];
    double recvUp[ny], recvDown[ny], recvLeft[nx], recvRight[nx];

    // start time here
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int iter = 0; iter < iters; ++iter) {
        // Pack send buffers from interior edges
        for (int j = 0; j < ny; ++j) {
            sendUp[j] = grid[1][j + 1];
            sendDown[j] = grid[nx][j + 1];
        }
        for (int i = 0; i < nx; ++i) {
            sendLeft[i] = grid[i + 1][1];
            sendRight[i] = grid[i + 1][ny];
        }

        // Nonblocking communication
        MPI_Request requests[8];
        int idxReq = 0;

        if (up != MPI_PROC_NULL)
            MPI_Irecv(recvUp, ny, MPI_DOUBLE, up, 0, cart_comm, &requests[idxReq++]);
        if (down != MPI_PROC_NULL)
            MPI_Irecv(recvDown, ny, MPI_DOUBLE, down, 1, cart_comm, &requests[idxReq++]);
        if (left != MPI_PROC_NULL)
            MPI_Irecv(recvLeft, nx, MPI_DOUBLE, left, 2, cart_comm, &requests[idxReq++]);
        if (right != MPI_PROC_NULL)
            MPI_Irecv(recvRight, nx, MPI_DOUBLE, right, 3, cart_comm, &requests[idxReq++]);

        if (up != MPI_PROC_NULL)
            MPI_Isend(sendUp, ny, MPI_DOUBLE, up, 1, cart_comm, &requests[idxReq++]);
        if (down != MPI_PROC_NULL)
            MPI_Isend(sendDown, ny, MPI_DOUBLE, down, 0, cart_comm, &requests[idxReq++]);
        if (left != MPI_PROC_NULL)
            MPI_Isend(sendLeft, nx, MPI_DOUBLE, left, 3, cart_comm, &requests[idxReq++]);
        if (right != MPI_PROC_NULL)
            MPI_Isend(sendRight, nx, MPI_DOUBLE, right, 2, cart_comm, &requests[idxReq++]);

        if (idxReq > 0) MPI_Waitall(idxReq, requests, MPI_STATUSES_IGNORE);

        // Unpack halos
        if (up != MPI_PROC_NULL)    for (int j = 0; j < ny; ++j) grid[0][j + 1]      = recvUp[j];
        if (down != MPI_PROC_NULL)  for (int j = 0; j < ny; ++j) grid[Nx - 1][j + 1] = recvDown[j];
        if (left != MPI_PROC_NULL)  for (int i = 0; i < nx; ++i) grid[i + 1][0]      = recvLeft[i];
        if (right != MPI_PROC_NULL) for (int i = 0; i < nx; ++i) grid[i + 1][Ny - 1] = recvRight[i];

        // Jacobi update
        for (int i = 1; i <= nx; ++i) {
            for (int j = 1; j <= ny; ++j) {
                newGrid[i][j] = 0.25 * (
                    grid[i - 1][j] + grid[i + 1][j] +
                    grid[i][j - 1] + grid[i][j + 1]
                );
            }
        }

        // Copy back interiors
        for (int i = 1; i <= nx; ++i)
            for (int j = 1; j <= ny; ++j)
                grid[i][j] = newGrid[i][j];

        // Re-apply physical boundaries so boundary halos remain correct
        if (up == MPI_PROC_NULL)    for (int j = 0; j < Ny; ++j) grid[0][j]      = 0.0;
        if (down == MPI_PROC_NULL)  for (int j = 0; j < Ny; ++j) grid[Nx - 1][j] = 0.0;
        if (left == MPI_PROC_NULL)  for (int i = 0; i < Nx; ++i) grid[i][0]      = 100.0;
        if (right == MPI_PROC_NULL) for (int i = 0; i < Nx; ++i) grid[i][Ny - 1] = 100.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Gather local results to rank 0, including halos
    const int localCount = Nx * Ny;
    double localData[localCount];
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            localData[i * Ny + j] = grid[i][j];

    double* globalData = nullptr;
    if (world_rank == 0)
        globalData = new double[localCount * world_size];

    MPI_Gather(localData, localCount, MPI_DOUBLE,
               globalData, localCount, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        int Px = dims[0], Py = dims[1];
        int globalNx = Px * Nx;
        int globalNy = Py * Ny;

        double* assembled = new double[globalNx * globalNy];
        // initialize just in case
        for (int i = 0; i < globalNx * globalNy; ++i) assembled[i] = 0.0;

        for (int r = 0; r < world_size; ++r) {
            int c[2];
            MPI_Cart_coords(cart_comm, r, 2, c);
            int x0 = c[0] * Nx;
            int y0 = c[1] * Ny;
            for (int i = 0; i < Nx; ++i)
                for (int j = 0; j < Ny; ++j)
                    assembled[(x0 + i) * globalNy + (y0 + j)] =
                        globalData[r * localCount + i * Ny + j];
        }

        std::ofstream out("./newSolution.csv");
        out << "x,y,temperature\n";
        for (int i = 0; i < globalNx; ++i)
            for (int j = 0; j < globalNy; ++j)
                out << i << "," << j << "," << assembled[i * globalNy + j] << "\n";
        out.close();

        std::cout << "Global solution written to newSolution.csv ("
                  << globalNx << "x" << globalNy << ")\n";

        delete[] globalData;
        delete[] assembled;
    }

    std::cout << "Rank " << world_rank << " total time was = " << (t1 - t0) << " s\n";

    MPI_Finalize();
    return 0;
}
