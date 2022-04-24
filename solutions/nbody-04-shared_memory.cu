#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

typedef struct { float *x, *y, *z, *vx, *vy, *vz; } Body_soa;

__global__
void aos2soa(Body* p_aos, Body_soa p_soa, int n){
  const int absoluteThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for(int i = absoluteThreadIdx; i < n; i+=stride){
    p_soa.vx[i] = p_aos[i].vx;
    p_soa.vy[i] = p_aos[i].vy;
    p_soa.vz[i] = p_aos[i].vz;
    p_soa.x[i] = p_aos[i].x;
    p_soa.y[i] = p_aos[i].y;
    p_soa.z[i] = p_aos[i].z;
  }
}

__global__
void soa2aos(Body* p_aos, Body_soa p_soa, int n){
  const int absoluteThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for(int i = absoluteThreadIdx; i < n; i+=stride) {
    p_aos[i].vx = p_soa.vx[i];
    p_aos[i].vy = p_soa.vy[i];
    p_aos[i].vz = p_soa.vz[i];
    p_aos[i].x = p_soa.x[i];
    p_aos[i].y = p_soa.y[i];
    p_aos[i].z = p_soa.z[i];
  }
}

// A simple parallelization of the outer for block.
// We use a Structure of Arrays instead of 
// an Arrays of Structures.
// Each thread in a block caches a data point in shared 
// memory; they then iterate through a block of data points
// starting from an offset and wrapping back around as to 
// avoid conflicts in shared memory requests. 
// The position integration is also computed by the GPU 
// and to avoid the need to synchronize between 2 kernels
// we must write on a different location.
// The output becomes the input for the next iteration
__global__
void bodyForce(Body_soa p_in, Body_soa p_out, float dt, int n) {
  const int absoluteThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  extern __shared__ float curr_block_xyz[];
  float* curr_block_x = curr_block_xyz;
  float* curr_block_y = curr_block_xyz + blockDim.x;
  float* curr_block_z = curr_block_xyz + 2 * blockDim.x;

  for(int i = absoluteThreadIdx; i < n; i+=stride) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    //read thread's old_body data
    const float old_x = p_in.x[i];
    const float old_y = p_in.y[i];
    const float old_z = p_in.z[i];
    const float old_vx = p_in.vx[i];
    const float old_vy = p_in.vy[i];
    const float old_vz = p_in.vz[i];

    int n_blocks = ceilf(n / blockDim.x);
    for(int curr_block = 0; curr_block < n_blocks; ++curr_block){

      const int thread_offset = curr_block * blockDim.x + threadIdx.x;
      if(thread_offset >= n)
        break;

      __syncthreads();
      curr_block_x[threadIdx.x] = p_in.x[thread_offset];
      curr_block_y[threadIdx.x] = p_in.y[thread_offset];
      curr_block_z[threadIdx.x] = p_in.z[thread_offset];
      __syncthreads();

      for(int j = 0; j < blockDim.x; ++j){

        int body_idx 
          = threadIdx.x + j < blockDim.x
          ? threadIdx.x + j
          : threadIdx.x + j - blockDim.x;

        if(curr_block * blockDim.x + body_idx > n)
          continue;

        float const dx = curr_block_x[body_idx] - old_x;
        float const dy = curr_block_y[body_idx] - old_y;
        float const dz = curr_block_z[body_idx] - old_z;
        float const distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float const invDist = rsqrtf(distSqr);
        float const invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }
    }

    //compute new speed and new postion
    const float new_vx = old_vx + dt*Fx;
    const float new_vy = old_vy + dt*Fy;
    const float new_vz = old_vz + dt*Fz;
    p_out.vx[i] = new_vx;
    p_out.vy[i] = new_vy;
    p_out.vz[i] = new_vz;
    p_out.x[i] = old_x + new_vx*dt; 
    p_out.y[i] = old_y + new_vy*dt; 
    p_out.z[i] = old_z + new_vz*dt;
  }
}


int main(const int argc, const char** argv) {

  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you generate ./nbody report files
  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  // The assessment will pass hidden initialized values to check for correctness.
  // You should not make changes to these files, or else the assessment will not work.
  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;
  buf = (float *)malloc(bytes);
  Body *p = (Body*)buf;
  read_values_from_file(initialized_values, buf, bytes);

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  const unsigned int blockDim_x = 64;
  const unsigned int max_gridDim_x = (nBodies + blockDim_x - 1) / blockDim_x;
  const unsigned int gridDim_x = min(deviceProp.multiProcessorCount*4, max_gridDim_x);

  Body* p_d;
  cudaMalloc(&p_d, bytes);
  cudaMemcpy(p_d, p, bytes, cudaMemcpyHostToDevice);

  Body_soa p_soa_in_d;
  cudaMalloc(&(p_soa_in_d.x), bytes/6);
  cudaMalloc(&(p_soa_in_d.y), bytes/6);
  cudaMalloc(&(p_soa_in_d.z), bytes/6);
  cudaMalloc(&(p_soa_in_d.vx), bytes/6);
  cudaMalloc(&(p_soa_in_d.vy), bytes/6);
  cudaMalloc(&(p_soa_in_d.vz), bytes/6);
  Body_soa p_soa_out_d;
  cudaMalloc(&(p_soa_out_d.x), bytes/6);
  cudaMalloc(&(p_soa_out_d.y), bytes/6);
  cudaMalloc(&(p_soa_out_d.z), bytes/6);
  cudaMalloc(&(p_soa_out_d.vx), bytes/6);
  cudaMalloc(&(p_soa_out_d.vy), bytes/6);
  cudaMalloc(&(p_soa_out_d.vz), bytes/6);

  
  aos2soa<<<blockDim_x, gridDim_x>>>(p_d, p_soa_in_d, nBodies);

  Body_soa* curr_p_in_d = &p_soa_in_d;
  Body_soa* curr_p_out_d = &p_soa_out_d;
  for (int iter = 0; iter < nIters; iter++) {
    bodyForce<<<blockDim_x, gridDim_x, blockDim_x*sizeof(float)*3>>>(*curr_p_in_d, *curr_p_out_d, dt, nBodies);
    Body_soa* tmp = curr_p_in_d;
    curr_p_in_d = curr_p_out_d;
    curr_p_out_d = tmp;
  }

  soa2aos<<<blockDim_x, gridDim_x>>>(p_d, *curr_p_in_d, nBodies);
  cudaMemcpy(p, p_d, bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  write_values_to_file(solution_values, buf, bytes);

  free(buf);
}
