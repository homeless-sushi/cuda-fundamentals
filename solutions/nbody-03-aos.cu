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
// The position integration is also computed by the GPU 
// and to avoid the need to synchronize between 2 kernels
// we must write on a different location.
// The output becomes the input for the next iteration
__global__
void bodyForce(Body_soa p_in, Body_soa p_out, float dt, int n) {
  const int absoluteThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for(int i = absoluteThreadIdx; i < n; i+=stride) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    //read thread's old_body data
    const float old_x = p_in.x[i];
    const float old_y = p_in.y[i];
    const float old_z = p_in.z[i];
    const float old_vx = p_in.vx[i];
    const float old_vy = p_in.vy[i];
    const float old_vz = p_in.vz[i];

    for(int j = 0; j < n; ++j){
        float dx = p_in.x[j] - old_x;
        float dy = p_in.y[j] - old_y;
        float dz = p_in.z[j] - old_z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    //compute new speed and new postion
    const float curr_vx = old_vx + dt*Fx;
    const float curr_vy = old_vy + dt*Fy;
    const float curr_vz = old_vz + dt*Fz;
    p_out.vx[i] = curr_vx;
    p_out.vy[i] = curr_vy;
    p_out.vz[i] = curr_vz;
    p_out.x[i] = old_x + curr_vx*dt; 
    p_out.y[i] = old_y + curr_vy*dt; 
    p_out.z[i] = old_z + curr_vz*dt;
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
    bodyForce<<<blockDim_x, gridDim_x>>>(*curr_p_in_d, *curr_p_out_d, dt, nBodies);
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
