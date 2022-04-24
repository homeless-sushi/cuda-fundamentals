#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "files.h"

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

// A simple parallelization of the outer for block.
// We create a thread for every data point
__global__
void bodyForce(Body *p, float dt, int n) {
  unsigned int absoluteThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
  
  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[absoluteThreadIndex].x;
      float dy = p[j].y - p[absoluteThreadIndex].y;
      float dz = p[j].z - p[absoluteThreadIndex].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[absoluteThreadIndex].vx += dt*Fx; 
    p[absoluteThreadIndex].vy += dt*Fy; 
    p[absoluteThreadIndex].vz += dt*Fz;
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
  int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;
  buf = (float *)malloc(bytes);
  Body *p = (Body*)buf;

  Body* p_d;
  cudaMalloc(&p_d, bytes);

  read_values_from_file(initialized_values, buf, bytes);

  const unsigned int blockDim_x = 64;
  const unsigned int max_blocks = (nBodies + blockDim_x - 1) / blockDim_x;
  
  for (int iter = 0; iter < nIters; iter++) {

    cudaMemcpy(p_d, p, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<max_blocks, blockDim_x>>>(p_d, dt, nBodies); 
    cudaMemcpy(p, p_d, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    for (int i = 0 ; i < nBodies; i++) { 
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
  }

  write_values_to_file(solution_values, buf, bytes);

  free(buf);
  cudaDeviceReset();
}
