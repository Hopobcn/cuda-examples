#pragma once

#include <util/cuda_vector.hpp>


void run_saxpy_c(const float* px,
                 const float* py,
                       float* pz,
                 unsigned N,
                 float alpha,
                 unsigned repetitions);

void run_saxpy_c(const double* px,
                 const double* py,
                       double* pz,
                 unsigned N,
                 double alpha,
                 unsigned repetitions);

void run_saxpy_cpp(const cuda::vector<float>& x,
                   const cuda::vector<float>& y,
                         cuda::vector<float>& z,
                   unsigned N,
                   float alpha,
                   unsigned repetitions);

void run_saxpy_cpp(const cuda::vector<double>& x,
                   const cuda::vector<double>& y,
                         cuda::vector<double>& z,
                   unsigned N,
                   double alpha,
                   unsigned repetitions);

void run_saxpy_cublas(const float* px,
                            float* py,
                      unsigned N,
                      const float alpha,
                      unsigned repetitions);


void run_saxpy_cublas(const double* px,
                      double* py,
                      unsigned N,
                      const double alpha,
                      unsigned repetitions);



void run_saxpy_cub(const float* px,
                   const float* py,
                   float* pz,
                   unsigned N,
                   float alpha,
                   unsigned repetitions);


void run_saxpy_cub(const double* px,
                   const double* py,
                   double* pz,
                   unsigned N,
                   double alpha,
                   unsigned repetitions);