#pragma once

#include <util/cuda_device.hpp>


void launch_saxpy(cuda::device& gpu, unsigned N, unsigned repetitions);