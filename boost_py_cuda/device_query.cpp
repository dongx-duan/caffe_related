#include <cstdio>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <boost/python.hpp>

void device_query()
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess)
  {
    LOG(FATAL) << "cudaGetDeviceCount FAILED! " << (int)error_id;
    LOG(FATAL) << cudaGetErrorString(error_id);
  }
  if (deviceCount == 0)
  {
    LOG(FATAL) << "No Cuda GPU!";
  }
  else
  {
    LOG(INFO) << "Total GPU(s): " <<  deviceCount;
    int driverVersion, runtimeVersion;
    for (int i=0; i<deviceCount; i++)
    {
      cudaSetDevice(i);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, i);
      cudaDriverGetVersion(&driverVersion);
      cudaRuntimeGetVersion(&runtimeVersion);
      
      LOG(INFO) << "";
      LOG(INFO) << "Device id:                     " << i;
      LOG(INFO) << "CUDA Driver Version            " << driverVersion/1000 << "." << (driverVersion%100)/10;
      LOG(INFO) << "CUDA Runtime Version           " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10;
      LOG(INFO) << "Major revision number:         " << deviceProp.major;
      LOG(INFO) << "Minor revision number:         " << deviceProp.minor;
      LOG(INFO) << "Device name:                   " << deviceProp.name;
      LOG(INFO) << "Total global memory:           " << deviceProp.totalGlobalMem;
    }
  }
}

BOOST_PYTHON_MODULE(pycuda)
{
    using namespace boost::python;
    def("device_query", &device_query);
}