all : device_query.cpp
	nvcc --compiler-options '-fPIC' -o pycuda.so -shared device_query.cpp -lpython2.7 -lboost_python -lglog

clean :
	rm pycuda.so