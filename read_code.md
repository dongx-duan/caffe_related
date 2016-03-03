start with /tools folder for command interfaces   
multi gpu caffe function:   
```cpp
caffe::P2PSync<float> sync(solver, NULL, solver->param());
sync.run(gpus);
```
