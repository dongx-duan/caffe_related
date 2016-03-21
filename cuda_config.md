#### 1 download and install cuda driver   
download cuda driver from http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/   
```
dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb       
apt-get update      
apt-get install cuda      
```
#### 2  config cuda
```  
echo "/usr/local/cuda/lib64" | sudo tee /etc/ld.so.conf.d/cuda64.conf    
sudo ldconfig    
```
#### 3 install cudnn
unzip cudnn and copy head files and library to cuda
```
tar -zxvf cudnn-7.0-linux-x64-v3.0-prod.tgz    
sudo cp lib* /usr/local/cuda/lib64/    
sudo cp cudnn.h /usr/local/cuda/include/    
```
create soft link
```
cd /usr/local/cuda/lib64/    
sudo rm -rf libcudnn.so libcudnn.so.7.0    
sudo ln -s libcudnn.so.7.0.64 libcudnn.so.7.0    
sudo ln -s libcudnn.so.7.0 libcudnn.so    
sudo ldconfig -v | grep cudnn    
```

#### 4 check cuda
```
nvidia-smi
```
