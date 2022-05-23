# stixel
Stixel c++ code based on gish523's algo.

### Requirement
Opencv4.0.0

### Compilation
```bash
mkdir build && cd build
cmake ..
make
```

Make sure your are in the `build` folder to run the executables.

### Running
```bash

./stixel [dir] [camera param]

dir should be compatible with Kitti dataset.   
./stixel ../data/ ../camera.yml

../data/ 下应该有两个文件夹, 分别是image_2 和image_3, 分别代表双目视觉的左右两幅图片

