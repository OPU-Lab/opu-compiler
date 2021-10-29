# opu-compiler

Frontend is developed based on TVM framework.

<img src="https://github.com/OPU-Lab/opu-compiler/blob/master/image/overview.png" width="1000" height="350" />

### Build Environment
```
cd docker
```

```
sudo docker build -f Dockerfile --tag opu-compiler:1.0 .
```
launch docker
```
sudo docker run --rm --pid=host\
                     --mount src="$(pwd)"/../..,target=/workspace,type=bind\
                     -w /workspace\
                     -e "CI_BUILD_HOME=/workspace"\
                     -e "CI_BUILD_USER=$(id -u -n)"\
                     -e "CI_BUILD_UID=$(id -u)"\
                     -e "CI_BUILD_GROUP=$(id -g -n)"\
                     -e "CI_BUILD_GID=$(id -g)"\
                     -h opu-compiler-docker\
                     --name opu-compiler-docker\
                     -it --net=host\
                     opu-compiler:1.0\
                     /bin/bash
```

### Build Compiler
```
cd opu-compiler
cd frontend;mkdir build;cmake ..;cd build;make -j4;cd ../..
cd data-layout-generator;mkdir build;cd build;cmake ..;make -j4;cd ../..
```

### Example
```
cd example/tiny_yolo
```
Download model exported from Tensorflow
```
sh download_model.sh
```
Frontend parse freezed model to OPU_IR.json (layer-wise parameters)
```
python3 frontend.py --input tiny_yolo_lp_detection.pb --input_shape 1 416 416 3
```
Check and run OPU_IR.json
```
python3 ../../util/run_ir_json.py --config OPU_IR.json --input input.npy --weight_dir dump_raw
```
Backend performs target-specific code transformations and generate straight line code for simulation.
```
../../backend/backend -i OPU_IR.json --codegen-non-isa
```
data-layout-gen generates weight and bias laytout in DRAM.
```
../../data-layout-generator/build/data-layout-gen dram-weight-layout.json dram-bias-layout.json dump
```
