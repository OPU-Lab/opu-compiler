# opu-compiler
```
cd docker
```

```
sudo docker build -f Dockerfile --tag opu-compiler:1.0 .
```

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

Build
```
cd frontend;mkdir build;cd build;make -j4;cd ../..
cd data-layout-generator;mkdir build;cd build;make -j4;cd ../..
```

Example
```
cd example/tiny_yolo
sh download_model.sh
python3 frontend.py --input tiny_yolo_lp_detection.pb --input_shape 1 416 416 3
python3 ../../util/run_ir_json.py --config OPU_IR.json --input input.npy --weight_dir dump_raw
../../backend/backend -i OPU_IR.json --codegen-non-isa
../../data-layout-generator/build/data-layout-gen dram-weight-layout.json dram-bias-layout.json dump
```
