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
