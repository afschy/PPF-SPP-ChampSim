git submodule update --init
vcpkg/bootstrap-vcpkh.sh
vcpkg/vcpkg install

./config.sh champsim_config.json
make
