## Installation

### Requirements

- Linux
- Python 3.6+
- PyTorch 
- CUDA 
- **CMake 3.13.2 or higher**
- spconv 1.2.1
- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)




### Install Requirements

#### spconv

```bash
 $ sudo apt-get install libboost-all-dev
 $ git clone https://github.com/poodarchu/spconv --recursive
 $ cd spconv && python setup.py bdist_wheel
 $ cd ./dist && pip install *
```

#### nuscenes-devkit

```bash
pip install nuscenes-devkit
```

### Install Det3D

The installation should be under the gpu environment.

#### Install Det3D

```bash
$ python setup.py build develop
```

### Common Installation Issues

#### ModuleNotFoundError: No module named 'det3d.ops.nms.nms' when installing det3d

Run `python setup.py build develop` again.

#### "values of 'package_data' dict" must be a list of strings (got '*.json') when installing nuscenes-devikit

Use `setuptools 39.1.0 `

#### cannot import name PILLOW_VERSION
`pip install Pillow==6.1`

#### Installing a suitable pytorch version by replacing the previous version
`pip install torch==1.3.0 torchvision==0.4.1`

#### Upgrading cmake in case if needed
`sudo apt remove cmake`

`pip install cmake --upgrade`

#### Installing suitable setuptools version by replacing the previous version
`pip install setuptools==39.1.0`

