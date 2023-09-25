# H-Signature

A C++ or Python Library for computing the h-signature as defined by Bhattacharya, Likhachev, and Kumar in their [RSS Paper](https://www.roboticsproceedings.org/rss07/p02.pdf).

What is the H-signature? In the simplest case, it tells you whether two closed 3D curves are linked, or unlinked. The figure below shows some examples.
![Four visual examples of the H-signature](docs/simple_h.png)

In the general case, the H-signature is computed between one closed curve `$\tau$` and a skeleton of closed curves `$S=\{S_1,\dots,S_m\}$`. In code, each curve is a matrix of [3xN] (C++) or [Nx3] (Python) where N is the number of points in the curve. The H-signature is a vector (ordered list) of integers, where each element correspond to the h-signature of `$\tau$` with respect to one curve in `$S$`.

## Installation

For python:

```shell
pip install h-signature

```

For C++:

```shell
# The only option we offer is building from source with cmake 
git clone git@github.com:UM-ARM-Lab/h-signature.git
cd h-signature
mkdir build
# The only dependency is Eigen3
sudo apt install libeigen3-dev
# Build and run tests
cd build
cmake ..
make
ctest
```

## API

There is only one function!

```python
# Python with numpy
from h_signature.h_signature_np import get_h_signature
# Tau is an [Nx3] numpy array and skeleton is a dict of [3xN] numpy arrays
h_sig = get_h_signature(tau, skeleton)
print(h_sig)
```

```c++
// C++ with Eigen
#include <h_signature/h_signature.hpp>
// The above header has these typedefs
// typedef Eigen::Matrix3Xd Loop;
// typedef std::map<std::string, Eigen::Matrix3Xd> Skeleton;

// And this is the function, that's it! 
HSignature get_h_signature(Loop const &loop, Skeleton const & skeleton);
```

## Citation
