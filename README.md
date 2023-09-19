# h-signature

A C++ Library w/ Python Bindings for computing the h-signature as defined in \[1\].

## Installation

```shell
pip install h-signature

```

## Quickstart

```python

import numpy as np
import h_signature

# Create two matrices [N, 3] which represents two closed loops in 3D
N = 10
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
tau_loop = np.stack([np.cos(angles), np.sin(angles), np.zeros(N)], axis=1)
obs1_loop = np.stack([np.cos(angles), np.zeros(N), np.sin(angles)], axis=1)
obs2_loop = np.stack([np.cos(angles), np.zeros(N), np.sin(angles)], axis=1) + 1
skeleton = {
    'obs1': obs1_loop,
    'obs2': obs2_loop,
}

# Compute the h-signature of tau_loop with respect to the skeleton
h_sig = h_signature.get_h_signature(tau_loop, skeleton)
print(h_sig)
```

## Citation
