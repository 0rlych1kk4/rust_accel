# rust_accel

**rust_accel** is a Rust-accelerated Python extension module built with **PyO3** and **maturin**, designed to speed up computation-heavy numerical workloads while keeping a clean, Python-native API.

It is ideal for performance-critical paths in data processing, numerical optimization, and machine-learning pipelines where Python ergonomics and Rust speed are both required.

---

## Features

-  High-performance Rust kernels exposed to Python
-  Seamless NumPy-compatible API
-  Parallelized computation using Rayon
-  Clean `python/` source layout (no import shadowing)
-  Prebuilt wheels via maturin
-  Memory-safe Rust implementation

---
## Installation

### From PyPI (recommended)

```bash
pip install rust-accel

From source (development)

```bash
git clone https://github.com/0rlych1kk4/rust_accel.git
cd rust_accel
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin
maturin develop

---
## Usage

### optimize_reconstruction
Computes a batched dot-product–style loss between an input matrix and multiple gradient matrices.

```python

import numpy as np
import rust_accel

x = np.ones((2, 3), dtype=np.float32)

grads = np.stack([
    np.full((2, 3), 2, np.float32),
    np.full((2, 3), 3, np.float32),
], axis=0)

loss = rust_accel.optimize_reconstruction(x, grads)
print(loss)  # 30.0

### clone_grad
Creates a Rust-side copy of a NumPy array and returns it as a flat Python list.

```python
import numpy as np
import rust_accel

x = np.ones((2, 3), dtype=np.float32)
flat = rust_accel.clone_grad(x)

print(len(flat))     # 6
print(flat[:3])      # [1.0, 1.0, 1.0]

---
## Real-World Use Cases
-  Accelerating inner loops in ML / optimization pipelines
-  High-frequency numerical aggregation
-  Scientific computing with Python frontends
-  Hybrid Python–Rust systems where correctness and speed matter
-  Safe native extensions without manual memory management

---

### **2. Contributing Section**

The **Contributing** section is where you invite others to contribute to your project. This usually includes basic steps like forking the repository, making changes, and submitting a pull request.

```markdown
---

## Contributing

We welcome contributions to the `rust_accel` project! If you'd like to contribute, please follow these steps:

1. Fork this repository to your GitHub account.
2. Clone your forked repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/rust_accel.git

