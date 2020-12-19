# cassie
A minimalist repo which contains a gym-like environment for simulating the Cassie robot. This repo is built on the Oregon State University Dynamics Robotics Lab's [cassie-mujoco-sim](https://github.com/osudrl/cassie-mujoco-sim), which is a C-based library for simulating Cassie. In this repo, there is a libcassiemujoco.so file which is precompiled for convenience, but if you would like to recompile from source, simply follow the instructions there.

To use the simulator, you must have MuJoCo 2.0 installed in your home directory, at `$(HOME)/.mujoco/mujoco200`. You should additionally have your `mjkey.txt` at `$(HOME)/.mujoco/mjkey.txt`.

## Setup
```bash
# These commands can go in your ~/.bashrc:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export MUJOCO_KEY_PATH=$HOME/.mujoco/mjkey.txt
```

## Usage
```python
import numpy as np
import time
from cassie import CassieEnv_v2
env = CassieEnv_v2()

while True:
  env.reset()
  done = False
  while not done:
    action = np.random.randn(10)/10
    state, reward, done, _ = env.step(action)
    env.render()
    print(reward, done)
    time.sleep(0.02)

```
