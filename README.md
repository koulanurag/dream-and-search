## dream-and-search
Code for "Dream and Search to Control: Latent Space Planning for Continuous Control".

## Installation
1. Install [pytorch (1.6)](https://pytorch.org/)
2. Install [mujoco200 and mujoco license](https://www.roboti.us/index.html)
3. ```pip install -r requirements.txt```

## Usage
- Train: ```$ python main.py --case dm_control --env cheetah-run --opr train --action-repeat 2 --search-mode no-search```
- Test: ```$ python main.py --case dm_control --env cheetah-run --opr test --action-repeat 2 --search-mode no-search```

## Reference:
1. https://github.com/yusukeurakami/dreamer-pytorch
