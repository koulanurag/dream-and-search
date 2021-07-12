## dream-and-search
Code for ["Dream and Search to Control: Latent Space Planning for Continuous Control"](https://arxiv.org/abs/2010.09832), published in [Deep RL Workshop, NeurIPS 2020](https://sites.google.com/view/deep-rl-workshop-neurips2020/home)


If you would like to cite it, please use following:
```
@article{koul2020dream,
  title={Dream and Search to Control: Latent Space Planning for Continuous Control},
  author={Koul, Anurag and Kumar, Varun V and Fern, Alan and Majumdar, Somdeb},
  journal={arXiv preprint arXiv:2010.09832},
  year={2020}
}
```

## Installation
1. Install [pytorch (1.6)](https://pytorch.org/)
2. Install [mujoco200 and mujoco license](https://www.roboti.us/index.html)
3. ```pip install -r requirements.txt```

## Usage
- Argument info:```$ python main.py --help```
- Train: ```$ python main.py --case dm_control --env cheetah-run --opr train --action-repeat 2 --search-mode no-search```
- Test: ```$ python main.py --case dm_control --env cheetah-run --opr test --action-repeat 2 --search-mode no-search```

## References:
1. https://github.com/danijar/dreamer
2. https://github.com/yusukeurakami/dreamer-pytorch

