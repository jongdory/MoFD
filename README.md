# Paper

Mixture of Multi-Sequence Feature Difference for Contrast-Enhanced MRI Synthesis

[Jonghun Kim],  [Hyunjin Park]

 [![arXiv](https://img.shields.io/badge/12345-6789-0F75BC.svg?style=plastic)]() [![arXiv](https://img.shields.io/badge/arXiv-1234.5678-b31b1b.svg?style=plastic)]()<br>



# Overview

This repository contains the code for Mixture of Multi-Sequence Feature Difference for Contrast-Enhanced MRI Synthesis

![fig1](./assets/fig1.png)



# Usage

For training, follow the commands below 

```shell
python main.py -b configs/brain.yaml -t True --gpus 0,
```



For sampling, follow the commands below 

```shell
python sampling.py -b configs/brain.yaml -t True --gpus 0,
```



# Citation

```tex
@article{
@misc{kim2026mixt,
      title={Mixture of Multi-Sequence Feature Difference for Contrast-Enhanced MRI Synthesis}, 
      author={Jonghun Kim and Junmo Kwon and Hyunjin Park},
      year={2026},
      eprint={2610.21703},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.21703}, 
}
}
```

