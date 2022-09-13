# DeepAL: Deep Active Learning in Python

Python implementations of the following active learning algorithms:

- Bayesian Active Learning Disagreement [1]

## Prerequisites 

- numpy            1.21.2
- scipy            1.7.1
- pytorch          1.10.0
- torchvision      0.11.1
- scikit-learn     1.0.1
- tqdm             4.62.3
- ipdb             0.13.9
- pandas           1.4.4

You can also use the following command to install conda environment

```
conda env create -f environment.yml
```

## Demo 

```
  python demo.py \
      --n_round 10 \
      --n_query 100 \
      --n_init_labeled 10000 \
      --dataset_name MNIST \
      --strategy_name BALDDropout \
      --seed 1
```

Please refer [here](https://arxiv.org/abs/2111.15258) for more details.

## Citing

Forked from:

```
@article{Huang2021deepal,
    author    = {Kuan-Hao Huang},
    title     = {DeepAL: Deep Active Learning in Python},
    journal   = {arXiv preprint arXiv:2111.15258},
    year      = {2021},
}
```

## Reference

[1] Deep Bayesian Active Learning with Image Data, ICML, 2017






