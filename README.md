# Optimization of Rank Losses for Image Retrieval

This repo contains the official PyTorch implementation of our paper: [Optimization of Rank Losses for Image Retrieval](https://arxiv.org/abs/2207.04873) (under-review TPAMI).

![figure_methode](https://github.com/elias-ramzi/SupRank/blob/main/.github/figures/figure_intro_tpami.png)

## Use SupRank

This will create a virtual environment and install the dependencies described in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Datasets

We use the following datasets for our paper:

- [H-GLDv2](https://github.com/cvdfoundation/google-landmark)
- [iNaturalist-2018](https://github.com/visipedia/inat_comp/tree/master/2018#Data) with [splits](https://drive.google.com/file/d/1sXfkBTFDrRU3__-NUs1qBP3sf_0uMB98/view?usp=sharing)
- [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
- [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- [DyML-datasets](https://onedrive.live.com/?authkey=%21AMLHa5h%2D56ZZL94&id=F4EF5F480284E1C2%21106&cid=F4EF5F480284E1C2)

Once extracted the code should work with the base structure of the datasets. You must precise the direction of the dataset to run an experiment:

```bash
dataset.data_dir=/Path/To/Your/Data/Stanford_Online_Products
```

For iNat you must put the split in the folder of the dataset: `Inaturalist/Inat_dataset_splits`.

You can also tweak the `lib/expand_path.py` function, as it is called for most path handling in the code.

### Add you dataset

When implementing your custom dataset it shoud herit from `BaseDataset`

```python
from suprank.datasets.base_dataset import BaseDataset


class CustomDataset(BaseDataset):
  HIERARCHY_LEVEL = L

  def __init__(data_dir, mode, transform, **kwargs):
    self.paths = ...
    self.labels = ...  # should a numpy array of ndim == 2

    super().__init__(**kwargs)  # this should be at the end.

```

Then add you `CustomDataset` to the `__init__.py` file of `datasets`.

```python
from .custom_dataset import CustomDataset

__all__ = [
    'CustomDataset',
]
```

Finally you should create a config file `custom_dataset.yaml` in `suprank/config/dataset`.

## Run the code

The code uses Hydra for the config. You can override arguments from command line or change a whole config. You can easily add other configs in suprank/config.

Do not hesitate to create an issue if you have trouble understanding the configs, I will gladly answer you.

### iNaturalist

<details>
  <summary><b>iNat-base</b></summary><br/>

```bash
CUDA_VISIBLE_DEVICES='0' python suprank/run.py \
'experiment_name=HAPPIER_iNat_base' \
'log_dir=experiments/HAPPIER' \
seed=0 \
accuracy_calculator.compute_for_hierarchy_levels=[0,1] \
warmup_step=5 \
optimizer=inat \
model=resnet_ln \
dataset=inat_base \
loss=HAPPIER_inat
```

</details>

<details>
  <summary><b>iNat-full</b></summary><br/>

```bash
CUDA_VISIBLE_DEVICES='0' python suprank/run.py \
'experiment_name=HAPPIER_iNat_full' \
'log_dir=experiments/HAPPIER/' \
seed=0 \
accuracy_calculator.compute_for_hierarchy_levels=[0,1,2,3,4,5,6] \
warmup_step=5 \
optimizer=inat \
model=resnet_ln \
dataset=inat_full \
loss=HAPPIER_inat
```

</details>

### Stanford Online Products

<details>
  <summary><b>SOP</b></summary><br/>

```bash
CUDA_VISIBLE_DEVICES='0' python suprank/run.py \
'experiment_name=HAPPIER_SOP' \
'log_dir=experiments/HAPPIER' \
seed=0 \
max_iter=100 \
warmup_step=5 \
accuracy_calculator.compute_for_hierarchy_levels=[0,1] \
optimizer=sop \
model=resnet_ln \
dataset=sop \
loss=HAPPIER_SOP
```

</details>

### Dynamic Metric Learning

<details>
  <summary><b>DyML-Vehicle</b></summary><br/>

```bash
CUDA_VISIBLE_DEVICES='0' python suprank/run.py \
'experiment_name=HAPPIER_dyml_vehicle' \
'log_dir=experiments/HAPPIER' \
seed=0 \
accuracy_calculator.compute_for_hierarchy_levels=[0] \
accuracy_calculator.overall_accuracy=True \
accuracy_calculator.exclude=[NDCG,H-AP] \
accuracy_calculator.recall_rate=[10,20] \
accuracy_calculator.with_binary_asi=True \
optimizer=dyml \
model=dyml_resnet34 \
dataset=dyml_vehicle \
loss=HAPPIER
```

</details>

<details>
  <summary><b>DyML-Animal</b></summary><br/>

```bash
CUDA_VISIBLE_DEVICES='2' python suprank/run.py \
'experiment_name=HAPPIER_dyml_animal' \
'log_dir=experiments/HAPPIER' \
seed=0 \
accuracy_calculator.compute_for_hierarchy_levels=[0] \
accuracy_calculator.overall_accuracy=True \
accuracy_calculator.exclude=[NDCG,H-AP] \
accuracy_calculator.recall_rate=[10,20] \
accuracy_calculator.with_binary_asi=True \
optimizer=dyml \
model=dyml_resnet34 \
dataset=dyml_animal \
loss=HAPPIER_5
```

</details>

<details>
  <summary><b>DyML-Product</b></summary><br/>

```bash
CUDA_VISIBLE_DEVICES='1' python suprank/run.py \
'experiment_name=HAPPIER_dyml_product' \
'log_dir=experiments/HAPPIER' \
seed=0 \
max_iter=20 \
warmup_step=5 \
accuracy_calculator.compute_for_hierarchy_levels=[0,1,2] \
accuracy_calculator.overall_accuracy=True \
accuracy_calculator.exclude=[NDCG,H-AP] \
accuracy_calculator.recall_rate=[10,20] \
accuracy_calculator.with_binary_asi=True \
optimizer=dyml_product \
model=dyml_resnet34_product \
dataset=dyml_product \
loss=HAPPIER_product
```

</details>

## Suggested citation

Please consider citing our works:

```text
@inproceedings{ramzi2023optimization,
  author = {Ramzi, E. and Audebert, N. and Rambour, C. and Araujo, A. and Bitot, X. and Thome, N.},
  title = {{Optimization of Rank Losses for Image Retrieval}},
  year = {2023},
  booktitle = {In submission to: IEEE Transactions on Pattern Analysis and Machine Intelligence},
}

@inproceedings{ramzi2022hierarchical,
  title={Hierarchical Average Precision Training for Pertinent Image Retrieval},
  author={Ramzi, Elias and Audebert, Nicolas and Thome, Nicolas and Rambour, Cl{\'e}ment and Bitot, Xavier},
  booktitle={European Conference on Computer Vision},
  pages={250--266},
  year={2022},
  organization={Springer}
}

@article{ramzi2021robust,
  title={Robust and decomposable average precision for image retrieval},
  author={Ramzi, Elias and Thome, Nicolas and Rambour, Cl{\'e}ment and Audebert, Nicolas and Bitot, Xavier},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={23569--23581},
  year={2021}
}
```

## Resources

Links to repo with useful features used for this code:

- ROADMAP: <https://github.com/elias-ramzi/ROADMAP>
- HAPPIER: <https://github.com/elias-ramzi/HAPPIER>
- GLDv2: <https://github.com/cvdfoundation/google-landmark>
- Hydra: <https://github.com/facebookresearch/hydra>
- NSM: <https://github.com/azgo14/classification_metric_learning>
- PyTorch: <https://github.com/pytorch/pytorch>
- Pytorch Metric Learning (PML): <https://github.com/KevinMusgrave/pytorch-metric-learning>
