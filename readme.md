# HCRP
This is the implementation of our paper **Exploring Task Difficulty for Few-Shot Relation Extraction**. 

### Requirements
- ``python 3.6``
- ``PyTorch 1.7.0``
- ``transformers 4.0.0``
- ``numpy 1.19``

## Datasets
We experiment our model on two few-shot relation extraction datasets,
 1. [FewRel 1.0](https://thunlp.github.io/1/fewrel1.html)
 2. [FewRel 2.0](https://thunlp.github.io/2/fewrel2_da.html)
 
Please download data from the official links and put it under the ``./data/``. 

## Evaluation
Please download trained model from [here](https://drive.google.com/drive/folders/1GRQXNNTW-HRpYmwnfI9t3x3VP9V0rvDr?usp=sharing) and put it under the ``./checkpoint/``. To evaluate our model, use command

**FewRel 1.0**
```bash
python train.py \
    --N 10 --K 1 --Q 1 --test_iter 10000\
    --only_test True --load_ckpt "checkpoint/hcrp.pth.tar"
```

**FewRel 2.0**
```bash
python train.py \
    --N 10 --K 1 --Q 1 --test_iter 10000\
    --val val_pubmed --test val_pubmed --ispubmed True\
    --only_test True --load_ckpt "checkpoint/hcrp-da.pth.tar"
```

## Training
**FewRel 1.0**

To run our model, use command

```bash
python train.py
```

This will start the training and evaluating process of HCRP in a 10-way-1-shot setting. You can also use different args to start different process. Some of them are here:

* `train / val / test`: Specify the training / validation / test set.
* `trainN`: N in N-way K-shot. `trainN` is the specific N in training process.
* `N`: N in N-way K-shot.
* `K`: K in N-way K-shot.
* `Q`: Sample Q query instances for each relation.

There are also many args for training (like `batch_size` and `lr`) and you can find more details in our codes.

**FewRel 2.0**

Use command
```bash
python train.py \
    --val val_pubmed --test val_pubmed --ispubmed True --lamda 2.5
```
## Results

**FewRel 1.0**

|                   | 5-way-1-shot | 5-way-5-shot | 10-way-1-shot | 10-way-5-shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 90.90 | 93.22 | 84.11 | 87.79 |
| Test              | 93.76 | 95.66 | 89.95 | 92.10 |

**FewRel 2.0**

|                   | 5-way-1-shot | 5-way-5-shot | 10-way-1-shot | 10-way-5-shot |
|  ---------------  | -----------  | ------------- | ------------ | ------------- |
| Val               | 78.90 | 83.22 | 68.99 | 74.45 |
| Test              | 76.34 | 83.03 | 63.77 | 72.94 |

## Cite
If you use the code, please cite the following paper: "Exploring Task Difficulty for Few-Shot Relation Extraction" Jiale Han, Bo Cheng and Wei Lu. EMNLP (2021)
```bash
@inproceedings{han2021exploring,
    title = {Exploring Task Difficulty for Few-Shot Relation Extraction},
    author = {Han, Jiale and Cheng, Bo and Lu, Wei},
    booktitle = {Proc. of EMNLP},
    year={2021}
}
```
