# Beyond Part Models: Person Retrieval with Refined Part Pooling

This project tries to reproduce paper [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349), and now is in processing.

## Current Results

In Market-1501:

|                          | mAP (%) | Rank-1(%) |
|:------------------------:|:-------:|:---------:|
| Market-1501(paper, PCB)  |  77.30  |   92.40   |
| Market-1501(paper, +RPP) |  81.60  |   93.80   |
|        Market-1501       |  74.03  |   89.43   |

## Usage

```text
usage: main.py [-h] [--params-filename PARAMS_FILENAME] [--use-gpu USE_GPU]
               [--world-size WORLD_SIZE] [--dist-url DIST_URL]
               [--dist-rank DIST_RANK] [--last-conv LAST_CONV]
               [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]
               [--load-once LOAD_ONCE] [--epoch EPOCH] [--stage STAGE]
               [--test-type TEST_TYPE] [--rpp-std RPP_STD]
               [--conv-std CONV_STD]

Person Re-Identification Reproduce

optional arguments:
  -h, --help            show this help message and exit
  --params-filename PARAMS_FILENAME
                        filename of model parameters.
  --use-gpu USE_GPU     set 1 if want to use GPU, otherwise 0. (default 1)
  --world-size WORLD_SIZE
                        number of distributed processes. (default 1)
  --dist-url DIST_URL   the master-node's address and port
  --dist-rank DIST_RANK
                        rank of distributed process. (default 0)
  --last-conv LAST_CONV
                        whether contains last convolution layter. (default 1)
  --batch-size BATCH_SIZE
                        training data batch size. (default 64)
  --num-workers NUM_WORKERS
                        number of workers when loading data. (default 20)
  --load-once LOAD_ONCE
                        load all of data at once. (default 0)
  --epoch EPOCH         number of epochs. (default 60)
  --stage STAGE         running stage. train, test or all. (default train)
  --test-type TEST_TYPE
                        model type when testing. pcb, rpp or fnl. (default
                        pcb)
  --rpp-std RPP_STD     standard deviation of initialization of rpp layer.
                        (default 0.01)
  --conv-std CONV_STD   standard deviation of initialization of conv layer.
                        (default 0.001)
```