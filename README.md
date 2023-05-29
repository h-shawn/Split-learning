# Split-learning

A PyTorch Implementation for experiements in paper: Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge.

## Structure

```shell
Split-learning
├── cloud
│   ├── data.py
│   ├── initCloud.py
│   ├── models.py
│   └── predict.py
├── datasets
│   └── cifar_10
│       ├── batches.meta
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       ├── readme.html
│       └── test_batch
├── mobile
│   ├── data.py
│   ├── initMobile.py
│   ├── models.py
│   └── predict.py
├── model
│   ├── AlexNet.pkl
│   └── VGG16.pkl
├── README.md
└── train
    ├── checkpoint
    ├── data.py
    ├── main.py
    ├── models.py
    └── utils.py
```

## Usage

On cloud:

```
python initCloud.py
```

On mobile:

```
python initMobile.py
```

We have trained two models AlexNet and VGG16 in directory ./model and you can also add some new models to train in directory ./train.

You can change the **x** in initMobile.py to decide which layer to split. 1 means computing on cloud and 0 means computing on mobile.