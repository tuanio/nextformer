<p align="center"><img src="https://user-images.githubusercontent.com/30165828/207250159-1738d4f9-84d7-494f-b544-5bf5aec9239d.png" height=100>

<div align="center">

**PyTorch implementation of Nextformer: A ConvNeXt Augmented Conformer For End-To-End Speech Recognition.**

</div>

---

<p align="center"> 
<a href="https://github.com/tuanio/nextformer/blob/main/LICENSE">
    <img src="http://img.shields.io/badge/license-Apache--2.0-informational"> 
</a>
<a href="https://github.com/pytorch/pytorch">
    <img src="http://img.shields.io/badge/framework-PyTorch-informational"> 
</a>
<a href="https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html">
    <img src="https://img.shields.io/badge/code%20style-black-black"> 
</a>
<a href="https://github.com/tuanio/nextformer">
    <img src="http://img.shields.io/badge/build-passing-success"> 
</a>
<!-- <a href="https://github.com/tuanio/nextformer">
    <img src="http://img.shields.io/badge/build-passing-success"> 
</a> -->
</a>

Conformer models have achieved state-of-the-art (SOTA) results in end-to-end speech recognition. However Conformer mainly focuses on temporal modeling while pays less attention on time-frequency property of speech feature. Authors has augment Conformer with ConvNeXt and propose Nextformer structure, they stacks of ConvNeXt block to replace the commonly used subsampling module in Conformer for utilizing the information contained in timefrequency speech feature. Besides, they insert an additional downsampling module in middle of Conformer layers to make Nextformer model more efficient and accurate.

<img src="https://user-images.githubusercontent.com/30165828/207250519-2926ed6f-e895-4d62-b547-47854a5fee04.png" height=600>
  
This repository contains only model code.
  
## Installation
This project recommends Python 3.7 or higher.
We recommend creating a new virtual environment for this project (using virtual env or conda).
  
### Prerequisites
<!-- * Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.   -->
  
### Install from source
<!-- Currently we only support installation from source code using setuptools. Checkout the source code and run the
following commands:  
  
```
pip install -e .
``` -->

## Usage

<!-- ```python
import torch
import torch.nn as nn
from nextformer import Nextformer

batch_size, sequence_length, dim = 3, 12345, 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

criterion = nn.CTCLoss().to(device)

inputs = torch.rand(batch_size, sequence_length, dim).to(device)
input_lengths = torch.LongTensor([12345, 12300, 12000])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

model = Conformer(num_classes=10,
                  input_dim=dim,
                  encoder_dim=32,
                  num_encoder_layers=3).to(device)

# Forward propagate
outputs, output_lengths = model(inputs, input_lengths)

# Calculate CTC Loss
loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
``` -->

## Troubleshoots and Contributing

If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/tuanio/nextformer/issues) on github or  
contacts nvatuan3@gmail.com please.

I appreciate any kind of feedback or contribution. Feel free to proceed with small issues like bug fixes, documentation improvement. For major contributions and new features, please discuss with the collaborators in corresponding issues.

## Code Style

I follow [Black](https://black.readthedocs.io/en/stable/) for code style. Especially the style of docstrings is important to generate documentation.

## Reference

- [Nextformer: A ConvNeXt Augmented Conformer For End-To-End Speech Recognition](https://arxiv.org/abs/2206.14747)
- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/pdf/2005.08100.pdf)
- [sooftware/conformer](https://github.com/sooftware/conformer)
- [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

## Author

- Nguyen Van Anh Tuan [@tuanio](https://github.com/tuanio)
- Contacts: nvatuan3@gmail.com
