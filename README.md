Multi-class U-Net with Tensorflow v2
====

Overview

- Deep learning for defection
- U-Net algorithm
- DAGM dataset
- tensorflow v2
- multi-class

## Requirement

- python 3.6.9
- tensorflow-gpu 2.3.0

## Usage

### 1. Configurate image and lable  
place DAGM dataset like:

```
/home/*username*/DAGM  
 ├ Class1_def
 ├ Class2_def
 ├ Class3_def
 ├ Class4_def
 ├ Class5_def
 └ Class6_def
```

Note:  
Please modify *username* according to you environment.

Then, execute program in turn as:

1. transform_DAGM_to_array.py
1. configurate.cpp

### 2. Check image and label

start check_data.py  

### 3. Training with tensorflow  

start train.py

### 4. Check result

start eval.py

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)