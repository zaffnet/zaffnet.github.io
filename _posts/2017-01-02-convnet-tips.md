---
layout: post
title: "ConvNet Tips"
date:   2017-01-02 15:50:06 +0530
comments: true
permalink: convnet-tips
---

<a name='overfitting'></a>
### Addressing Overfitting

#### Data Augmentation

- Flip the training images over x-axis
- Sample random crops / scales in the original image
- Jitter the colors

#### Dropout
<!--more-->
Dropout is just as effective for Conv layers. Usually people apply less dropout right before early conv layers since there are not that many parameters there compared to later stages of the network (e.g. the fully connected layers).
<!--more-->
