# RAM
[![Packagist](https://img.shields.io/badge/Tensorflow-1.3.0-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Tensorlayer-1.6.1-blue.svg)]()    


Abstract
---
This is the re-implementation of the original paper - Recurrent Models of Visual Attention. There're two previous implementation link:    
[link1](https://github.com/zhongwen/RAM)     
[link2](https://github.com/jlindsey15/RAM)    
I also provide the [link](https://arxiv.org/abs/1406.6247)[1] to the original paper. 
    
Module Structure
---

![](https://github.com/SunnerLi/ram/blob/master/img/network.jpg) 

Contritution
----
1. First, the original project I refer cannot execute properly since some API can only support the version which are lower than 1.0.0. This project use more newer API to adapt the current version of Tensorflow.     
2. Next, I use tensorlayer to simplify some common operations. In original project, the author just use native tensorflow to build the whole module. This change can make the whole program be more clear.    
3. Last but not least, In the original paper, it shows that the baseline mechanism can reduce the influence of unstable variance. In my implementation, I use batch normalization layer to reach the same purpose. Luckily, the result is better over my expectation.     

Reference
----

[1] V.Mnih, N.Heess, A.Graves, and K.Kavukcuoglu, “Recurrent Models of Visual Attention,” _Arxiv_, 2014.