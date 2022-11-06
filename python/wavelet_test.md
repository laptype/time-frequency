```python
import pywt
import matplotlib.pyplot as plt
import torch
import numpy as np
```


```python
wave = 'haar'
w = pywt.Wavelet(wave)

dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])

print('dec_hi: ', dec_hi, 'dec_lo', dec_lo)

w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

for i in [w_ll, w_lh, w_hl, w_hh]:
    print(i)
    
w_ll = w_ll.unsqueeze(0).unsqueeze(0)
w_lh = w_lh.unsqueeze(0).unsqueeze(0)
w_hl = w_hl.unsqueeze(0).unsqueeze(0)
w_hh = w_hh.unsqueeze(0).unsqueeze(0)

for i in [w_ll, w_lh, w_hl, w_hh]:
    print(i.shape)
```

    dec_hi:  tensor([ 0.7071, -0.7071]) dec_lo tensor([0.7071, 0.7071])
    tensor([[0.5000, 0.5000],
            [0.5000, 0.5000]])
    tensor([[ 0.5000,  0.5000],
            [-0.5000, -0.5000]])
    tensor([[ 0.5000, -0.5000],
            [ 0.5000, -0.5000]])
    tensor([[ 0.5000, -0.5000],
            [-0.5000,  0.5000]])
    torch.Size([1, 1, 2, 2])
    torch.Size([1, 1, 2, 2])
    torch.Size([1, 1, 2, 2])
    torch.Size([1, 1, 2, 2])



```python
img = torch.load('img0.pt')
x = img.contiguous()
x.shape
x.unsqueeze(0).shape
print(x)
```

    tensor([[[0.8824, 0.8824, 0.8706,  ..., 0.7059, 0.7020, 0.6902],
             [0.8784, 0.8863, 0.8784,  ..., 0.7020, 0.7098, 0.6941],
             [0.8706, 0.8824, 0.8824,  ..., 0.6941, 0.7059, 0.6980],
             ...,
             [0.2745, 0.2784, 0.2863,  ..., 0.2627, 0.2588, 0.2471],
             [0.2824, 0.2745, 0.2824,  ..., 0.2627, 0.2510, 0.2667],
             [0.2863, 0.2863, 0.2863,  ..., 0.2549, 0.2588, 0.2784]],
    
            [[0.6353, 0.6353, 0.6235,  ..., 0.4941, 0.4902, 0.4824],
             [0.6314, 0.6392, 0.6314,  ..., 0.4941, 0.5059, 0.4941],
             [0.6235, 0.6353, 0.6353,  ..., 0.4980, 0.5020, 0.4941],
             ...,
             [0.2902, 0.2941, 0.3020,  ..., 0.2471, 0.2431, 0.2314],
             [0.3020, 0.2941, 0.2980,  ..., 0.2471, 0.2353, 0.2510],
             [0.2980, 0.2980, 0.2980,  ..., 0.2392, 0.2431, 0.2627]],
    
            [[0.6706, 0.6706, 0.6588,  ..., 0.5961, 0.5922, 0.5804],
             [0.6667, 0.6745, 0.6667,  ..., 0.5882, 0.5961, 0.5804],
             [0.6588, 0.6706, 0.6706,  ..., 0.5843, 0.5882, 0.5804],
             ...,
             [0.5569, 0.5725, 0.5804,  ..., 0.5451, 0.5412, 0.5294],
             [0.5647, 0.5686, 0.5843,  ..., 0.5451, 0.5333, 0.5490],
             [0.5765, 0.5725, 0.5804,  ..., 0.5373, 0.5451, 0.5647]]])



```python
x_ll = torch.nn.functional.conv2d(x.unsqueeze(0), w_ll.expand(3, 1, -1, -1), stride = 2, groups = 3)
x_lh = torch.nn.functional.conv2d(x.unsqueeze(0), w_lh.expand(3, 1, -1, -1), stride = 2, groups = 3)
x_hl = torch.nn.functional.conv2d(x.unsqueeze(0), w_hl.expand(3, 1, -1, -1), stride = 2, groups = 3)
x_hh = torch.nn.functional.conv2d(x.unsqueeze(0), w_hh.expand(3, 1, -1, -1), stride = 2, groups = 3)

x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
x_hh.shape
```




    torch.Size([1, 3, 450, 1440])




```python
for i, img in enumerate([x_ll, x_lh, x_hl, x_hh]):
    plt.figure(figsize=(20,10))
    plt.subplot(2,2,i+1)
    img = img[0,:,:,:].numpy().transpose(1,2,0)
#     imagee = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    imagee = img * 255
#     print(imagee)
#     print(img)
#     plt.imshow(img,vmin=0,vmax=255)
    plt.imshow(imagee.astype('uint8'))
```


​    
![png](https://gcore.jsdelivr.net/gh/laptype/cloud@main/img/test2_i/22-07-529-output_4_0-56.png)
​    




![png](https://gcore.jsdelivr.net/gh/laptype/cloud@main/img/test2_i/22-07-530-output_4_1-b6.png)
    




![png](https://gcore.jsdelivr.net/gh/laptype/cloud@main/img/test2_i/22-07-530-output_4_2-7b.png)
    




![png](https://gcore.jsdelivr.net/gh/laptype/cloud@main/img/test2_i/22-07-530-output_4_3-c8.png)
    



```python
import torch.nn as nn
import torch
import cv2
import numpy as np
 
 

img = cv2.imread('img/lenna.png')
# img = cv2.imread('img/mountain.png')
print(img.shape)
img = np.swapaxes(img, 0, 2)
print(img.shape)
img = np.swapaxes(img, 1, 2)
print(img.shape)
img = img.astype(np.float32) / 255.0
print(img.shape)
inputs = torch.from_numpy(np.array([img, img]))
inputs = inputs
print(inputs.shape)

# conv = nn.Conv2d(3, 1, kernel_size=3, padding=1, groups=1, stride=1, bias=False)

x_ll = torch.nn.functional.conv2d(inputs, w_ll.expand(3, -1, -1, -1), stride = 2, groups = 3)
x_lh = torch.nn.functional.conv2d(inputs, w_lh.expand(3, -1, -1, -1), stride = 2, groups = 3)
x_hl = torch.nn.functional.conv2d(inputs, w_hl.expand(3, -1, -1, -1), stride = 2, groups = 3)
x_hh = torch.nn.functional.conv2d(inputs, w_hh.expand(3, -1, -1, -1), stride = 2, groups = 3)

# a = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
# conv.weight.data = torch.Tensor([a, a, a]).unsqueeze(0)
# print(conv.weight.shape)

# outputs = conv(inputs)

# outputs = outputs
# print(outputs.shape)
plt.figure(figsize=(20,10))

for i, img in enumerate([x_ll, x_lh, x_hl, x_hh]):
    img = img.detach().numpy()
    print(img.shape)
    img = np.clip(img, 0, 200)
    img = 255 * ((img - img.min()) / (img.max() - img.min()))
    img1 = img[0, 2, :, :]

    plt.subplot(2,2,i+1)
    plt.imshow(img1.astype(np.uint8))
    # cv2.imwrite("/home/wang/xxx.jpg", img1.astype(np.uint8))
    # cv2.imwrite("/home/wang/yyy.jpg", img2.astype(np.uint8))
    # cv2.imshow("window", img.astype(np.uint8))
```

    (512, 512, 3)
    (3, 512, 512)
    (3, 512, 512)
    (3, 512, 512)
    torch.Size([2, 3, 512, 512])
    (2, 3, 256, 256)
    (2, 3, 256, 256)
    (2, 3, 256, 256)
    (2, 3, 256, 256)




![png](https://gcore.jsdelivr.net/gh/laptype/cloud@main/img/test2_i/22-07-530-output_5_1-81.png)
    



```python
import torch.nn as nn
import torch
import cv2
import numpy as np
 
 

# img = cv2.imread('img/lenna.png')
img = cv2.imread('img/mountain.png')
print(img.shape)
img = np.swapaxes(img, 0, 2)
print(img.shape)
img = np.swapaxes(img, 1, 2)
print(img.shape)
img = img.astype(np.float32) / 255.0
print(img.shape)
inputs = torch.from_numpy(np.array([img, img]))
inputs = inputs
print(inputs.shape)

# conv = nn.Conv2d(3, 1, kernel_size=3, padding=1, groups=1, stride=1, bias=False)

x_ll = torch.nn.functional.conv2d(inputs, w_ll.expand(3, -1, -1, -1), stride = 2, groups = 3)
x_lh = torch.nn.functional.conv2d(inputs, w_lh.expand(3, -1, -1, -1), stride = 2, groups = 3)
x_hl = torch.nn.functional.conv2d(inputs, w_hl.expand(3, -1, -1, -1), stride = 2, groups = 3)
x_hh = torch.nn.functional.conv2d(inputs, w_hh.expand(3, -1, -1, -1), stride = 2, groups = 3)

# a = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
# conv.weight.data = torch.Tensor([a, a, a]).unsqueeze(0)
# print(conv.weight.shape)

# outputs = conv(inputs)

# outputs = outputs
# print(outputs.shape)
plt.figure(figsize=(20,10))

for i, img in enumerate([x_ll, x_lh, x_hl, x_hh]):
    img = img.detach().numpy()
    print(img.shape)
    img = np.clip(img, 0, 200)
    img = 255 * ((img - img.min()) / (img.max() - img.min()))
    img1 = img[0, 2, :, :]

    plt.subplot(2,2,i+1)
    plt.imshow(img1.astype(np.uint8))
    # cv2.imwrite("/home/wang/xxx.jpg", img1.astype(np.uint8))
    # cv2.imwrite("/home/wang/yyy.jpg", img2.astype(np.uint8))
    # cv2.imshow("window", img.astype(np.uint8))
```

    (900, 2880, 3)
    (3, 2880, 900)
    (3, 900, 2880)
    (3, 900, 2880)
    torch.Size([2, 3, 900, 2880])
    (2, 3, 450, 1440)
    (2, 3, 450, 1440)
    (2, 3, 450, 1440)
    (2, 3, 450, 1440)




![png](https://gcore.jsdelivr.net/gh/laptype/cloud@main/img/test2_i/22-07-530-output_6_1-4a.png)
    



```python
from test2 import *

img = cv2.imread('img/lenna.png')
# img = cv2.imread('img/mountain.png')
print(img.shape)
img = np.swapaxes(img, 0, 2)
print(img.shape)
img = np.swapaxes(img, 1, 2)
print(img.shape)
img = img.astype(np.float32) / 255.0
print(img.shape)
inputs = torch.from_numpy(np.array([img, img]))
inputs = inputs
print(inputs.shape)

dwt(inputs)
```

    (512, 512, 3)
    (3, 512, 512)
    (3, 512, 512)
    (3, 512, 512)
    torch.Size([2, 3, 512, 512])
    (2, 3, 256, 256)
    (2, 3, 256, 256)
    (2, 3, 256, 256)
    (2, 3, 256, 256)




![png](https://gcore.jsdelivr.net/gh/laptype/cloud@main/img/test2_i/22-07-530-output_7_1-f9.png)
    



```python
B, _, H, W = inputs.shape
x = inputs.view(B, 4, -1, H, W)

C = x.shape[1]

x = x.reshape(B, -1, H, W)
```


    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    ~\AppData\Local\Temp\ipykernel_14820\3951709181.py in <module>
          1 B, _, H, W = inputs.shape
    ----> 2 x = inputs.view(B, 4, -1, H, W)
          3 
          4 C = x.shape[1]
          5 


    RuntimeError: shape '[2, 4, -1, 512, 512]' is invalid for input of size 1572864



```python

```


    ---------------------------------------------------------------------------
    
    NameError                                 Traceback (most recent call last)
    
    ~\AppData\Local\Temp\ipykernel_14820\4009357998.py in <module>
    ----> 1 input_img()


    NameError: name 'input_img' is not defined



```python
from test2 import input_img

input_img()
```


    ---------------------------------------------------------------------------
    
    ImportError                               Traceback (most recent call last)
    
    ~\AppData\Local\Temp\ipykernel_14820\573653075.py in <module>
    ----> 1 from test2 import input_img
          2 
          3 input_img()


    ImportError: cannot import name 'input_img' from 'test2' (D:\study\DL_learning\wavelet_vit\self_test\test2.py)



```python

```
