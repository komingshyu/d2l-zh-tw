# 圖像增廣
:label:`sec_image_augmentation`

 :numref:`sec_alexnet`提到過大型資料集是成功應用深度神經網路的先決條件。
圖像增廣在對訓練圖像進行一系列的隨機變化之後，產生相似但不同的訓練樣本，從而擴大了訓練集的規模。
此外，應用圖像增廣的原因是，隨機改變訓練樣本可以減少模型對某些屬性的依賴，從而提高模型的泛化能力。
例如，我們可以以不同的方式裁剪圖像，使感興趣的物件出現在不同的位置，減少模型對於物件出現位置的依賴。
我們還可以調整亮度、顏色等因素來降低模型對顏色的敏感度。
可以說，圖像增廣技術對於AlexNet的成功是必不可少的。本節將討論這項廣泛應用於計算機視覺的技術。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

```{.python .input}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.vision as paddlevision
from paddle import nn
```

## 常用的圖像增廣方法

在對常用圖像增廣方法的探索時，我們將使用下面這個尺寸為$400\times 500$的圖像作為範例。

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch, paddle
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

大多數圖像增廣方法都具有一定的隨機性。為了便於觀察圖像增廣的效果，我們下面定義輔助函式`apply`。
此函式在輸入圖像`img`上多次執行圖像增廣方法`aug`並顯示所有結果。

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### 翻轉和裁剪

[**左右翻轉圖像**]通常不會改變物件的類別。這是最早且最廣泛使用的圖像增廣方法之一。
接下來，我們使用`transforms`模組來建立`RandomFlipLeftRight`例項，這樣就各有50%的機率使圖像向左或向右翻轉。

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

```{.python .input}
#@tab paddle
apply(img, paddlevision.transforms.RandomHorizontalFlip())
```

[**上下翻轉圖像**]不如左右圖像翻轉那樣常用。但是，至少對於這個範例圖像，上下翻轉不會妨礙識別。接下來，我們建立一個`RandomFlipTopBottom`例項，使圖像各有50%的機率向上或向下翻轉。

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

```{.python .input}
#@tab paddle
apply(img,  paddlevision.transforms.RandomVerticalFlip())
```

在我們使用的範例圖像中，貓位於圖像的中間，但並非所有圖像都是這樣。
在 :numref:`sec_pooling`中，我們解釋了匯聚層可以降低卷積層對目標位置的敏感性。
另外，我們可以透過對圖像進行隨機裁剪，使物體以不同的比例出現在圖像的不同位置。
這也可以降低模型對目標位置的敏感性。

下面的程式碼將[**隨機裁剪**]一個面積為原始面積10%到100%的區域，該區域的寬高比從0.5～2之間隨機取值。
然後，區域的寬度和高度都被縮放到200畫素。
在本節中（除非另有說明），$a$和$b$之間的隨機數指的是在區間$[a, b]$中透過均勻取樣獲得的連續值。

```{.python .input}
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab paddle
shape_aug =  paddlevision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### 改變顏色

另一種增廣方法是改變顏色。
我們可以改變圖像顏色的四個方面：亮度、對比度、飽和度和色調。
在下面的範例中，我們[**隨機更改圖像的亮度**]，隨機值為原始圖像的50%（$1-0.5$）到150%（$1+0.5$）之間。

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

```{.python .input}
#@tab paddle
apply(img,  paddlevision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

同樣，我們可以[**隨機更改圖像的色調**]。

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

```{.python .input}
#@tab paddle
apply(img,  paddlevision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

我們還可以建立一個`RandomColorJitter`例項，並設定如何同時[**隨機更改圖像的亮度（`brightness`）、對比度（`contrast`）、飽和度（`saturation`）和色調（`hue`）**]。

```{.python .input}
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab paddle
color_aug =  paddlevision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### [**結合多種圖像增廣方法**]

在實踐中，我們將結合多種圖像增廣方法。比如，我們可以透過使用一個`Compose`例項來綜合上面定義的不同的圖像增廣方法，並將它們應用到每個圖像。

```{.python .input}
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab paddle
augs =  paddlevision.transforms.Compose([
     paddle.vision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**使用圖像增廣進行訓練**]

讓我們使用圖像增廣來訓練模型。
這裡，我們使用CIFAR-10資料集，而不是我們之前使用的Fashion-MNIST資料集。
這是因為Fashion-MNIST資料集中物件的位置和大小已被規範化，而CIFAR-10資料集中物件的顏色和大小差異更明顯。
CIFAR-10資料集中的前32個訓練圖像如下所示。

```{.python .input}
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

```{.python .input}
#@tab paddle
all_images =  paddlevision.datasets.Cifar10(mode='train' , download=True)
print(len(all_images))
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

為了在預測過程中得到確切的結果，我們通常對訓練樣本只進行圖像增廣，且在預測過程中不使用隨機操作的圖像增廣。
在這裡，我們[**只使用最簡單的隨機左右翻轉**]。
此外，我們使用`ToTensor`例項將一批圖像轉換為深度學習框架所要求的格式，即形狀為（批次大小，通道數，高度，寬度）的32位浮點數，取值範圍為0～1。

```{.python .input}
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

```{.python .input}
#@tab paddle
train_augs = paddlevision.transforms.Compose([
     paddlevision.transforms.RandomHorizontalFlip(),
     paddlevision.transforms.ToTensor()])

test_augs = paddlevision.transforms.Compose([
     paddlevision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
接下來，我們定義了一個輔助函式，以便於讀取圖像和應用圖像增廣。Gluon資料集提供的`transform_first`函式將圖像增廣應用於每個訓練樣本的第一個元素（由圖像和標籤組成），即應用在圖像上。有關`DataLoader`的詳細介紹，請參閱 :numref:`sec_fashion_mnist`。
:end_tab:

:begin_tab:`pytorch`
接下來，我們[**定義一個輔助函式，以便於讀取圖像和應用圖像增廣**]。PyTorch資料集提供的`transform`引數應用圖像增廣來轉化圖像。有關`DataLoader`的詳細介紹，請參閱 :numref:`sec_fashion_mnist`。
:end_tab:

```{.python .input}
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

```{.python .input}
#@tab paddle
def load_cifar10(is_train, augs, batch_size):
    dataset = paddlevision.datasets.Cifar10(mode="train", 
                                            transform=augs, download=True)
    dataloader = paddle.io.DataLoader(dataset, batch_size=batch_size, 
                    num_workers=d2l.get_dataloader_workers(), shuffle=is_train)
    return dataloader
```

### 多GPU訓練

我們在CIFAR-10資料集上訓練 :numref:`sec_resnet`中的ResNet-18模型。
回想一下 :numref:`sec_multi_gpu_concise`中對多GPU訓練的介紹。
接下來，我們[**定義一個函式，使用多GPU對模型進行訓練和評估**]。

```{.python .input}
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """用多GPU進行小批次訓練"""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # True標誌允許使用過時的梯度，這很有用（例如，在微調BERT中）
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU進行小批次訓練"""
    if isinstance(X, list):
        # 微調BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab paddle
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU進行小批次訓練
    飛槳不支援在notebook上進行多GPU訓練
    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # 微調BERT中所需（稍後討論）
        X = [paddle.to_tensor(x, place=devices[0]) for x in X]
    else:
        X = paddle.to_tensor(X, place=devices[0])
    y = paddle.to_tensor(y, place=devices[0])
    net.train()
    trainer.clear_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """用多GPU進行模型訓練"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # 4個維度：儲存訓練損失，訓練準確度，例項數，特點數
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU進行模型訓練"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4個維度：儲存訓練損失，訓練準確度，例項數，特點數
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab paddle
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU進行模型訓練
    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = paddle.DataParallel(net)
    for epoch in range(num_epochs):
        # 4個維度：儲存訓練損失，訓練準確度，例項數，特點數
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

現在，我們可以[**定義`train_with_data_aug`函式，使用圖像增廣來訓練模型**]。該函式獲取所有的GPU，並使用Adam作為訓練的最佳化演算法，將圖像增廣應用於訓練集，最後呼叫剛剛定義的用於訓練和評估模型的`train_ch13`函式。

```{.python .input}
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab paddle
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2D]:
        nn.initializer.XavierUniform(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = paddle.optimizer.Adam(learning_rate=lr, parameters=net.parameters())
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices[:1])
```

讓我們使用基於隨機左右翻轉的圖像增廣來[**訓練模型**]。

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## 小結

* 圖像增廣基於現有的訓練資料產生隨機圖像，來提高模型的泛化能力。
* 為了在預測過程中得到確切的結果，我們通常對訓練樣本只進行圖像增廣，而在預測過程中不使用帶隨機操作的圖像增廣。
* 深度學習框架提供了許多不同的圖像增廣方法，這些方法可以被同時應用。

## 練習

1. 在不使用圖像增廣的情況下訓練模型：`train_with_data_aug(no_aug, no_aug)`。比較使用和不使用圖像增廣的訓練結果和測試精度。這個對比實驗能支援圖像增廣可以減輕過擬合的論點嗎？為什麼？
2. 在基於CIFAR-10資料集的模型訓練中結合多種不同的圖像增廣方法。它能提高測試準確性嗎？
3. 參閱深度學習框架的線上文件。它還提供了哪些其他的圖像增廣方法？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/2828)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2829)
:end_tab:

:begin_tab:`paddle`
[Discussions](https://discuss.d2l.ai/t/11801)
:end_tab:
