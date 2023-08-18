# 計算機視覺
:label:`chap_cv`

近年來，深度學習一直是提高計算機視覺系統效能的變革力量。
無論是醫療診斷、自動駕駛，還是智慧濾波器、攝影頭監控，許多計算機視覺領域的應用都與我們當前和未來的生活密切相關。
可以說，最先進的計算機視覺應用與深度學習幾乎是不可分割的。
有鑑於此，本章將重點介紹計算機視覺領域，並探討最近在學術界和行業中具有影響力的方法和應用。

在 :numref:`chap_cnn`和 :numref:`chap_modern_cnn`中，我們研究了計算機視覺中常用的各種卷積神經網路，並將它們應用到簡單的圖像分類任務中。
本章開頭，我們將介紹兩種可以改進模型泛化的方法，即*圖像增廣*和*微調*，並將它們應用於圖像分類別。
由於深度神經網路可以有效地表示多個層次的圖像，因此這種分層表示已成功用於各種計算機視覺任務，例如*目標檢測*（object detection）、*語義分割*（semantic segmentation）和*樣式遷移*（style transfer）。
秉承計算機視覺中利用分層表示的關鍵思想，我們將從物體檢測的主要元件和技術開始，繼而展示如何使用*完全卷積網路*對圖像進行語義分割，然後我們將解釋如何使用樣式遷移技術來產生像本書封面一樣的圖像。
最後在結束本章時，我們將本章和前幾章的知識應用於兩個流行的計算機視覺基準資料集。

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```
