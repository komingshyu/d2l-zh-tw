# 使用Amazon EC2例項
:label:`sec_aws`

本節將展示如何在原始Linux機器上安裝所有庫。回想一下， :numref:`sec_sagemaker`討論瞭如何使用Amazon SageMaker，而在雲上自己建構例項的成本更低。本示範包括三個步驟。

1. 從AWS EC2請求GPU Linux例項。
1. 安裝CUDA（或使用預裝CUDA的Amazon機器映像）。
1. 安裝深度學習框架和其他庫以執行本書的程式碼。

此過程也適用於其他例項（和其他雲），儘管需要一些細微的修改。在繼續操作之前，你需要建立一個AWS帳戶，有關更多詳細資訊，請參閱 :numref:`sec_sagemaker`。

## 建立和執行EC2例項

登入到你的aws賬戶後，單擊“EC2”（在 :numref:`fig_aws`中用紅色方框標記）進入EC2面板。

![開啟EC2控制檯](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2`顯示EC2面板，敏感帳戶資訊變為灰色。

![EC2面板](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### 預置位置
選擇附近的資料中心以降低延遲，例如“Oregon”（俄勒岡）( :numref:`fig_ec2`右上角的紅色方框）。如果你位於中國，你可以選擇附近的亞太地區，例如首爾或東京。請注意，某些資料中心可能沒有GPU例項。

### 增加限制

在選擇例項之前，請點選 :numref:`fig_ec2`所示左側欄中的“Limits”（限制）標籤檢視是否有數量限制。 :numref:`fig_limits`顯示了此類限制的一個例子。帳號目前無法按地域開啟p2.xlarge例項。如果你需要開啟一個或多個例項，請點選“Request limit increase”（請求增加限制）連結，申請更高的例項配額。一般來說，需要一個工作日的時間來處理申請。

![例項數量限制](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### 啟動例項

接下來，單擊 :numref:`fig_ec2`中紅框標記的“Launch Instance”（啟動例項）按鈕，啟動你的例項。

我們首先選擇一個合適的Amazon機器映像（Amazon Machine Image，AMI）。在搜尋框中輸入“ubuntu”（ :numref:`fig_ubuntu`中的紅色框標記）。

![選擇一個AMI](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2提供了許多不同的例項配置可供選擇。對初學者來說，這有時會讓人感到困惑。 :numref:`tab_ec2`列出了不同合適的計算機。

:不同的EC2例項型別

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | 過時的                         |
| p2   | Kepler K80  | 舊的GPU但Spot例項通常很便宜         |
| g3   | Maxwell M60 | 好的平衡                       |
| p3   | Volta V100  | FP16的高效能                   |
| g4   | Turing T4   | FP16/INT8推理最佳化              |
:label:`tab_ec2`

所有這些伺服器都有多種型別，顯示了使用的GPU數量。例如，p2.xlarge有1個GPU，而p2.16xlarge有16個GPU和更多記憶體。有關更多詳細資訊，請參閱[Amazon EC2 文件](https://aws.amazon.com/ec2/instance-types/)。

![選擇一個例項](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

注意，你應該使用支援GPU的例項以及合適的驅動程式和支援GPU的深度學習框架。否則，你將感受不到使用GPU的任何好處。

到目前為止，我們已經完成了啟動EC2例項的七個步驟中的前兩個步驟，如 :numref:`fig_disk`頂部所示。在本例中，我們保留“3. Configure Instance”（3. 配置例項）、“5. Add Tags”（5. 新增標籤）和“6. Configure Security Group”（6. 配置安全組）步驟的預設配置。點選“4.新增儲存”並將預設硬碟大小增加到64GB( :numref:`fig_disk`中的紅色框標記)。請注意，CUDA本身已經佔用了4GB空間。

![修改硬碟大小](../img/disk.png)
:width:`700px`
:label:`fig_disk`

最後，進入“7. Review”（7. 檢視），點選“Launch”（啟動），即可啟動配置好的例項。系統現在將提示你選擇用於存取例項的金鑰對。如果你沒有金鑰對，請在 :numref:`fig_keypair`的第一個下拉選單中選擇“Create a new key pair”（新建金鑰對），即可產生金鑰對。之後，你可以在此選單中選擇“Choose an existing key pair”（選擇現有金鑰對），然後選擇之前產生的金鑰對。單擊“Launch Instances”（啟動例項）即可啟動建立的例項。

![選擇一個金鑰對](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

如果生成了新金鑰對，請確保下載金鑰對並將其儲存在安全位置。這是你透過SSH連線到伺服器的唯一方式。單擊 :numref:`fig_launching`中顯示的例項ID可檢視該例項的狀態。

![單擊例項ID](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### 連線到例項

如 :numref:`fig_connect`所示，例項狀態變為綠色後，右鍵單擊例項，選擇`Connect`（連線）檢視例項存取方式。

![檢視例項存取方法](../img/connect.png)
:width:`700px`
:label:`fig_connect`

如果這是一個新金鑰，它必須是不可公開檢視的，SSH才能工作。轉到儲存`D2L_key.pem`的資料夾，並執行以下命令以使金鑰不可公開檢視：

```bash
chmod 400 D2L_key.pem
```

![檢視例項存取和啟動方法](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`

現在，複製 :numref:`fig_chmod`下方紅色框中的ssh命令並貼上到命令列：

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

當命令列提示“Are you sure you want to continue connecting (yes/no)”（“你確定要繼續連線嗎？（是/否）”）時，輸入“yes”並按回車鍵登入例項。

你的伺服器現在已就緒。

## 安裝CUDA

在安裝CUDA之前，請確保使用最新的驅動程式更新例項。

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

我們在這裡下載CUDA 10.1。存取NVIDIA的[官方儲存庫](https://developer.nvidia.com/cuda-toolkit-archive) 以找到下載連結，如 :numref:`fig_cuda`中所示。

![查詢CUDA 10.1下載地址](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

將說明覆制貼上到終端上，以安裝CUDA 10.1。

```bash
# 連結和檔名可能會發生更改，以NVIDIA的官方為準
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

安裝程式後，執行以下命令檢視GPU：

```bash
nvidia-smi
```

最後，將CUDA新增到庫路徑以幫助其他庫找到它。

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```

## 安裝庫以執行程式碼

要執行本書的程式碼，只需在EC2例項上為linux使用者執行 :ref:`chap_installation`中的步驟，並使用以下提示在遠端linux伺服器上工作。

* 要在Miniconda安裝頁面下載bash指令碼，請右擊下載連結並選擇“copy Link address”，然後執行`wget [copied link address]`。
* 執行`~/miniconda3/bin/conda init`, 你可能需要執行`source~/.bashrc`，而不是關閉並重新開啟當前shell。

## 遠端執行Jupyter筆記本

要遠端執行Jupyter筆記本，你需要使用SSH埠轉發。畢竟，雲中的伺服器沒有顯示器或鍵盤。為此，請從你的桌上型電腦（或膝上型電腦）登入到你的伺服器，如下所示：

```
# 此命令必須在本地命令列中執行
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```

接下來，轉到EC2例項上本書下載的程式碼所在的位置，然後執行：

```
conda activate d2l
jupyter notebook
```

:numref:`fig_jupyter`顯示了執行Jupyter筆記本後可能的輸出。最後一行是埠8888的URL。

![執行Jupyter Notebook後的輸出（最後一行是埠8888的URL）](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

由於你使用埠轉發到埠8889，請複製 :numref:`fig_jupyter`紅色框中的最後一行，將URL中的“8888”替換為“8889”，然後在本地瀏覽器中開啟它。

## 關閉未使用的例項

由於雲服務是按使用時間計費的，你應該關閉不使用的例項。請注意，還有其他選擇：

* “Stopping”（停止）例項意味著你可以重新啟動它。這類似於關閉常規伺服器的電源。但是，停止的例項仍將按保留的硬碟空間收取少量費用；
* “Terminating”（終止）例項將刪除與其關聯的所有資料。這包括磁碟，因此你不能再次啟動它。只有在你知道將來不需要它的情況下才這樣做。

如果你想要將該例項用作更多例項的範本，請右擊 :numref:`fig_connect`中的例子，然後選擇“Image”$\rightarrow$“Create”以建立該例項的鏡像。完成後，選擇“例項狀態”$\rightarrow$“終止”以終止例項。下次要使用此例項時，可以按照本節中的步驟基於儲存的鏡像建立例項。唯一的區別是，在 :numref:`fig_ubuntu`所示的“1.選擇AMI”中，你必須使用左側的“我的AMI”選項來選擇你儲存的鏡像。建立的例項將保留鏡像硬碟上儲存的資訊。例如，你不必重新安裝CUDA和其他執行時環境。

## 小結

* 我們可以按需啟動和停止例項，而不必購買和製造我們自己的計算機。
* 在使用支援GPU的深度學習框架之前，我們需要安裝CUDA。
* 我們可以使用埠轉發在遠端伺服器上執行Jupyter筆記本。

## 練習

1. 雲提供了便利，但價格並不便宜。瞭解如何啟動[spot例項](https://aws.amazon.com/ec2/spot/)以降低成本。
1. 嘗試使用不同的GPU伺服器。它們有多快？
1. 嘗試使用多GPU伺服器。你能把事情擴大到什麼程度？

[Discussions](https://discuss.d2l.ai/t/5733)
