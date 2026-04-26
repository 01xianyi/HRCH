[English](README.md) | 简体中文

# HRCH

![HRCH Framework](main.png)

## 项目简介

HRCH 是一个面向跨模态检索的层次化哈希模型。本仓库整理了论文实验所需的主要训练、评估、数据加载和辅助代码，方便直接复现和继续实验。

## 环境配置

```bash
conda env create -f HRCH_env.yaml
conda activate HRCH
```

## 数据集

我们提供两类数据：

- 原始数据集
- 可直接使用的处理后 zip 压缩包

原始数据集下载：

- MIRFLICKR25K: [百度网盘](https://pan.baidu.com/s/1vgqIfGeD8-KxXNekoQ-6cw)，提取码：`ubwu`
- IAPR-TC12: [百度网盘](https://pan.baidu.com/s/1c779W9I_E3szolBcPc_uTQ?pwd=f2ij)，提取码：`f2ij`
- NUS-WIDE（top-10 concept）: [百度网盘](https://pan.baidu.com/s/1mlCVxiSsjp7pcQeQQX-yDw)，提取码：`ru9j`
- MSCOCO: [百度网盘](https://pan.baidu.com/s/1h7tje0LdSH2x7pZxyvS3-A?pwd=s286)，提取码：`s286`

处理后 zip 下载：

- `iapr_tc12.zip`: [百度网盘](https://pan.baidu.com/s/15Wb8F1MiJoQ7H8FzGnjUQA?pwd=i7nn)，提取码：`i7nn`
- `mirflickr25k.zip`: [百度网盘](https://pan.baidu.com/s/16gHGAs02I3osv62waH8b6Q?pwd=vfm8)，提取码：`vfm8`
- `mscoco.zip`: [百度网盘](https://pan.baidu.com/s/1DJWEj-AH61vPtTea6N7eXg?pwd=2unx)，提取码：`2unx`
- `nuswide_tc10.zip`: [百度网盘](https://pan.baidu.com/s/1TRmLMXfRmtDNYUcuXtHKjw?pwd=z5iu)，提取码：`z5iu`

如果你使用处理后的 zip 压缩包，推荐统一解压到：

```text
/root/autodl-tmp/datasets/
|- mirflickr25k/
|- iapr_tc12/
|- mscoco/
`- nuswide_tc10/
```

训练时将 `--data_root` 指向这个目录即可。

如果你使用原始数据集，建议按下面的目录名放在：

```text
/root/autodl-tmp/mirflickr25k
/root/autodl-tmp/IAPR-TC12
/root/autodl-tmp/MSCOCO
/root/autodl-tmp/NUSWIDE
```

仓库已经包含对应的数据加载逻辑，保持这些目录名不变即可。

更详细的数据放置说明请参考 [data/DATASET_LOADING.zh-CN.md](data/DATASET_LOADING.zh-CN.md)。

## 模型权重

预训练权重文件位于 `weights/` 目录。仓库中使用的权重组织方式说明请参考 [weights/README.zh-CN.md](weights/README.zh-CN.md)。

## 训练示例

```bash
python HRCH.py --data_name mirflickr25k --alpha 0.4 --bit 128 --max_epochs 15 --train_batch_size 512 --eval_batch_size 256 --lr 0.000075 --optimizer Adam --warmup_epoch 2 --shift 0.3 --margin 2.8 --tau 0.4 --ins 0.6 --pro 0.8 --entroy 0.01 --qua 0.1 --cluster_num 6000,4000,2000,1000 --gpu 0 --layers 2,2,4,3 --seed 3470 --ld 1
```

## 目录结构

```text
HRCH/
|- clustering_loss/
|- data/
|- models/
|- nets/
|- src/
|- utils/
|- weights/
|- HRCH_env.yaml
|- HRCH.py
|- train.py
|- finetune.py
`- README.md
```
