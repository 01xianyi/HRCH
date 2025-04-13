# HRCH
![main](https://github.com/user-attachments/assets/e39edbaf-852b-45e5-8afe-3fedb07efee1)

## Setup

1. Download the datasets:
   - IAPR TC-12: https://pan.baidu.com/s/1LuNtnZRieuJuk7gF64htrg (code: 0411)
   - MIRFlickr-25K: https://pan.baidu.com/s/12EF9fnYxaz_tyunvuUNAxQ (code: 0411)

2. Place data in `./HRCH/data/` directory:
   - For IAPR TC-12: Put `final_data` folder and three additional files
   - For MIRFlickr-25K: Put `mirflickr` folder and three additional files

3. Download RevCol pre-trained weights:
   - Get the tiny model from [RevCol GitHub](https://github.com/megvii-research/RevCol)
   - Place it in the main `./HRCH/` directory

## Training

For MIRFlickr-25K with 128-bit:

```bash
python HRCH.py --data_name mirflickr25k --alpha 0.4 --bit 128 --max_epochs 15 \
  --train_batch_size 224 --eval_batch_size 224 --lr 0.00005 --optimizer Adam \
  --shift 0.1 --margin 0.2 --tau 0.12 --ins 0.8 --pro 1.0 --entroy 0.05 \
  --qua 0.01 --cluster_num 5000,4000,3000,2000 --layers 2,2,4,2 --ld 1
```

## Evaluation

To evaluate using pre-trained weights:

1. Modify the weight path in `utils/config.py` file's `test_path` variable
2. Run evaluation:

```
python HRCH.py --data_name mirflickr25k --bit 128 \
  --cluster_num 5000,4000,3000,2000 --layers 2,2,4,2 --resume True
```
