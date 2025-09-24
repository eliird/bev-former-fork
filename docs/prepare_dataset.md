

## NuScenes
Download nuScenes V1.0-mini dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running

```sh
bash download_dataset.sh
bash extract_dataset.sh
# untar the folder to ./data
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*
```sh
# python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data

# remove them version flag in case you are using full dataset
python reimplementation/tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes  --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

**Folder structure**
```
bevformer
├── reimplementation/
├── tools/
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```


**Test Quick Training**

```sh
cd reimplementation
python example_training.py quick_test
```