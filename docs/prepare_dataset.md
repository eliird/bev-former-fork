

## NuScenes
Download nuScenes V1.0-mini dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running

```sh
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
# untar the folder to ./data
```

**Download CAN bus expansion**
```sh
# download 'can_bus.zip'
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip
unzip can_bus.zip
mv can_bus data/can_bus 
# move can_bus to data dir
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
