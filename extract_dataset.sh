#!bin/bash 
cd data/nuscenes

for f in *.tgz; do 
    tar -xvzf "$f" 

unzip can_bus.zip -d ../
done