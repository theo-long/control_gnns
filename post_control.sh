#! /bin/bash

model="gcn"
base_name="post_nlnti"
control_init="0.1"

dataset="ENZYMES"
depth="4"

name="${base_name}_${model}_d${depth}_${dataset}_init${control_init}"
python local_grid_post.py -n ${name} --model ${model} --conv_depth ${depth} --dataset ${dataset} --control_type "mp" --control_k 0.05 --freeze_base --control_init=${control_init}

dataset="pubmed"
depth="3"

name="${base_name}_${model}_d${depth}_${dataset}_init${control_init}"
python local_grid_post.py -n ${name} --model ${model} --conv_depth ${depth} --dataset ${dataset} --control_type "mp" --control_k 0.05 --freeze_base --control_init=${control_init}

dataset="chameleon"
depth="3"

name="${base_name}_${model}_d${depth}_${dataset}_init${control_init}"
python local_grid_post.py -n ${name} --model ${model} --conv_depth ${depth} --dataset ${dataset} --control_type "mp" --control_k 0.05 --freeze_base --control_init=${control_init}
