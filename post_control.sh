#! /bin/bash

model="gcn"
base_name="post_control_nlnti"


dataset="ENZYMES"
depth="4"

name="${base_name}_${model}_d${depth}_${dataset}"
python local_grid_post.py -n ${name} --model ${model} --conv_depth ${depth} --dataset ${dataset} --control_type "mp" --control_k 0.05

dataset="pubmed"
depth="3"

name="${base_name}_${model}_d${depth}_${dataset}"
python local_grid_post.py -n ${name} --model ${model} --conv_depth ${depth} --dataset ${dataset} --control_type "mp" --control_k 0.05

dataset="chameleon"
depth="3"

name="${base_name}_${model}_d${depth}_${dataset}"
python local_grid_post.py -n ${name} --model ${model} --conv_depth ${depth} --dataset ${dataset} --control_type "mp" --control_k 0.05
