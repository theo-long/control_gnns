#! /bin/bash

datasets=(
	"ENZYMES"
	"pubmed"
	"chameleon"
	)
model="gcn"

base_name="baseline_nlnti"
conv_depth=4

for dataset in ${datasets[@]}; do
	name="${base_name}_${model}_d${conv_depth}_${dataset}"
	python local_grid.py -n ${name} --model ${model} --conv_depth ${conv_depth} --dataset ${dataset}
done

base_name="control_nlnti"
conv_depth=4

for dataset in ${datasets[@]}; do
	name="${base_name}_${model}_d${conv_depth}_${dataset}"
	python local_grid.py -n ${name} --model ${model} --conv_depth ${conv_depth} --dataset ${dataset} --control_type "mp" --control_k 0.1
done

base_name="baseline_ti"
conv_depth=10

for dataset in ${datasets[@]}; do
	name="${base_name}_${model}_d${conv_depth}_${dataset}"
	python local_grid.py -n ${name} --model ${model} --conv_depth ${conv_depth} --dataset ${dataset} --time_inv
done

base_name="control_ti"
conv_depth=10

for dataset in ${datasets[@]}; do
	name="${base_name}_${model}_d${conv_depth}_${dataset}"
	python local_grid.py -n ${name} --model ${model} --conv_depth ${conv_depth} --dataset ${dataset} --control_type "mp" --time_inv --control_k 0.1 
done
