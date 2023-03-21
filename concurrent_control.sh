#! /bin/bash

datasets=("ENZYMES" "pubmed" "chameleon")
depths=("2" "3" "4")

model="gcn"
base_name="concurrent_control_nlnti"

for dataset in ${datasets[@]}; 
do
	for depth in ${depths[@]};
	do
		name="${base_name}_${model}_d${depth}_${dataset}"

		python local_grid.py -n ${name} --model ${model} --conv_depth ${depth} --dataset ${dataset} --control_type "mp" --control_k 0.05 &

	done
	wait
done
