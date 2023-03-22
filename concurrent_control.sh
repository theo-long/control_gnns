#! /bin/bash

datasets=("ENZYMES" "pubmed" "chameleon")
depths=("2" "3" "4")

model="gcn"
base_name="concurrent_norelu_nlnti_init0.1"

for dataset in ${datasets[@]}; 
do
	for depth in ${depths[@]};
	do
		name="${base_name}_${model}_d${depth}_${dataset}"

		python local_grid.py -n ${name} --model ${model} --conv_depth ${depth} --dataset ${dataset} --control_type "mp" --control_k 0.05 --control_init 0.1 &

	done
	wait
done
