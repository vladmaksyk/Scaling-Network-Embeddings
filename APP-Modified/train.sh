dataset=$1
output=$2
direction=$3
diroption='-undirected 0'
if [[ $direction == "undirected" ]]; then diroption='-undirected 1';
fi
	# if [[ ! -f $dataset-unitweight ]]; then
	awk '{print $0, "1"}' $dataset > $dataset-unitweight
	input=$dataset-unitweight
	echo "created" $dataset-unitweight
	# fi

# if [[ $direction == "undirected" ]]; the
./cli/app -train $dataset-unitweight -save $output-64 $diroption -dimensions 64 -walk_times 80 -sample_times 50 -jump 0.15 -negative_samples 5 -alpha 0.025 -threads 32
./cli/app -train $dataset-unitweight -save $output-128 $diroption -dimensions 128 -walk_times 80 -sample_times 50 -jump 0.15 -negative_samples 5 -alpha 0.025 -threads 32
 	
# else
 	 #./proNet-core/cli/app -train $dataset-unitweight -save $output-128-s50 $diroption -dimensions 128 -walk_times 80 -sample_times 50 -jump 0.15 -negative_samples 5 -alpha 0.025 -threads 32
 	# ./proNet-core/cli/app -train $dataset-unitweight -save $output-128 $diroption -dimensions 128 -walk_times 80 -sample_times 10 -jump 0.15 -negative_samples 5 -alpha 0.025 -threads 32
# fi
