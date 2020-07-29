mkdir created_data
mkdir created_data/tmp
mkdir created_data/trained_RS
mkdir created_data/l2r
mkdir created_data/hypothesis_data
mkdir created_data/results

mkdir ../experiment3/created_data/
mkdir ../experiment3/created_data/train
mkdir ../experiment3/created_data/test
mkdir ../experiment3/created_data/predictions

for DATASET in "ml1m"
do
	echo "Running experiments for ${DATASET}"
	echo "	time_based_split"
	python time_based_split.py -d $DATASET 
	echo "	create_hypothesis_dataset"
	python create_hypothesis_dataset.py -d $DATASET 
	echo "	h2_ensemble"
	python h2_ensemble.py -d $DATASET 

	cd ../experiment3/ 
	echo "	learn_to_rank"
	python learn_to_rank.py -d $DATASET
	echo "	borda_count"
	python borda_count.py -d $DATASET

	cd ../experiment2/ 
	echo "	evaluate_ensemble"
	python evaluate_ensembles.py -d $DATASET
done


