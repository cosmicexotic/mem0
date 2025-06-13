dataset_name="longmemeval_oracle_sample_seed_42"
# dataset_name="longmemeval_example2"
python evaluation/run_experiments_longmemeval.py --technique_type mem0 --dataset_name ${dataset_name}

# evaluate the results
# python evaluation/evals.py --input_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_False.json --output_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_False_evals.json --dataset longmemeval

# generate the final results
# python evaluation/generate_scores.py --input_file results/${dataset_name}_mem0_results_top_30_filter_False_graph_False_evals.json