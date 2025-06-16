python evaluation/run_experiments_local.py --technique_type mem0 --is_graph

# # evaluate the results
# python evaluation/evals.py --input_file results/mem0_results_top_30_filter_False_graph_False.json --output_file results/mem0_results_top_30_filter_False_graph_False_evals.json

# # generate the final results
# python evaluation/generate_scores.py --input_file results/mem0_results_top_30_filter_False_graph_False_evals.json