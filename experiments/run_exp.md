rank 1: sbatch experiments/single_node_rank_1.sh 10000 results/precision_ablations 32 10

rank r: sbatch experiments/single_node_rank_r.sh 1000 2 graphs/graphs_rank_2 results/precision_ablations 32 10

sbatch experiments/multi_node_rank_1.sh 20000 results/candidates_per_task 32 10

sbatch experiments/multi_node_rank_r.sh 500 2 graphs/grap
hs_rank_2 results/candidates_per_task 32 10

sbatch experiments/multi_node_rank_1.sh 30000 results/scaling_tests 32 10

sbatch experiments/generate_graphs/single_node_gen_graph.sh 40000 1 graphs/graphs_rank_1

sbatch experiments/multi_node_rank_1_sparse.sh 20000 results/sparsity_tests 32 10