rank 1: sbatch experiments/single_node_rank_1.sh 10000 results/precision_ablations 32 10

rank r: sbatch experiments/single_node_rank_r.sh 1000 2 /scratch/bs82/graphs/graphs_rank_2 results/precision_ablations 32 10

sbatch experiments/multi_node_rank_1.sh 20000 results/candidates_per_task 32 10

sbatch experiments/multi_node_rank_r.sh 800 2 /scratch/bs82/graphs/graphs_rank_2 results/candidates_per_task 32 10

sbatch experiments/multi_node_rank_1.sh 30000 results/scaling_tests 32 10

sbatch experiments/multi_node_rank_r.sh 1000 2 results/scaling_tests 32 50000

sbatch experiments/generate_graphs/single_node_gen_graph.sh 40000 1 /scratch/bs82/graphs/graphs_rank_1

sbatch experiments/generate_graphs/single_node_gen_graph.sh 800 2 /scratch/bs82/graphs/graphs_rank_2

sbatch experiments/multi_node_rank_1_sparse.sh 20000 results/sparsity_tests 32 10

sbatch experiments/multi_node_rank_1_incremental.sh 20000 results/incremental_tests 32 10

sbatch experiments/multi_node_rank_r.sh 800 2 /scratch/bs82/graphs/graphs_rank_2 results/scaling_tests 32 10

sbatch experiments/multi_node_rank_1_two_stage.sh 10000 results/two_stage_tests 32 10 42 10

sbatch experiments/multi_node_rank_1.sh 10000 results/scaling_tests 32 10

sbatch experiments/multi_node_rank_r.sh 800 2 /scratch/bs82/graphs/graphs_rank_2 results/scaling_tests 32 10000

sbatch experiments/multi_node_rank_r.sh 1000 2 /scratch/bs82/graphs/graphs_rank_2 results/scaling_tests 32 10000

debug: sbatch experiments/multi_node_rank_r.sh 15 3 /scratch/bs82/graphs/debug_graphs results/debug 32 1000 1

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset results/gset/pignn pignn


sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p01/n20 results/erdos_renyi/rank_1/p01/n20 1 32 10
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p01/n50 results/erdos_renyi/rank_1/p01/n50 1 32 10
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p01/n100 results/erdos_renyi/rank_1/p01/n100 1 32 10

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p025/n20 results/erdos_renyi/rank_1/p025/n20 1 32 10
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p025/n50 results/erdos_renyi/rank_1/p025/n50 1 32 10
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p025/n100 results/erdos_renyi/rank_1/p025/n100 1 32 10

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p05/n20 results/erdos_renyi/rank_1/p05/n20 1 32 10
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p05/n50 results/erdos_renyi/rank_1/p05/n50 1 32 10
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p05/n100 results/erdos_renyi/rank_1/p05/n100 1 32 10

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p075/n20 results/erdos_renyi/rank_1/p075/n20 1 32 10
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p075/n50 results/erdos_renyi/rank_1/p075/n50 1 32 10
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_1/p075/n100 results/erdos_renyi/rank_1/p075/n100 1 32 10


sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/3_regular_graph_rank_2/n20 results/regular_graph/3_regular_graph_rank_2/n20 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/3_regular_graph_rank_2/n50 results/regular_graph/3_regular_graph_rank_2/n50 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/3_regular_graph_rank_2/n100 results/regular_graph/3_regular_graph_rank_2/n100 2 32 10000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/5_regular_graph_rank_2/n20 results/regular_graph/5_regular_graph_rank_2/n20 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/5_regular_graph_rank_2/n50 results/regular_graph/5_regular_graph_rank_2/n50 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/5_regular_graph_rank_2/n100 results/regular_graph/5_regular_graph_rank_2/n100 2 32 10000


sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p01/n20 results/erdos_renyi/rank_2/p01/n20 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p01/n50 results/erdos_renyi/rank_2/p01/n50 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p01/n100 results/erdos_renyi/rank_2/p01/n100 2 32 10000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p025/n20 results/erdos_renyi/rank_2/p025/n20 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p025/n50 results/erdos_renyi/rank_2/p025/n50 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p025/n100 results/erdos_renyi/rank_2/p025/n100 2 32 10000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p05/n20 results/erdos_renyi/rank_2/p05/n20 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p05/n50 results/erdos_renyi/rank_2/p05/n50 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p05/n100 results/erdos_renyi/rank_2/p05/n100 2 32 10000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p075/n20 results/erdos_renyi/rank_2/p075/n20 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p075/n50 results/erdos_renyi/rank_2/p075/n50 2 32 1000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_2/p075/n100 results/erdos_renyi/rank_2/p075/n100 2 32 10000

