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




# 2026-1-1
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_first_30_rank_2 results/gset_rank_2 2 32 10000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/3_regular_graph_rank_3/n20 results/regular_graph/3_regular_graph_rank_3/n20 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/3_regular_graph_rank_3/n50 results/regular_graph/3_regular_graph_rank_3/n50 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/3_regular_graph_rank_3/n100 results/regular_graph/3_regular_graph_rank_3/n100 3 32 100000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/5_regular_graph_rank_3/n20 results/regular_graph/5_regular_graph_rank_3/n20 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/5_regular_graph_rank_3/n50 results/regular_graph/5_regular_graph_rank_3/n50 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/regular_graph/5_regular_graph_rank_3/n100 results/regular_graph/5_regular_graph_rank_3/n100 3 32 100000


sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p01/n20 results/erdos_renyi/rank_3/p01/n20 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p01/n50 results/erdos_renyi/rank_3/p01/n50 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p01/n100 results/erdos_renyi/rank_3/p01/n100 3 32 10000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p025/n20 results/erdos_renyi/rank_3/p025/n20 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p025/n50 results/erdos_renyi/rank_3/p025/n50 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p025/n100 results/erdos_renyi/rank_3/p025/n100 3 32 10000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p05/n20 results/erdos_renyi/rank_3/p05/n20 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p05/n50 results/erdos_renyi/rank_3/p05/n50 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p05/n100 results/erdos_renyi/rank_3/p05/n100 3 32 10000

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p075/n20 results/erdos_renyi/rank_3/p075/n20 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p075/n50 results/erdos_renyi/rank_3/p075/n50 3 32 10000
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/erdos_renyi/rank_3/p075/n100 results/erdos_renyi/rank_3/p075/n100 3 32 10000

# gset random many
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/70/1_1 results/gset_random_many/70/1_1 1 32 100
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/70/10 results/gset_random_many/70/10 1 32 100
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/70/100 results/gset_random_many/70/100 1 32 100

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/72/1_1 results/gset_random_many/72/1_1 1 32 100
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/72/10 results/gset_random_many/72/10 1 32 100
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/72/100 results/gset_random_many/72/100 1 32 100

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/77/1_1 results/gset_random_many/77/1_1 1 32 100
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/77/10 results/gset_random_many/77/10 1 32 100
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/77/100 results/gset_random_many/77/100 1 32 100

sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/81/1_1 results/gset_random_many/81/1_1 1 32 100
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/81/10 results/gset_random_many/81/10 1 32 100
sbatch experiments/multi_node_rank_r_dir.sh /scratch/bs82/graphs/gset_random_many/81/100 results/gset_random_many/81/100 1 32 100


# 1-8 generate graphs
sbatch experiments/generate_graphs/gen_gset_many.sh 48 /scratch/bs82/graphs/gset_random_many/48/1_1 1 0 20 1 0.9 1.1 gset 1
sbatch experiments/generate_graphs/gen_gset_many.sh 48 /scratch/bs82/graphs/gset_random_many/48/10  1 40 20 1 0   10  gset 1
sbatch experiments/generate_graphs/gen_gset_many.sh 48 /scratch/bs82/graphs/gset_random_many/48/100 1 80 20 1 0   100 gset 1

sbatch experiments/generate_graphs/gen_gset_many.sh 49 /scratch/bs82/graphs/gset_random_many/49/1_1 1 0 20 1 0.9 1.1 gset 1
sbatch experiments/generate_graphs/gen_gset_many.sh 49 /scratch/bs82/graphs/gset_random_many/49/10  1 40 20 1 0   10  gset 1
sbatch experiments/generate_graphs/gen_gset_many.sh 49 /scratch/bs82/graphs/gset_random_many/49/100 1 80 20 1 0   100 gset 1

sbatch experiments/generate_graphs/gen_gset_many.sh 50 /scratch/bs82/graphs/gset_random_many/50/1_1 1 0 20 1 0.9 1.1 gset 1
sbatch experiments/generate_graphs/gen_gset_many.sh 50 /scratch/bs82/graphs/gset_random_many/50/10  1 40 20 1 0   10  gset 1
sbatch experiments/generate_graphs/gen_gset_many.sh 50 /scratch/bs82/graphs/gset_random_many/50/100 1 80 20 1 0   100 gset 1