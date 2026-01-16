from pathlib import Path
import json
import argparse
from collections import defaultdict
import re 
import seaborn as sns
import matplotlib.pyplot as plt

ALG_NAMES = {
    "rank1": "Our Rank-1 Algorithm",
    "rank2": "Our Rank-2 Algorithm",
    "rank3": "Our Rank-3 Algorithm",
    "ros" : "ROS",
    "ANYCSP": "ANYCSP",
    "pignn": "PI-GNN",
    "md": "MD",
    "gw": "Goemans-Williamson",
    "random": "Random",
    "bqp": "BQP",
    "genetic": "Genetic",
    "coloring": "Coloring",
}

FAMILY_NAMES = {
    "erdos_renyi": "Erdos-Renyi",
    "regular_graph": "Regular",
    "gset": "GSet",
}

PARAM_MAP = {
    "Erdos-Renyi": "p = ",
    "Regular": "Degree ",
    "GSet": "Size ",
}

GSET_RE = re.compile(r'(?:_result_n|Q_gset_)(\d+)')
SMALL_RE = re.compile(r'Q(?:_\d+)?_seed_(\d+)(?:_r?_?\d+)?')

def recursive_defaultdict_to_dict(d):
    if isinstance(d, dict):
        return {k: recursive_defaultdict_to_dict(v) for k, v in d.items()}
    else:
        return d
    
def nested():
    return defaultdict(nested)

def load_gset_results(folder, args):
    scores_dict = nested()
    time_dict = nested()
    
    all_algs = set()
    sizes = defaultdict()

    for result_path in folder.rglob("*.json"):
        rel_path = result_path.relative_to(folder)
        if len(rel_path.parts) != 2:
            continue

        alg = rel_path.parts[0]
        try:
            alg = ALG_NAMES[alg]
        except:
            pass

        all_algs.add(alg)

        match = GSET_RE.search(rel_path.stem)
        if not match:
            raise ValueError(f"Cannot extract Gset number for {str(result_path)}")
        gsetNumber = "gset" + (match.group(1) or match.group(2))

        with open(result_path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                print(alg, gsetNumber)
                continue
    
        try:
            if "Rank" not in alg and "Random" not in alg:
                sizes[gsetNumber] = data["n"]
            else:
                sizes[gsetNumber] = len(data["best_k"])
        except KeyError:
            print(f"{alg}, {result_path} does not contain required size information")
            sizes[gsetNumber] = 0

        size = sizes[gsetNumber]

        # --- extract score ---
        if "best_score" in data:
            score = data["best_score"] / 3
        elif "maxcut" in data:
            score = data["maxcut"] / 3
        else:
            print("failed to find score for ", result_path)
            continue
        
        if abs(score - round(score)) > 1e-5:
            print(f"score {score} at {result_path} is not whole")
            continue
        score = round(score)

        # --- extract time ---
        if "time_seconds" in data:
            time = data["time_seconds"]
        elif "alg_time_seconds" in data:
            time = data["alg_time_seconds"]
        else:
            print("failed to find time for ", result_path)
        
        scores_dict[size][gsetNumber][alg] = score 
        time_dict[size][gsetNumber][alg] = time
    return scores_dict, time_dict
        
def load_small_results(folder, args):
    scores_dict = nested()
    time_dict = nested()
    
    all_algs = set()
    sizes = defaultdict()

    for result_path in folder.rglob("*.json"):
        rel_path = result_path.relative_to(folder)
        if len(rel_path.parts) != 4:
            continue

        alg, param, size, _ = rel_path.parts
        try:
            alg = ALG_NAMES[alg]
        except:
            pass

        all_algs.add(alg)

        match = SMALL_RE.search(rel_path.stem)
        if not match:
            raise ValueError(f"Cannot extract seed from {str(result_path)}")
        seed = "seed" + (match.group(1) or match.group(2))

        with open(result_path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                print(f"Failed to load json at {str(result_path)}")
                continue

        # --- extract score ---
        if "best_score" in data:
            score = data["best_score"] / 3
        elif "maxcut" in data:
            score = data["maxcut"] / 3
        else:
            print("failed to find score for ", result_path)
            continue

        if abs(score - round(score)) > 1e-5:
            print(f"score {score} at {result_path} is not whole")
            continue
        score = round(score)

        # --- extract time ---
        if "time_seconds" in data:
            time = data["time_seconds"]
        elif "alg_time_seconds" in data:
            time = data["alg_time_seconds"]
        else:
            print("failed to find time for ", result_path)
            continue

        scores_dict[size][param][seed][alg] = score
        time_dict[size][param][seed][alg] = time
        
    return scores_dict, time_dict
     
def load_data(root, args):
    scores_dicts = {}
    time_dicts = {}
    
    all_algs = set()
    sizes = defaultdict()

    for graph_folder in root.iterdir():
        if graph_folder.is_dir() and graph_folder.name != "summaries":
            if graph_folder.name == "gset":
                print(f"Entering folder {graph_folder.name}.")
                scores_dicts[graph_folder.name], time_dicts[graph_folder.name] = load_gset_results(graph_folder, args)
            else:
                print(f"Entering folder {graph_folder.name}.")
                scores_dicts[graph_folder.name], time_dicts[graph_folder.name] = load_small_results(graph_folder, args)

    return scores_dicts, time_dicts
    
def compute_ratios(scores_dicts):
    # compute ratios
    ratios = nested()
    
    for graph_family, scores_dict in scores_dicts.items():  
        if graph_family ==  "gset": 
            for size, size_dict in scores_dict.items():
                for graph, alg_scores in size_dict.items():
                    max_score = max(alg_scores.values())

                    if max_score == 0:
                        print(f"scores all 0 for {graph}" )
                        ratios[graph_family][size][graph] = [0 for _ in alg_scores]
                    else:
                        ratios[graph_family][size][graph] = {alg: alg_scores[alg] / max_score for alg in alg_scores}
        else:
            for size, size_dict in scores_dict.items():
                for param, param_dict in size_dict.items():
                    for seed, alg_scores in param_dict.items():
                        max_score = max(alg_scores.values())

                        if max_score == 0:
                            print(f"scores all 0 for {graph}" )
                            ratios[graph_family][size][param][seed] = [0 for _ in alg_scores]
                        else:
                            ratios[graph_family][size][param][seed] = {alg: alg_scores[alg] / max_score for alg in alg_scores}

    return ratios

def calculate_averages(ratio_dicts, time_dicts):
    avg_ratios = nested()
    avg_times = nested()
    for graph_family, ratio_dict in ratio_dicts.items():
        print(f"Calculating averages for {graph_family}.")
        try:
            # compute average ratios
            for size, gdict in ratio_dict.items():
                alg_dict = nested()
                for graph, alg_ratios in gdict.items():
                    for alg, ratio in alg_ratios.items():
                        if alg not in alg_dict:
                            alg_dict[alg] = [ratio]
                        else:
                            alg_dict[alg].append(ratio)
                for alg in alg_dict:
                    avg_ratios[graph_family][size][alg] = sum(alg_dict[alg]) / len(alg_dict[alg])

            # compute average times
            for size, gdict in time_dicts[graph_family].items():
                alg_dict = nested()
                for graph, alg_times in gdict.items():
                    for alg, time in alg_times.items():
                        if alg not in alg_dict:
                            alg_dict[alg] = [time]
                        else:
                            alg_dict[alg].append(time)
                for alg in alg_dict:
                    avg_times[graph_family][size][alg] = sum(alg_dict[alg]) / len(alg_dict[alg])

        except TypeError as e:
            # compute average ratios
            for size, size_dict in ratio_dict.items():
                for param, param_dict in size_dict.items():
                    for seed, alg_ratios in param_dict.items():
                        alg_dict = nested()
                        for alg, ratio in alg_ratios.items():
                            if alg not in alg_dict:
                                alg_dict[alg] = [ratio]
                            else:
                                alg_dict[alg].append(ratio)
                    for alg in alg_dict:
                        avg_ratios[graph_family][size][param][alg] = sum(alg_dict[alg]) / len(alg_dict[alg])

            # compute average times
            for size, size_dict in time_dicts[graph_family].items():
                for param, param_dict in size_dict.items():
                    for seed, alg_times in param_dict.items():
                        alg_dict = nested()
                        for alg, time in alg_times.items():
                            if alg not in alg_dict:
                                alg_dict[alg] = [time]
                            else:
                                alg_dict[alg].append(time)
                    for alg in alg_dict:
                        avg_times[graph_family][size][param][alg] = sum(alg_dict[alg]) / len(alg_dict[alg])

    return avg_ratios, avg_times

def construct_one_chart(d, dstr, out_dir, graph_family):
    spl = graph_family.split()
    graph_size = None
    if len(spl) == 2:
        graph_family, graph_size_str = spl
        try:
            graph_size = graph_size_str.split("n")[1]
        except:
            pass
    if graph_family in FAMILY_NAMES:
        graph_family = FAMILY_NAMES[graph_family]

    algs = set()
    for alg_list in d.values():
       algs.update(alg_list)
    algs = list(algs)
    algs.sort(key=lambda x: (
        0 if x.startswith("Our Rank-") else 1,
        int(x.split()[1].split("-")[1]) if x.startswith("Our Rank-") else 0
    ))

    x_positions, heights, bar_colors, bar_labels = [], [], [], [] 
    used_labels = set()

    group_centers, group_labels = [], []

    x = 0
    group_gap = 1.5  # space between (graph, param) groups
    bar_width = 0.8

    colors = sns.color_palette("muted")
    alg_colors = {
        alg: colors[i % len(colors)]
        for i, alg in enumerate(algs)
    }

    for param, avgs in sorted(d.items()):
        start_x = x
        for alg in algs:
            x_positions.append(x)
            if alg in avgs:
                heights.append(avgs[alg])
            else:
                heights.append(0)
            x += 1
            bar_colors.append(alg_colors[alg])
            if alg not in used_labels:  
                bar_labels.append(alg)
                used_labels.add(alg)
            else:
                bar_labels.append(None)

        end_x = x - 1
        center = (start_x + end_x) / 2
        group_centers.append(center)

        if isinstance(param, str):
            pspl = param.split("p")
            if len(pspl) != 1:
                param = pspl[1].replace("0", "0.", 1)
        group_label = f"{PARAM_MAP[graph_family]}{param}"
        group_labels.append(group_label)

        x += group_gap  # gap between groups

    plt.figure(figsize=(max(12, len(x_positions) * 0.4), 6))
    plt.bar(x_positions, heights, label=bar_labels, color=bar_colors, width=bar_width)

    plt.xticks(group_centers, group_labels, rotation=0)
    if dstr == "ratios":
        plt.ylabel("Average ratio")
        if graph_size is not None:
            plt.title(f"Empirical Approximation Ratios for {graph_family} Graphs of Size {graph_size}")
        else:
            plt.title(f"Empirical Approximation Ratios for {graph_family} Graphs")
        # plt.ylim(0, 1.05)

    else: 
        plt.ylabel("Average time (seconds)")
        plt.yscale("log")

        if graph_size is not None:
            plt.title(f"Time to Approximate Max-3-Cut for {graph_family} Graphs of Size {graph_size}")
        else:
            plt.title(f"Time to Approximate Max-3-Cut for {graph_family} Graphs")

    plt.legend(title="Algorithm")

    plt.tight_layout()

    if graph_size is not None:
        out_path = Path(out_dir) / f"avg_{dstr}_{graph_family}_size_{graph_size}.jpg"
    else:
        out_path = Path(out_dir) / f"avg_{dstr}_{graph_family}.jpg"
    plt.savefig(out_path, dpi=300)
    plt.close()


def construct_charts(ratio_dicts, time_dicts, out_dir):
    stack = [(ratio_dicts, "ratios", ""), (time_dicts, "time", "")]
    while stack:
        current, dstr, name = stack.pop()

        if isinstance(current, dict):
            potential_dict = next(iter(current.values()))
            if isinstance(potential_dict, dict):
                child = next(iter(potential_dict.values()))
                if not isinstance(child, dict):
                    construct_one_chart(current, dstr, out_dir, name)
                else:
                    for new_name, v in current.items():
                        if name != "":
                            stack.append((v, dstr, f"{name} {new_name}"))
                        else:
                            stack.append((v, dstr, f"{new_name}"))
                    
def main():
    parser = argparse.ArgumentParser(
        description="Summarize data from experiments."
    )
    parser.add_argument("--in_dir", type=str, default=".")
    parser.add_argument("--f_scores_out", type=str, default="./scores.json")
    parser.add_argument("--f_times_out", type=str, default="times.json")
    parser.add_argument("--f_ratios_out", type=str, default="ratios.json")
    parser.add_argument("--f_avg_ratios_out", type=str, default="avg_ratios.json")
    parser.add_argument("--f_avg_times_out", type=str, default="avg_times.json")
    parser.add_argument("--out_dir", type=str, default="./summaries")
    
    args = parser.parse_args()
    in_dir = args.in_dir
    f_scores_out = args.f_scores_out
    f_times_out = args.f_times_out
    f_ratios_out = args.f_ratios_out
    f_avg_ratios_out = args.f_avg_ratios_out
    f_avg_times_out = args.f_avg_times_out
    out_dir = args.out_dir

    root = Path(in_dir) 

    scores_dicts, time_dicts = load_data(root, args)
    ratios_dict = compute_ratios(scores_dicts)
    avg_ratios, avg_times = calculate_averages(ratios_dict, time_dicts)

    scores_dict_json = recursive_defaultdict_to_dict(scores_dicts)
    time_dict_json  = recursive_defaultdict_to_dict(time_dicts)
    ratios_json_dict = recursive_defaultdict_to_dict(ratios_dict)
    avg_ratios_json = recursive_defaultdict_to_dict(avg_ratios)
    avg_times_json = recursive_defaultdict_to_dict(avg_times)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    construct_charts(avg_ratios_json, avg_times_json, out_dir)

    with open(out_dir_path / f_scores_out, "w", encoding="utf-8") as f:
        json.dump(scores_dict_json, f, indent=2)
    with open(out_dir_path / f_times_out, "w", encoding="utf-8") as f:
        json.dump(time_dict_json, f, indent=2)
    with open(out_dir_path / f_ratios_out, "w", encoding="utf-8") as f:
        json.dump(ratios_json_dict, f, indent=2)
    with open(out_dir_path / f_avg_ratios_out, "w", encoding="utf-8") as f:
        json.dump(avg_ratios_json, f, indent=2)
    with open(out_dir_path / f_avg_times_out, "w", encoding="utf-8") as f:
        json.dump(avg_times_json, f, indent=2)

if __name__ == "__main__":
    main()