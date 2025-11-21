import os
import socket
from collections import Counter

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def main():
    print("Connecting with ray.init(address='auto')", flush=True)
    ray.init(address="auto")

    # List nodes in the cluster
    nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    print("\nAlive nodes:")
    for n in nodes:
        print(f"  NodeID={n['NodeID']}  Hostname={n['NodeManagerAddress']}")
    print(flush=True)

    if len(nodes) != 2:
        raise RuntimeError(f"Expected exactly 2 nodes, but found {len(nodes)}")

    node0_id = nodes[0]["NodeID"]
    node1_id = nodes[1]["NodeID"]
    node0_host = nodes[0]["NodeManagerAddress"]
    node1_host = nodes[1]["NodeManagerAddress"]

    print(f"\nNode 0: {node0_id} (host {node0_host})")
    print(f"Node 1: {node1_id} (host {node1_host})\n", flush=True)

    @ray.remote
    def worker_info(i):
        ctx = ray.get_runtime_context()
        return {
            "task_id": i,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "node_id": ctx.get_node_id(),
        }

    num_tasks = 20
    print(f"Launching {num_tasks} tasks", flush=True)

    half = num_tasks // 2
    futures = []

    # First half pinned to node 0
    for i in range(half):
        futures.append(
            worker_info.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node0_id,
                    soft=False,
                )
            ).remote(i)
        )

    # Second half pinned to node 1
    for i in range(half, num_tasks):
        futures.append(
            worker_info.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node1_id,
                    soft=False,
                )
            ).remote(i)
        )

    results = ray.get(futures)

    # Print each task location
    for r in sorted(results, key=lambda x: x["task_id"]):
        print(
            f"task {r['task_id']:2d}  host={r['hostname']}  pid={r['pid']}  node_id={r['node_id']}"
        )

    # Count by hostname and node_id
    host_counts = Counter(r["hostname"] for r in results)
    node_counts = Counter(r["node_id"] for r in results)

    print("\nTask distribution by hostname:")
    for host, cnt in host_counts.items():
        print(f"  {host}: {cnt} tasks")

    print("\nTask distribution by node_id:")
    for nid, cnt in node_counts.items():
        print(f"  {nid}: {cnt} tasks")

    print("\nExpected split:")
    print(f"  Node 0 ({node0_host}) should have {half} tasks")
    print(f"  Node 1 ({node1_host}) should have {num_tasks - half} tasks")

    assert node_counts[node0_id] == half, "Node 0 did not get the expected number of tasks"
    assert node_counts[node1_id] == num_tasks - half, "Node 1 did not get the expected number of tasks"

    print("\nExact half half split confirmed.")


if __name__ == "__main__":
    main()
