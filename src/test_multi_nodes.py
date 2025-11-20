import os
import socket
from collections import Counter
import ray


def main():
    print("Connecting with ray.init(address='auto')", flush=True)
    ray.init(address="auto")

    print("\nCluster resources:")
    print(ray.cluster_resources(), flush=True)

    @ray.remote
    def worker_info(i):
        import os as _os
        import socket as _socket
        import ray as _ray
        ctx = _ray.get_runtime_context()
        return {
            "task_id": i,
            "pid": _os.getpid(),
            "hostname": _socket.gethostname(),
            "node_id": ctx.node_id.hex(),
        }

    num_tasks = 300
    print(f"\nLaunching {num_tasks} tasks\n", flush=True)
    futures = [worker_info.remote(i) for i in range(num_tasks)]
    results = ray.get(futures)

    for r in results:
        print(f"task {r['task_id']:2d}  host={r['hostname']}  pid={r['pid']}")

    host_counts = Counter(r["hostname"] for r in results)
    print("\nTask distribution by host:")
    for host, cnt in host_counts.items():
        print(f"{host}: {cnt} tasks")

    if len(host_counts) >= 2:
        print("\nMulti node behavior confirmed.")
    else:
        print("\nWarning: tasks only ran on one node.")


if __name__ == "__main__":
    main()
