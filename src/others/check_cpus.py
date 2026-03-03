import ray

def main():
    # Connect to existing Ray cluster
    ray.init(address="auto")

    resources = ray.cluster_resources()
    print("Cluster resources reported by Ray:")
    print(resources)

    if "CPU" in resources:
        print(f"Total CPUs available: {resources['CPU']}")
    else:
        print("Ray does not detect any CPU resources.")

if __name__ == "__main__":
    main()
