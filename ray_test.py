import ray 
ray.init()
print(ray.available_resources())
import time

database = [
    "Learning", "Ray",
    "Flexible", "Distributed", "Python", "for", "Machine", "Learning"
]


def retrieve(item):
    time.sleep(item / 10.)
    return item, database[item]

# ray task = a function that ray executes on a different process from 
# where it was called, possibly on different machine

# intended use: func.remote()
@ray.remote 
def retrieve_task(item):
    return retrieve(item)

def print_runtime(input_data, start_time):
    print(f'Runtime: {time.time() - start_time:.2f} seconds, data:')
    print(*input_data, sep="\n")
    
start = time.time()
object_references = [
    retrieve_task.remote(item) for item in range(8)
]
data = ray.get(object_references)
print_runtime(data, start)