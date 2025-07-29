import multiprocessing
from job_queue import run_worker

NUM_WORKERS = 3  # Or however many you want

def start_worker(index):
    print(f"[Worker {index}] starting...")
    run_worker()

if __name__ == "__main__":
    processes = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(target=start_worker, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
