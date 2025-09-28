from simulator import Simulator
from algorithms import FIFO, LRU, MarkovPredictive
import numpy as np

def load_trace(filepath: str) -> list[int]:
    """Loads a memory trace file into a list of integers."""
    try:
        with open(filepath, 'r') as f:
            # Convert addresses to page numbers right away
            return [int(line.strip()) // PAGE_SIZE for line in f]
    except FileNotFoundError:
        print(f"Error: Trace file not found at {filepath}")
        return []

def main():
    """Main function to run the simulator."""
    # --- Configuration ---
    global PAGE_SIZE
    NUM_FRAMES = 50
    PAGE_SIZE = 100
    TLB_SIZE = 16  # <-- ADD TLB CONFIGURATION
    TRACE_FILE = "traces/realistic_trace.txt"

    # --- Setup and Run ---
    page_trace = load_trace(TRACE_FILE)
    if not page_trace:
        return

    unique_pages = set(page_trace)

    algorithms_to_test = [
        FIFO(num_frames=NUM_FRAMES),
        LRU(num_frames=NUM_FRAMES),
        MarkovPredictive(num_frames=NUM_FRAMES, all_pages=unique_pages)
    ]

    print("--- Starting Simulation Comparison ---")
    for algorithm in algorithms_to_test:
        sim = Simulator(
            num_frames=NUM_FRAMES, 
            page_size=PAGE_SIZE, 
            tlb_size=TLB_SIZE,  # <-- PASS TLB SIZE
            algorithm=algorithm
        )

        if isinstance(algorithm, MarkovPredictive):
            algorithm.train(page_trace)

        sim.run(page_trace)
        print("-" * 40)


if __name__ == "__main__":
    main()