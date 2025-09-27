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
    TRACE_FILE = "traces/realistic_trace.txt"

    # --- Setup and Run ---
    page_trace = load_trace(TRACE_FILE)
    if not page_trace:
        return

    # The Markov model needs to know all unique pages in advance
    unique_pages = set(page_trace)

    # --- ALGORITHM COMPARISON ---
    algorithms_to_test = [
        FIFO(num_frames=NUM_FRAMES),
        LRU(num_frames=NUM_FRAMES),
        MarkovPredictive(num_frames=NUM_FRAMES, all_pages=unique_pages)
    ]

    print("--- Starting Simulation Comparison ---")
    for algorithm in algorithms_to_test:
        # We create a new simulator for each run to ensure a clean state
        sim = Simulator(
            num_frames=NUM_FRAMES,
            # Page size is now handled during trace loading
            page_size=PAGE_SIZE,
            algorithm=algorithm
        )

        # --- SPECIAL STEP: Train the predictive model ---
        if isinstance(algorithm, MarkovPredictive):
            algorithm.train(page_trace)

        sim.run(page_trace) # Pass the page trace directly
        print("-" * 40)


if __name__ == "__main__":
    main()