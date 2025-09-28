from simulator import Simulator
from algorithms import FIFO, LRU, MarkovPredictive
import numpy as np



def load_real_trace(filepath: str, max_lines: int = 100000) -> list[int]:
    """
    Loads and parses a real-world MSR Cambridge trace file.
    """
    global PAGE_SIZE
    page_trace = []
    print(f"Loading and parsing real trace file: {filepath}...")
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                # We'll just read the first N lines to keep it fast
                if i >= max_lines:
                    break
                try:
                    parts = line.strip().split(',')
                    # The address is the 5th column (index 4)
                    address = int(parts[4])
                    page_trace.append(address // PAGE_SIZE)
                except (ValueError, IndexError):
                    # Skip any lines that are malformed
                    continue
        print(f"Trace loading complete. Loaded {len(page_trace)} accesses.")
        return page_trace
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
    TRACE_FILE = "traces/web_0.csv"
    page_trace = load_real_trace(TRACE_FILE)

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