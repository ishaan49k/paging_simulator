# In generate_trace.py

import random

def generate_trace(
    filename="traces/realistic_trace.txt",
    num_accesses=5000,
    num_pages=100,
    working_set_size=10,
    working_set_shift_freq=500,
    locality_prob=0.85
):
    """
    Generates a memory trace file with locality of reference.
    """
    print(f"Generating trace file at '{filename}'...")
    
    working_set_start = 0
    trace = []

    for i in range(num_accesses):
        # Periodically shift the working set
        if i > 0 and i % working_set_shift_freq == 0:
            working_set_start = random.randint(0, num_pages - working_set_size)
            print(f"  ... Shifting working set to start at page {working_set_start}")

        # Decide if we access from the working set or randomly
        if random.random() < locality_prob:
            # Access within the working set
            page_offset = random.randint(0, working_set_size - 1)
            page_num = working_set_start + page_offset
        else:
            # Random access across all pages
            page_num = random.randint(0, num_pages - 1)
        
        # Convert page number to a memory address (using our 100 page size)
        address = page_num * 100 + random.randint(0, 99)
        trace.append(address)

    with open(filename, 'w') as f:
        for addr in trace:
            f.write(f"{addr}\n")
    
    print(f"Successfully generated {num_accesses} memory accesses.")

if __name__ == "__main__":
    generate_trace()