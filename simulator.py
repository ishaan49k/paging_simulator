from typing import Optional, Dict, List
from algorithms import Algorithm
from collections import OrderedDict




class TLB:
    """A Translation Lookaside Buffer (TLB) implemented with an LRU cache."""
    def __init__(self, size: int):
        self.size = size
        self.cache = OrderedDict()

    def lookup(self, virtual_page_num: int) -> int | None:
        """
        Look for a page in the TLB. If found (a hit), move it to the
        end to mark it as most recently used.
        """
        if virtual_page_num in self.cache:
            # TLB Hit: Move to end to signify it's the most recently used
            self.cache.move_to_end(virtual_page_num)
            return self.cache[virtual_page_num]
        return None # TLB Miss

    def add(self, virtual_page_num: int, frame_num: int):
        """Add a new entry to the TLB, evicting the LRU entry if full."""
        self.cache[virtual_page_num] = frame_num
        self.cache.move_to_end(virtual_page_num)
        if len(self.cache) > self.size:
            self.cache.popitem(last=False) # Pop the least recently used item





class PageTable:
    # (This class remains unchanged from the previous step)
    def __init__(self):
        self.mapping: Dict[int, int] = {}
    def lookup(self, virtual_page_num: int) -> Optional[int]:
        return self.mapping.get(virtual_page_num)
    def update(self, virtual_page_num: int, frame_num: int):
        self.mapping[virtual_page_num] = frame_num
    def evict(self, virtual_page_num: int):
        if virtual_page_num in self.mapping:
            del self.mapping[virtual_page_num]

class Memory:
    # (This class remains unchanged from the previous step)
    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        self.frames: List[Optional[int]] = [None] * num_frames
        self.free_frames: List[int] = list(range(num_frames))
        # We need a reverse map to find which frame a page is in
        self.page_to_frame: Dict[int, int] = {}

    def is_full(self) -> bool:
        return not self.free_frames

    def get_free_frame(self) -> Optional[int]:
        return self.free_frames.pop(0) if self.free_frames else None

    def load_page(self, virtual_page_num: int, frame_num: int):
        self.frames[frame_num] = virtual_page_num
        self.page_to_frame[virtual_page_num] = frame_num

    def evict_page(self, virtual_page_num: int):
        frame_num = self.page_to_frame.pop(virtual_page_num)
        self.frames[frame_num] = None
        self.free_frames.append(frame_num) # This frame is now free
        return frame_num

class Simulator:
    """The main simulator engine, now with a TLB."""
    def __init__(self, num_frames: int, page_size: int, tlb_size: int, algorithm: 'Algorithm'):
        self.num_frames = num_frames
        self.page_size = page_size
        self.algorithm = algorithm
        self.memory = Memory(self.num_frames)
        self.page_table = PageTable()
        self.tlb = TLB(size=tlb_size)
        
        # Expanded stats dictionary
        self.stats = {
            'total_accesses': 0, 'page_hits': 0, 'page_faults': 0,
            'tlb_hits': 0, 'tlb_misses': 0
        }
        print(f"Initialized simulator with {num_frames} frames, TLB size {tlb_size}.")

    def run(self, page_trace: list[int]):
        """Run the simulation with the corrected logic."""
        print(f"Starting simulation with {self.algorithm.__class__.__name__} algorithm...")
    
        for virtual_page_num in page_trace:
            self.stats['total_accesses'] += 1
            
            frame = self.tlb.lookup(virtual_page_num)

            if frame is not None:
                # TLB Hit
                self.stats['tlb_hits'] += 1
                self.stats['page_hits'] += 1
                self.algorithm.page_accessed(virtual_page_num)
            else:
                # TLB Miss
                self.stats['tlb_misses'] += 1
                
                frame = self.page_table.lookup(virtual_page_num)
                
                if frame is not None:
                    # TLB Miss, but Page Hit
                    self.stats['page_hits'] += 1
                    self.algorithm.page_accessed(virtual_page_num)
                    self.tlb.add(virtual_page_num, frame)
                else:
                    # Page Fault
                    self.stats['page_faults'] += 1
                    
                    if self.memory.is_full():
                        # Eviction is necessary
                        victim_page = self.algorithm.choose_victim(self.memory.frames)
                        frame = self.memory.evict_page(victim_page)
                        self.page_table.evict(victim_page)
                    else:
                        # There is free space
                        frame = self.memory.get_free_frame()

                    # Load the new page
                    self.memory.load_page(virtual_page_num, frame)
                    self.page_table.update(virtual_page_num, frame)
                    self.tlb.add(virtual_page_num, frame)
                    self.algorithm.page_loaded(virtual_page_num)

        print("Simulation finished.")
        self.print_results()

    def print_results(self):
        tlb_hit_rate = (self.stats['tlb_hits'] / self.stats['total_accesses']) * 100
        page_hit_rate = (self.stats['page_hits'] / self.stats['total_accesses']) * 100
        
        print(f"--- Results for {self.algorithm.__class__.__name__} ---")
        print(f"Total Accesses: {self.stats['total_accesses']}")
        print(f"TLB Hits:       {self.stats['tlb_hits']}")
        print(f"TLB Misses:     {self.stats['tlb_misses']}")
        print(f"Page Faults:    {self.stats['page_faults']}")
        print(f"TLB Hit Rate:     {tlb_hit_rate:.2f}%")
        print(f"Page Hit Rate:    {page_hit_rate:.2f}%")