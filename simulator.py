from typing import Optional, Dict, List
from algorithms import Algorithm # <-- IMPORT CHANGE

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
    """The main simulator engine."""
    # --- METHOD UPDATED ---
    def __init__(self, num_frames: int, page_size: int, algorithm: Algorithm):
        self.num_frames = num_frames
        self.page_size = page_size
        self.algorithm = algorithm # <-- OBJECT INSTEAD OF STRING
        self.memory = Memory(self.num_frames)
        self.page_table = PageTable()
        self.stats = {'hits': 0, 'misses': 0, 'total_accesses': 0}
        print(f"Initialized simulator with {num_frames} frames, page size {page_size}.")

    # --- METHOD UPDATED ---
    def run(self, page_trace: List[int]):
        """Run the simulation on a given memory trace."""
        print(f"Starting simulation with {self.algorithm.__class__.__name__} algorithm...")

        for virtual_page_num in page_trace:
            self.stats['total_accesses'] += 1
            
            frame = self.page_table.lookup(virtual_page_num)

            if frame is not None:
                # Page Hit
                self.stats['hits'] += 1
                self.algorithm.page_accessed(virtual_page_num)
            else:
                # Page Fault (Miss)
                self.stats['misses'] += 1
                
                if not self.memory.is_full():
                    # There is free space in memory
                    free_frame = self.memory.get_free_frame()
                    self.memory.load_page(virtual_page_num, free_frame)
                    self.page_table.update(virtual_page_num, free_frame)
                    self.algorithm.page_fault(virtual_page_num, self.memory.frames)
                else:
                    # Memory is full, need to evict a page
                    victim_page = self.algorithm.page_fault(virtual_page_num, self.memory.frames)
                    
                    # Evict the victim page
                    evicted_frame = self.memory.evict_page(victim_page)
                    self.page_table.evict(victim_page)
                    
                    # Load the new page
                    self.memory.load_page(virtual_page_num, evicted_frame)
                    self.page_table.update(virtual_page_num, evicted_frame)
        
        print("Simulation finished.")
        hit_rate = (self.stats['hits'] / self.stats['total_accesses']) * 100
        print(f"--- Results for {self.algorithm.__class__.__name__} ---")
        print(f"Total Accesses: {self.stats['total_accesses']}")
        print(f"Page Hits:      {self.stats['hits']}")
        print(f"Page Faults:    {self.stats['misses']}")
        print(f"Hit Rate:         {hit_rate:.2f}%")