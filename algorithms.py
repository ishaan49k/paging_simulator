from collections import deque
from abc import ABC, abstractmethod

import numpy as np

class Algorithm(ABC):
    """Abstract base class for all page replacement algorithms."""
    def __init__(self, num_frames: int):
        self.num_frames = num_frames

    @abstractmethod
    def page_fault(self, page_num: int, frames: list) -> int:
        """
        Handles a page fault. Must be implemented by subclasses.
        Returns the virtual page number of the page to evict.
        """
        pass

    def page_accessed(self, page_num: int):
        """Called when a page is accessed (hit). Can be used by LRU, etc."""
        pass



class FIFO(Algorithm):
    """First-In, First-Out page replacement algorithm."""
    def __init__(self, num_frames: int):
        super().__init__(num_frames)
        self.queue = deque()

    def page_fault(self, page_num: int, frames: list) -> int:
        """
        On a page fault, if memory is full, the first page added to the
        queue is the victim. The new page is added to the back of the queue.
        """
        victim_page = -1 # No victim if there's free space

        if len(self.queue) >= self.num_frames:
            victim_page = self.queue.popleft() # Evict from the front
        
        self.queue.append(page_num) # Add new page to the back
        return victim_page
    

# --- LRU ---
class LRU(Algorithm):
    """Least Recently Used page replacement algorithm."""
    def __init__(self, num_frames: int):
        super().__init__(num_frames)
        self.page_order: list[int] = []

    def page_accessed(self, page_num: int):
        """When a page is accessed, move it to the end of the list (most recent)."""
        if page_num in self.page_order:
            self.page_order.remove(page_num)
        self.page_order.append(page_num)

    def page_fault(self, page_num: int, frames: list) -> int:
        """
        On a page fault, if memory is full, the least recently used page
        (at the front of the list) is the victim.
        """
        victim_page = -1
        if len(self.page_order) >= self.num_frames:
            victim_page = self.page_order.pop(0) # Evict from the front (LRU)
        
        self.page_accessed(page_num) # Treat the new page as most recently used
        return victim_page
    




# --- Markov chain algorithm ---
class MarkovPredictive(Algorithm):
    """A predictive algorithm based on a Markov chain model."""
    def __init__(self, num_frames: int, all_pages: set):
        super().__init__(num_frames)
        self.all_pages = sorted(list(all_pages))
        self.page_to_index = {page: i for i, page in enumerate(self.all_pages)}
        num_unique_pages = len(self.all_pages)
        
        # Transition matrix: T[i][j] = probability of going from page i to page j
        self.transition_matrix = np.zeros((num_unique_pages, num_unique_pages))
        self.last_accessed_page = -1

    def train(self, trace: list[int]):
        """Train the model by building the transition count matrix."""
        print("Training Markov model...")
        for i in range(len(trace) - 1):
            current_page = trace[i]
            next_page = trace[i+1]
            
            if current_page in self.page_to_index and next_page in self.page_to_index:
                current_idx = self.page_to_index[current_page]
                next_idx = self.page_to_index[next_page]
                self.transition_matrix[current_idx][next_idx] += 1
        
        # Normalize rows to get probabilities
        for i in range(len(self.transition_matrix)):
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                self.transition_matrix[i] /= row_sum
        print("Training complete.")

    def page_accessed(self, page_num: int):
        """Update the last accessed page."""
        self.last_accessed_page = page_num

    def page_fault(self, page_num: int, frames: list) -> int:
        """
        On a page fault, predict the least likely page to be used next
        and evict it.
        """
        victim_page = -1
        
        # Get a list of pages currently in memory
        pages_in_memory = [p for p in frames if p is not None]

        if len(pages_in_memory) >= self.num_frames:
            if self.last_accessed_page != -1:
                last_page_idx = self.page_to_index.get(self.last_accessed_page)
                
                min_prob = float('inf')
                victim_page = -1

                # Find the page in memory with the lowest probability of being accessed next
                for mem_page in pages_in_memory:
                    mem_page_idx = self.page_to_index.get(mem_page)
                    prob = self.transition_matrix[last_page_idx][mem_page_idx]
                    
                    if prob < min_prob:
                        min_prob = prob
                        victim_page = mem_page
                
                # If all have 0 probability, fall back to evicting the first one
                if victim_page == -1:
                    victim_page = pages_in_memory[0]
            else:
                # If we have no history, just evict the first page in memory
                victim_page = pages_in_memory[0]

        self.page_accessed(page_num)
        return victim_page