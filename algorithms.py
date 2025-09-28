from collections import deque
from abc import ABC, abstractmethod
import numpy as np

class Algorithm(ABC):
    """Abstract base class for all page replacement algorithms."""
    def __init__(self, num_frames: int):
        self.num_frames = num_frames

    @abstractmethod
    def choose_victim(self, frames: list) -> int: # <-- Added 'frames' parameter
        """Chooses a page to evict when memory is full. Returns victim page number."""
        pass
    
    @abstractmethod
    def page_loaded(self, page_num: int):
        """Notifies the algorithm that a new page has been loaded into memory."""
        pass

    def page_accessed(self, page_num: int):
        """Notifies the algorithm that a page in memory has been accessed (a hit)."""
        pass

class FIFO(Algorithm):
    def __init__(self, num_frames: int):
        super().__init__(num_frames)
        self.queue = deque()

    def choose_victim(self, frames: list) -> int:
        return self.queue.popleft()

    def page_loaded(self, page_num: int):
        self.queue.append(page_num)

class LRU(Algorithm):
    def __init__(self, num_frames: int):
        super().__init__(num_frames)
        self.page_order: list[int] = []

    def choose_victim(self, frames: list) -> int:
        return self.page_order.pop(0)

    def page_loaded(self, page_num: int):
        self.page_order.append(page_num)

    def page_accessed(self, page_num: int):
        if page_num in self.page_order:
            self.page_order.remove(page_num)
            self.page_order.append(page_num)

class MarkovPredictive(Algorithm):
    """A predictive algorithm based on a Markov chain model."""
    def __init__(self, num_frames: int, all_pages: set):
        super().__init__(num_frames)
        self.all_pages = sorted(list(all_pages))
        self.page_to_index = {page: i for i, page in enumerate(self.all_pages)}
        num_unique_pages = len(self.all_pages)
        self.transition_matrix = np.zeros((num_unique_pages, num_unique_pages))
        self.last_accessed_page = -1

    def train(self, trace: list[int]):
        print("Training Markov model...")
        for i in range(len(trace) - 1):
            current_page = trace[i]
            next_page = trace[i+1]
            if current_page in self.page_to_index and next_page in self.page_to_index:
                current_idx = self.page_to_index[current_page]
                next_idx = self.page_to_index[next_page]
                self.transition_matrix[current_idx][next_idx] += 1
        for i in range(len(self.transition_matrix)):
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                self.transition_matrix[i] /= row_sum
        print("Training complete.")
    
    # --- METHOD RENAMED AND REFACTORED ---
    def choose_victim(self, frames: list) -> int:
        pages_in_memory = [p for p in frames if p is not None]
        
        if self.last_accessed_page != -1:
            last_page_idx = self.page_to_index.get(self.last_accessed_page)
            min_prob = float('inf')
            victim_page = -1

            for mem_page in pages_in_memory:
                mem_page_idx = self.page_to_index.get(mem_page)
                prob = self.transition_matrix[last_page_idx][mem_page_idx]
                if prob < min_prob:
                    min_prob = prob
                    victim_page = mem_page
            
            if victim_page != -1:
                return victim_page
        
        # Fallback: if no prediction can be made, just evict the first page found
        return pages_in_memory[0]

    # --- NEW METHOD ADDED ---
    def page_loaded(self, page_num: int):
        self.last_accessed_page = page_num
    
    def page_accessed(self, page_num: int):
        self.last_accessed_page = page_num