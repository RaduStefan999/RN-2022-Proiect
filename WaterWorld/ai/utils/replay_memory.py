import numpy as np
from collections import deque


class MemoryManager:
    def __init__(self, min_capacity, max_capacity, bucket_size):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.history_sars = deque(maxlen=max_capacity)
        self.bucket_size = bucket_size

    def add(self, tuple_sars):
        """add a sars in the history so we can learn in the future"""
        self.history_sars.append(tuple_sars)

    def enough_elements_to_learn(self) -> bool:
        """verify if there is at least min_capacity in our bucket,
        it should be higher than buckez_size so we get diversity"""

        return len(self.history_sars) >= self.min_capacity

    def get_bucket(self):
        """returns bucket_size elements when called"""
        size_history_elements = len(self.history_sars)

        if size_history_elements < self.bucket_size:
            raise Exception("Pare rau nenea, nu avem elemente in buffer istorie pentru tine")

        randomised_selection = np.random.choice(size_history_elements, size=self.bucket_size, replace=False)
        selected_for_bucket = []
        for selected in randomised_selection:
            selected_for_bucket.append(self.history_sars[selected])

        return np.array(selected_for_bucket)

    def get_random_element_history(self):
        """returns exactly one element"""
        size_history_elements = len(self.history_sars)
        return self.history_sars[np.random.randint(size_history_elements)] # the random element from history