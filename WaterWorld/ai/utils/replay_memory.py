import numpy as np
import copy
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

        print("We have the len before selection :  ", len(self.history_sars))

        randomised_selection = np.random.choice(size_history_elements, size=self.bucket_size, replace=False)
        randomised_selection = np.sort(randomised_selection)[::-1]
        selected_for_bucket = []
        shallow_copy = copy.deepcopy(self.history_sars)
        for selected in randomised_selection:
            selected_for_bucket.append(self.history_sars[selected])
            del shallow_copy[selected]
            # print("Deleted something and we have the size : ",len(shallow_copy))
        self.history_sars = shallow_copy
        # print("We have the len after selection :  ", len(self.history_sars))

        return np.array(selected_for_bucket)

    def get_random_element_history(self):
        """returns exactly one element"""
        size_history_elements = len(self.history_sars)

        wanted_position = np.random.randint(size_history_elements)
        wanted_item = self.history_sars[wanted_position]

        del self.history_sars[wanted_position]

        return wanted_item # the random element from history