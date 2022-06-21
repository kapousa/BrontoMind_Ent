#  Copyright (c) 2022. Slonos Labs. All rights Reserved.
import random
from calendar import month_name
from datetime import date
import numpy as np

import numpy


class Helper:
    test_value = ''

    def __init__(self):
        self.test_value = '_'

    def generate_unique_random_nlist(self, start_index, end_index, length):
        # Generate n unique random numbers within a range
        num_list = random.sample(range(start_index, end_index), length)
        print(num_list)

        return numpy.array(num_list)

    def previous_n_months(n):
        current_month_idx = date.today().month - 1  # Value is now (0-11)
        months_list =[]
        for i in range(1, n + 1):
            # The mod operator will wrap the negative index back to the positive one
            previous_month_idx = (current_month_idx - i) % 12  # (0-11 scale)
            m = int(previous_month_idx + 1)
            months_list.append(month_name[m])
        return np.flip(months_list)
