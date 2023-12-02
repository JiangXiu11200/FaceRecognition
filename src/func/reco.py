import numpy as np

def easy_blink_detect(blink_list):
    state = False
    if 0 in blink_list and np.count_nonzero(blink_list == 0) >= 3 and np.count_nonzero(blink_list == 1) >= 3 and blink_list[0] != 0 and blink_list[-1] != 0:
        first_zero_index = np.argmax(blink_list == 0)
        last_zero_index = len(blink_list) - 1 - np.argmax(np.flip(blink_list) == 0)
        if (last_zero_index - first_zero_index + 1) == np.count_nonzero(blink_list == 0):
            state = True
    else:
        state = False
    return state