import numpy as np

from PIL import Image

def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def check_win(win):
    '''
    Input:
    	win: window for current node
    Output:
    	val: value of current window
    '''
    if win.min() == win.max():
        return win.min()
    else:
        return float(-1)


def node_val(raw_map, given_level, total_levels, key):
    '''
    Input:
    	raw_map: original segmentation map
    	level: depth level of segmentation map
    	key: hash-table key (x,y)
    Output:
    	val: value of current node
    '''
    x = key[0]
    y = key[1]
    im_x = raw_map.shape[0]
    im_y = raw_map.shape[1]
    win_size = np.power(2, (total_levels - given_level))
    win_x_min = int(max(x * win_size, 0))
    win_y_min = int(max(y * win_size, 0))
    win_x_max = int(min((x + 1) * win_size, im_x))
    win_y_max = int(min((y + 1) * win_size, im_y))
    win = raw_map[win_x_min:win_x_max, win_y_min:win_y_max]
    val = check_win(win)
    return val


def cur_key_to_last_key(key):
    '''
    Input:
    	key: (x,y) cordinates of current pixel
    Output:
    	last_key: (x,y) cordinates of super-pixel for current pixel
    '''
    last_key = (int(key[0] / 2), int(key[1] / 2))
    return last_key


def node_val_inter(raw_map, given_level, total_levels, key, last_map, return_255):
    '''
    Input:
            raw_map: original segmentation map
            level: depth level of segmentation map
            key: hash-table key (x,y)
    Output:
            val: value of current node
    '''
    last_key = cur_key_to_last_key(key)
    last_x = last_key[0]
    last_y = last_key[1]
    last_val = last_map[last_x, last_y]
    if last_val >= 0:
        if return_255:
            return 255
        else:
            return last_val
    x = key[0]
    y = key[1]
    im_x = raw_map.shape[0]
    im_y = raw_map.shape[1]
    win_size = np.power(2, (total_levels - given_level))
    win_x_min = int(max(x * win_size, 0))
    win_y_min = int(max(y * win_size, 0))
    win_x_max = int(min((x + 1) * win_size, im_x))
    win_y_max = int(min((y + 1) * win_size, im_y))
    win = raw_map[win_x_min:win_x_max, win_y_min:win_y_max]
    val = check_win(win)
    return val


def depth_sub(previous_map, current_map):
    '''
    Input:
        previous_map: seg map of last layer
        current_map: seg map of current layer
    Output:
        current_map: current_map - scaled(previous_map)
    '''
    current_map_x = current_map.shape[0]
    current_map_y = current_map.shape[1]
    previous_map = Image.fromarray(np.array(previous_map, dtype=np.uint8)).resize(size=(current_map_x, current_map_y),
                                                                                  resample=Image.NEAREST)
    previous_map = np.array(previous_map)
    return current_map - previous_map


def dense2quad(raw_map, num_levels=6, return_255=False):
    '''
    raw_map: input is raw segmentation map
    out_map: output is quadtree output representation
    '''
    size_x = raw_map.shape[0]
    size_y = raw_map.shape[1]

    init_res_x = int(size_x / np.power(2, num_levels - 1))
    init_res_y = int(size_y / np.power(2, num_levels - 1))

    out_map = {}
    for given_level in range(1, num_levels + 1):
        level_res_x = init_res_x * np.power(2, given_level - 1)
        level_res_y = init_res_y * np.power(2, given_level - 1)
        level_map = np.zeros((level_res_x, level_res_y), dtype=np.float32)
        for x in range(0, level_res_x):
            for y in range(0, level_res_y):
                if given_level == 1:
                    level_map[x, y] = node_val(raw_map, given_level, num_levels, (x, y))
                else:
                    level_map[x, y] = node_val_inter(raw_map, given_level, num_levels, (x, y), out_map[given_level - 1],
                                                     return_255)
        out_map[given_level] = level_map
    return out_map