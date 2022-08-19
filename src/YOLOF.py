'''
Selecting Outstanding Frames within GOP size is 16
Selecting by frame indexing
The case is divided to 2:
    1) 2N+1 == 3
    2) 2N+1 == 5
'''

#1) 2N+1 == 3
def get_YOLOF_idx(idx_frame, step):
    left_frame = 0
    right_frame = 0

    i = idx_frame // 8
    p = idx_frame % 8

    left_I = 0 + i * 8
    right_I = left_I + 8

    is_b = idx_frame % 2

    if step == 2:
        if is_b == 1:
            left_frame = idx_frame - 1
            right_frame = idx_frame + 1

        elif is_b == 0:
            left_frame = idx_frame - 2
            right_frame = idx_frame + 2

    if step == 4:
        if p < 4 and p!=0:
            left_frame = left_I
            right_frame = left_I + 4

        elif p == 4:
            left_frame = left_I
            right_frame = right_I

        elif p == 0:
            left_frame = left_I - 4
            right_frame = right_I - 4

        else:
            left_frame = right_I - 4
            right_frame = right_I

    return left_frame, right_frame

#2) 2N+1 == 5
def get_YOLOF_idx(center_frame_idx):
    cur = center_frame_idx
    i = center_frame_idx // 8
    left_i = 0 + i * 8

    if (center_frame_idx - left_i) % 2 == 1:
        neighbor_list = [cur - 3, cur - 1, cur, cur + 1, cur + 3]

    else:
        neighbor_list = [cur - 4, cur - 2, cur, cur + 2, cur + 4]

    return neighbor_list
