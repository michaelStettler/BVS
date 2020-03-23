def unflatten_idx_scalar(idx, size):
    if len(size) == 1:
        return idx

    elif len(size) == 2:
        row = int(idx / size[1])
        col = idx - row * size[1]
        return (row, col)

    elif len(size) == 3:
        size_plane = size[1] * size[2]
        row = int(idx / size_plane)
        plane_idx = idx - row * size_plane
        col = int(plane_idx / size[2])
        depth = plane_idx - col * size[2]
        return (row, col, depth)
    else:
        raise NotImplementedError


def unflatten_idx(idx, size):
    """
    [0  1  2  3]
    [4  5  6  7]
    [8  9 10 11]

    [1 2 3 4 5 6 7 8 9 10 11]

    8 -> (2, 0)

    :param idx:
    :param size:
    :return:
    """

    if isinstance(idx, list):
        list_u_idx = []
        for i in idx:
            u_idx = unflatten_idx_scalar(i, size)
            list_u_idx.append(u_idx)
        return list_u_idx
    else:
        return unflatten_idx_scalar(idx, size)


if __name__ == '__main__':
    import numpy as np

    test1 = np.arange(12)
    test1 = np.reshape(test1, (3, 4))
    print(test1)
    idx1 = 8
    u_idx1 = unflatten_idx(idx1, np.shape(test1))
    print("idx 1:", idx1, "unflatten:", u_idx1, "val at pos:", test1[u_idx1])

    test2 = np.arange(60)
    test2 = np.reshape(test2, (3, 4, 5))
    print(test2)
    idx2 = 49
    u_idx2 = unflatten_idx(idx2, np.shape(test2))
    print(test2[0, 2])
    print("idx 2:", idx2, "unflatten:", u_idx2, "val at pos:", test2[u_idx2])