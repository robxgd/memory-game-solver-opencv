crayon_2 = [[1, 2, 3, 2, 4, 5, 6],
            [7, 8, 7, 9, 3, 1, 8],
            [6, 10, 11, 4, 9, 12, 5],
            [13, 10, 14, 11, 14, 12, 13]]

disney_1 = [[1, 2, 3, 1, 4, 4],
            [5, 6, 3, 7, 8, 8],
            [9, 10, 11, 6, 9, 11],
            [5, 10, 7, 12, 2, 12]]

strand_1 = [[1, 2, 3, 4],
            [5, 4, 6, 7],
            [8, 2, 8, 6],
            [3, 7, 5, 1]]

strand_2 = [[1, 2, 3, 4],
            [3, 4, 5, 2],
            [6, 7, 6, 8],
            [1, 5, 8, 7]]

vdk_1 = [[1, 2, 3, 4, 5, 6],
         [7, 8, 9, 7, 10, 4],
         [10, 11, 1, 12, 3, 11],
         [12, 2, 8, 9, 5, 6]]

vdk_2 = [[1, 2, 3, 4, 5, 6],
         [7, 3, 8, 9, 6, 1],
         [9, 10, 11, 12, 4, 8],
         [12, 5, 7, 11, 10, 2]]

winnie_1 = [[1, 2, 3, 4, 1, 5],
            [6, 7, 8, 8, 9, 10],
            [11, 4, 12, 9, 5, 3],
            [6, 10, 2, 12, 7, 11]]

winnie_2 = [[1, 2, 3, 4, 5, 6],
            [7, 5, 8, 3, 4, 9],
            [1, 2, 6, 10, 11, 12],
            [7, 8, 9, 10, 12, 11]]


def solution(game):
    if game.startswith('Crayon_2'):
        return crayon_2
    elif game.startswith('Disney_1'):
        return disney_1
    elif game.startswith('Strand_1'):
        return strand_1
    elif game.startswith('Strand_2'):
        return strand_2
    elif game.startswith('VDK_1'):
        return vdk_1
    elif game.startswith('VDK_2'):
        return vdk_2
    elif game.startswith('Winnie_1'):
        return winnie_1
    elif game.startswith('Winnie_2'):
        return winnie_2
    else:
        return None
