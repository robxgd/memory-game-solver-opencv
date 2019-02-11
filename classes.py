import math


class Tile:
    def __init__(self, index, image, center):
        self.index = index
        self.image = image
        self.paired = None
        self.matching_score = math.inf
        self.potential_matches = []
        self.center = center
        self.col = 0
        self.row = 0
        self.best_cdf_score = math.inf
