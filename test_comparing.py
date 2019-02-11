import os

import cv2
import matplotlib.pyplot as plt

# Test purposes, this will be automated
ditpath = os.path.dirname(os.path.abspath(__file__))


def read_images():
    frames = []

    i = 0
    frames.append(cv2.imread(ditpath + "/fixed_frames/11_13.png"))
    i = i + 1
    frames.append(cv2.imread(ditpath + "/fixed_frames/14_23.png"))
    i = i + 1
    frames.append(cv2.imread(ditpath + "/fixed_frames/22_44.png"))
    i = i + 1
    frames.append(cv2.imread(ditpath + "/fixed_frames/32_41.png"))
    i = i + 1
    frames.append(cv2.imread(ditpath + "/fixed_frames/12_34.png"))
    i = i + 1
    frames.append(cv2.imread(ditpath + "/fixed_frames/21_24.png"))
    i = i + 1
    frames.append(cv2.imread(ditpath + "/fixed_frames/31_42.png"))
    i = i + 1
    frames.append(cv2.imread(ditpath + "/fixed_frames/33_43.png"))
    return frames


all_the_frames = read_images()

allbacks = cv2.imread(ditpath + '/fixed_frames/frame1.png')


# Plot images in an 4x4 grid
def plot4by4(title, image1, image2, image3, image4):
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(221)
    plt.plot(image1)
    plt.subplot(222)
    plt.plot(image2)
    plt.subplot(223)
    plt.plot(image3)
    plt.subplot(224)
    plt.plot(image4)
    plt.show()


# calculate histogram of an image
def calc_cum_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cumsum = hist.cumsum()
    # print(cumsum)
    return cumsum


# Deze functie vergelijk twee cdf's van histogrammen en geeft een score terug.
def compare_cdf(cdf1, cdf2):
    if (len(cdf1) == len(cdf2)):
        score = 0
        for i in range(len(cdf1)):
            score += abs(cdf1[i] - cdf2[i])
        return score

    else:
        print("Error: cdf's hebben verschillende grootte.")


# gemiddelde cdf van een array bepalen
def get_mean_cdf(tiles):
    res = tiles[0].cdf
    length = len(tiles)
    for i in range(1, length):
        res += tiles[i].cdf
    mean = res / length
    return mean


class Tile:
    def __init__(self, index, image):
        self.index = index
        self.image = image
        self.cdf = calc_cum_hist(image)

    def compare_cdf(self, cdf):
        return compare_cdf(self.cdf, cdf)


# extract the grid from the picture
def get_grid(frame):
    # cany_edges = cv2.Canny(frame,200,800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 140, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    # cv2.imshow("edges", edges)

    tiles = []
    for i in range(0, 16):
        contour = cntsSorted[i]
        # print(str(cv2.contourArea(contour)))
        x, y, w, h = cv2.boundingRect(contour)
        tiles.append(Tile(i, frame[y:y + h, x:(x + w)]))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    meancdf = get_mean_cdf(tiles)
    print(meancdf)
    for i in range(0, len(tiles)):
        print(tiles[i].compare_cdf(meancdf))

    cv2.imshow("edges", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# get_grid(all_the_frames[1])
# ground truth
get_grid(allbacks)
