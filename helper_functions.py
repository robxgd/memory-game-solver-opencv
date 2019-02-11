import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# Plot graphs in n by m grid !!!!NOT WORKING FOR 1X1
def plot_n_by_m(title, images, n, m, property=None):
    fig = plt.figure()
    fig.suptitle(title)

    size_string = str(n) + str(m)
    print(size_string)
    number_of_tiles = n * m
    for i in range(0, number_of_tiles):
        plt.subplot(n, m, i + 1)
        plt.plot(images[i])
    plt.show()


# plot images in n by m grid  !!!!NOT WORKING FOR 1X1
def plot_n_by_m_image(title, images, n, m, property=None):
    fig = plt.figure()
    fig.suptitle(title)

    size_string = str(n) + str(m)
    number_of_tiles = n * m
    for i in range(0, number_of_tiles):
        plt.subplot(n, m, i + 1)

        if property:
            plt.imshow(cv2.cvtColor(getattr(images[i], property), cv2.COLOR_BGR2RGB), cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB), cmap='gray')
    plt.show()


# calculate cumulative histogram of an image (cdf)
def calc_cum_hist(img):
    cumsum = calc_histogram(img).cumsum()
    # print(cumsum)
    return cumsum


# calculate normalized histogram of image
def calc_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    res = cv2.normalize(hist, hist).flatten()
    return res


# compare histograms
def compare_hist(hist1, hist2):
    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    # print( res)
    return res


# get the average histograme of al the tiles in the frame
def average_histogram(hists):
    hists = np.array(hists)
    hist = np.sum(hists, axis=0)
    starthist = np.divide(hist, len(hists))
    # plt.plot(starthist)
    # plt.ylabel('value')
    # plt.show()   
    return starthist


def crop_minAreaRect_old(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    # showimg(img_rot)

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]
    # showimg(img_crop)
    return img_crop


def crop_minAreaRect(img, rect, box):
    mult = 1
    W = rect[1][0]
    H = rect[1][1]
    img_box = img.copy()

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(mult * (x2 - x1)), int(mult * (y2 - y1)))
    # cv2.circle(img_box, center, 10, (0,255,0), -1) #again this was mostly for debugging purposes

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img_box, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW * mult), int(croppedH * mult)),
                                       (size[0] / 2, size[1] / 2))

    return croppedRotated


def showimg(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_blurry_edge_image(img1, treshold_value, blurval):
    oriimggray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray11 = cv2.resize(oriimggray1, (300, 300))
    ret, tresh1 = cv2.threshold(gray11, treshold_value, 255, cv2.THRESH_BINARY)
    gray1 = cv2.resize(tresh1, (300, 300))
    blurred1 = cv2.GaussianBlur(gray1, (blurval, blurval), 0)
    # cv2.imshow("blurry", blurred1)
    # showimg(blurred1)
    return blurred1


# lower is more similar
def get_best_diff_score(img1, img2, treshold_value, blurval, disp=False):
    blurred1 = get_blurry_edge_image(img1, treshold_value, blurval)
    blurred2 = get_blurry_edge_image(img2, treshold_value, blurval)
    res = math.inf
    for i in range(0, 4):
        # res = np.sum(blurred1-blurred2)
        blurred1 = np.rot90(blurred1, i)
        temp = np.sum(cv2.subtract(blurred1, blurred2))
        if (disp):
            cv2.imshow(str(temp + i), cv2.subtract(blurred1, blurred2))
        if temp < res:
            res = temp
    if (disp):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return res


def compare_by_substraction(img1, img2, treshold_value, blurval):
    oriimggray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray11 = cv2.resize(oriimggray1, (300, 300))
    ret, tresh1 = cv2.threshold(gray11, treshold_value, 255, cv2.THRESH_BINARY)
    gray1 = cv2.resize(tresh1, (300, 300))
    blurred1 = cv2.GaussianBlur(gray1, (blurval, blurval), 0)

    oriimggray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray21 = cv2.resize(oriimggray2, (300, 300))
    ret, tresh2 = cv2.threshold(gray21, treshold_value, 255, cv2.THRESH_BINARY)
    gray2 = cv2.resize(tresh2, (300, 300))
    blurred2 = cv2.GaussianBlur(gray2, (blurval, blurval), 0)

    res = np.sum(blurred1 - blurred2)
    for i in range(0, 3):
        blurred1 = np.rot90(blurred1, i)
        temp = np.sum(blurred1 - blurred2)
        # print(temp)
        if temp < res:
            res = temp
    cv2.imshow("testingzzz", (blurred1 - blurred2))
    return res


def add_position_in_grid(tiles, max_tile_height, max_tile_width):
    tiles = sorted(tiles, key=lambda x: x.center[0], reverse=False)
    max_tile_height = max_tile_height * 0.5
    max_tile_width = max_tile_width * 0.5
    rows = -1
    cols = -1
    lastx = -max_tile_width
    lasty = -max_tile_height
    for i in range(len(tiles)):
        if cols < 0:
            cols = 0
        if (lastx + max_tile_width < tiles[i].center[0]):
            cols = cols + 1
        lastx = tiles[i].center[0]
        tiles[i].col = cols

    tiles = sorted(tiles, key=lambda x: x.center[1], reverse=False)
    for i in range(len(tiles)):
        if rows < 0:
            rows = 0
        if (lasty + max_tile_height < tiles[i].center[1]):
            rows = rows + 1
        lasty = tiles[i].center[1]
        tiles[i].row = rows
    # for i in range(len(tiles)):
    #     print("row "+str(tiles[i].row)+"col "+str(tiles[i].col))
    # print("there are "+str(cols)+" cols")
    return cols, rows


def get_orb_descriptors(image):
    kp1, des1 = orb.detectAndCompute(image, None)
    # print(des1)
    if des1 is None:
        return []
    else:
        return des1


def compare_with_orb(deseen, destwee):
    bf = cv2.BFMatcher()
    good = []
    if (len(deseen) and len(destwee)):

        matches = bf.knnMatch(deseen, destwee, k=2)
        for m, n in matches:

            if m.distance < 0.75 * n.distance:
                good.append([m])
        # print('matching gooooood')

    else:
        print("orb matching failed")
    return len(good)
