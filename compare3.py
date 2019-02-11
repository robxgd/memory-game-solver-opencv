import glob
import math

import cv2
import numpy as np

import Solution as s
from classes import Tile
from helper_functions import add_position_in_grid, plot_n_by_m_image
from helper_functions import crop_minAreaRect
from helper_functions import get_best_diff_score, get_orb_descriptors, compare_with_orb
from temp_helper import compare_cdf, calc_cdf

global game
global all_the_frames


# Working:
# game = "Disney_1"
# game = "Strand_1"
# game = "Strand_2"
# game = "Winnie_1"
# game = "VDK_2"
# game = "Winnie_2"
# game = "VDK_1"

# in progress

# not working
# game = "Crayon_2"

# read the images, later on this will be provided by a function.
# ditpath=os.path.dirname(os.path.abspath(__file__))
def read_images(game_name):
    frames = [cv2.imread(file) for file in glob.glob("fixed_frames/" + game_name + "/frame*.jpg")]

    return frames


orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# creates  array of tiles 
def get_grid(frame):
    height, width, channels = frame.shape
    print("------Start getting grid ---------")

    # treshold to find contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    # cv2.imshow("tresh", thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # make tiles array
    max_tile_width = 0
    max_tile_height = 0
    tiles = []
    for i in range(0, len(all_the_frames) * 2):
        # print("aantal contours "+str(len(cntsSorted)))
        contour = cntsSorted[i]
        orig = frame.copy()
        # ratio = frame.shape[0] / 300.0
        # frame = imutils.resize(frame, height = 300)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
        if (len(approx) == 4):
            screenCnt = approx

            pts = screenCnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            # the top-left point has the smallest sum whereas the
            # bottom-right has the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            # compute the difference between the points -- the top-right
            # will have the minumum difference and the bottom-left will
            # have the maximum difference
            diff = np.diff(pts, axis=1)
            rect[3] = pts[np.argmin(diff)]
            rect[1] = pts[np.argmax(diff)]

            # the width and height of the new image
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

            # take the maximum of the width and height values to reach
            # our final dimensions
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))

            # construct our destination points which will be used to
            # map the screen to a top-down, "birds eye" view
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            # calculate the perspective transform matrix and warp
            # the perspective to grab the screen
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

            # center of contour
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        else:
            rect = cv2.minAreaRect(contour)
            cX, cY = rect[0]
            cX = int(cX)
            cY = int(cY)
            w, h = rect[1]

            box = cv2.boxPoints(rect)
            box = np.int0(box)

            warp = crop_minAreaRect(frame, rect, box)

        # cv2.imshow(str(i), warp)

        tiles.append(Tile(i, warp, (cX, cY)))
        # cv2.drawContours(frame,[box],0,(0,0,255),2)

        if maxWidth > max_tile_width:
            max_tile_width = maxWidth
        if maxHeight > max_tile_height:
            max_tile_height = maxHeight

    glob_cols, glob_rows = add_position_in_grid(tiles, max_tile_height, max_tile_width)

    return tiles, glob_cols, glob_rows


# werkt voor strand 1 en 2 en disney1
def get_turned_frames2(tiles, backtile, disp):
    print("---------- start getting turned tiles -----------------")
    grayback = cv2.cvtColor(backtile.image, cv2.COLOR_BGR2GRAY)
    grayback = cv2.GaussianBlur(grayback, (3, 3), 0)

    cdf_back = calc_cdf(grayback, roi=False)
    des_back = get_orb_descriptors(grayback)
    # blurry_edges_back = get_blurry_edge_image(backtile.image, tresholdval, 15)
    # showimg(backtile.image)
    print("desback:")
    # print(des_back)
    max_diff1 = 0
    max_diff2 = 0
    max_index1 = 0
    max_index2 = 1
    orb_score1 = math.inf
    orb_score2 = math.inf
    max_substract_score1 = 0
    max_substract_score2 = 0

    for i in range(0, len(tiles)):
        # cv2.imshow("back", backtile.image)
        # cv2.imshow(str(i), tiles[i].image)
        # if(False):

        # temporary for testing vdk
        if (len(des_back)):
            gray = cv2.cvtColor(tiles[i].image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(grayback, (5, 5), 0)
            cdf = calc_cdf(gray, roi=False, equalize_hist=False)
            diff = compare_cdf(cdf_back, cdf)
            substraction_score = get_best_diff_score(backtile.image, tiles[i].image, 145, 15, disp=False)
            print("subtract score: " + str(substraction_score) + " current max" + str(
                max_substract_score1) + " and " + str(max_substract_score2))
            print("cdf score: " + str(diff))
            # if(True):
            if (substraction_score > max_substract_score1):
                # if(max_index1!=i):
                max_substract_score2 = max_substract_score1
                max_index2 = max_index1
                max_substract_score1 = substraction_score
                max_index1 = i
            elif (substraction_score > max_substract_score2):
                max_substract_score2 = substraction_score
                max_index2 = i
            print("we have index " + str(max_index2) + " and " + str(max_index1))


        elif (len(des_back)):
            gray = cv2.cvtColor(tiles[i].image, cv2.COLOR_BGR2GRAY)
            cdf = calc_cdf(gray, roi=False, equalize_hist=False)
            diff = compare_cdf(cdf_back, cdf)
            if (diff > max_diff1 * 0.5):
                temp_descr = get_orb_descriptors(gray)
                # cv2.imshow(str(i),gray)
                score = compare_with_orb(temp_descr, des_back)
                # print("turned_detection score voor "+str(i)+" is "+str(score))
                if (score < orb_score1):
                    orb_score2 = orb_score1
                    orb_score1 = score
                    max_diff2 = max_diff1
                    max_index2 = max_index1
                    max_diff1 = diff
                    max_index1 = i
                elif (score < orb_score2):
                    orb_score2 = score
                    max_diff2 = diff
                    max_index2 = i
            elif (diff > max_diff2 * 1.1):
                if (score > orb_score2):
                    orb_score2 = score
                    max_diff2 = diff
                    max_index2 = i
        else:
            gray = cv2.cvtColor(tiles[i].image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            temp_descr = get_orb_descriptors(gray)
            diff = get_best_diff_score(backtile.image, tiles[i].image, 90, 7)

            if (len(temp_descr)):
                if (len(temp_descr) > diff):
                    print("lets go for orb")
                    diff = len(temp_descr)

            if (diff > max_diff1):
                max_diff2 = max_diff1
                max_index2 = max_index1
                max_diff1 = diff
                max_index1 = i
            elif (diff > max_diff2):
                max_diff2 = diff
                max_index2 = i

    return tiles[max_index1], tiles[max_index2]


def match(tiles, tresholdval):
    for i in range(0, len(tiles)):
        print("---> TILE " + str(i))
        if tiles[i].paired == None:
            for l in range(0, len(tiles)):
                if i != l and tiles[l].paired == None:
                    # 215 strand
                    # how lower, how more similar
                    disp = False
                    this_score = get_best_diff_score(tiles[i].image, tiles[l].image, tresholdval, 9, disp)
                    if (disp):
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    print("voor " + str(l) + " " + str(this_score))
                    if (tiles[i].matching_score == math.inf):
                        tiles[i].matching_score = this_score
                    # print(str(l)+" "+str(abs(this_score)))
                    # print(str(l)+" "+str(abs(this_score-tiles[i].matching_score)))
                    # for strand_2 5 is ok. 

                    ##we add the potential candidates using a cascade
                    if this_score < np.multiply(15, tiles[i].matching_score):
                        if this_score < tiles[i].matching_score and this_score != 0:
                            tiles[i].matching_score = this_score
                        # calc the cdf
                        gray1 = cv2.cvtColor(tiles[i].image, cv2.COLOR_BGR2GRAY)
                        gray2 = cv2.cvtColor(tiles[l].image, cv2.COLOR_BGR2GRAY)
                        cdf1 = calc_cdf(gray1, roi=False)
                        cdf2 = calc_cdf(gray2, roi=False)
                        cdf_score = compare_cdf(cdf1, cdf2)
                        print("voor " + str(i) + " en " + str(l) + " hebben we een cdf-score van " + str(cdf_score))
                        if (cdf_score < 9 * tiles[i].best_cdf_score):
                            # potential candidate

                            tiles[i].potential_matches.append(l)
                            if (cdf_score < 1.2 * tiles[i].best_cdf_score):
                                tiles[i].best_cdf_score = cdf_score

            print(tiles[i].potential_matches)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # tiles[i].paired = tiles[i].potential_matches[0]
            gaussfactor = (1, 3)
            img = np.copy(tiles[i].image)
            img = cv2.GaussianBlur(img, gaussfactor, 0)
            kp1, des1 = orb.detectAndCompute(img, None)

            img2 = np.copy(tiles[tiles[i].potential_matches[0]].image)
            img2 = cv2.GaussianBlur(img2, gaussfactor, 0)

            kp2, des2 = orb.detectAndCompute(img2, None)

            matches = []
            score = 0
            for l in range(0, len(tiles[i].potential_matches)):
                img2 = np.copy(tiles[tiles[i].potential_matches[l]].image)
                img2 = cv2.GaussianBlur(img2, gaussfactor, 0)

                kp2, des2 = orb.detectAndCompute(img2, None)
                matches = bf.knnMatch(des1, des2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                print("for " + str(tiles[i].potential_matches[l]) + " we have score " + str(len(good)))
                if len(good) >= score:
                    score = len(good)
                    if tiles[i].paired != None:
                        tiles[tiles[i].paired].paired = None
                    tiles[i].paired = tiles[i].potential_matches[l]
            print("dan wordt " + str(i) + " bij " + str(tiles[i].paired) + " geplaatst")
            tiles[tiles[i].paired].paired = i


def main_function(file):
    global game
    global all_the_frames
    game = file
    all_the_frames = read_images(game)
    visible_tiles = []
    backpath = "fixed_frames/" + game + "/0.jpg"
    print(backpath)
    backframe = cv2.imread(backpath)
    back_of_tiles, g_cols, g_rows = get_grid(backframe)
    back_of_tile = back_of_tiles[0]
    # showimg(back_of_tile.image)
    all_tiles = []

    # find all turned tiles
    for i in range(0, len(all_the_frames)):
        print(i)
        tiles, cols, rows = get_grid(all_the_frames[i])

        disp = False
        # if(i==10):
        #     disp=True
        turned1, turned2 = get_turned_frames2(tiles, back_of_tile, disp)

        all_tiles.append(turned1)
        all_tiles.append(turned2)

    all_tiles = sorted(all_tiles, key=lambda x: (x.row, x.col))
    print("there are " + str(g_cols) + " cols and " + str(g_rows) + " rows")
    # all tile

    plot_n_by_m_image(game, all_tiles, g_rows, g_cols, "image")

    match(all_tiles, 145)
    print("all the tiles : " + str(len(all_tiles)))
    # all_tiles_sorted = sorted(all_tiles, key=lambda x: (x.row, x.col))

    # plot_n_by_m_image("test", all_tiles_sorted, g_rows, g_cols, "image")

    pairs = []
    done_pairs = []
    for i in range(0, len(all_the_frames) * 2):
        if i not in done_pairs:
            pairs.append(all_tiles[i])
            pairs.append(all_tiles[all_tiles[i].paired])
            done_pairs.append(i)
            done_pairs.append(all_tiles[i].paired)
    plot_n_by_m_image(game, pairs, g_rows, g_cols, "image")

    solutions = np.zeros(cols * rows, dtype=int)
    print(solutions)
    pair = 1
    for i in range(len(done_pairs)):
        if i % 2 == 0:
            solutions[done_pairs[i]] = pair
            solutions[done_pairs[i + 1]] = pair
            pair += 1

    nslices = 4
    solutions = solutions.reshape((nslices, -1))
    print('calculated')
    print(solutions)

    check = np.array(s.solution(game=game))
    print('solution')
    print(check)

    controle = np.count_nonzero(solutions == check) / (cols * rows) * 100
    print('Percentage correct: ' + str(controle) + '%')

    np.savetxt(game + '_' + str(all_the_frames[0].shape[0]) + ".txt", solutions, fmt="%02d", delimiter=",")

# main_function()
