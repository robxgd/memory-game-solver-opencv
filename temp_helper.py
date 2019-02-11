import numpy as np
import cv2
import time


# Deze functie vergelijk twee cdf's van histogrammen en geeft een score terug.
# lower = more similar
def compare_cdf(cdf1, cdf2):
    if (len(cdf1) == len(cdf2)):
        score = 0
        for i in range(len(cdf1)):
            score += abs(cdf1[i] - cdf2[i])
        return np.sum(score)

    else:
        print("Error: cdf's hebben verschillende grootte.")


# Deze functie knipt een afbeelding in x*y rechthoekige stukjes en geeft
# deze als een array terug.
def cut_img(img, x, y, debug=False):
    (rows, cols) = img.shape[0:2]
    res = []
    height = int(rows / x)
    width = int(cols / y)

    if (debug):
        print("Height: " + str(height))
        print("Width: " + str(width))

    for i in range(x):
        for j in range(y):
            res.append(img[i * height:(i + 1) * height:1, j * width:(j + 1) * width:1])

    return res


def calc_cdf(img, roi=True, equalize_hist=True, sections_x=2, sections_y=1, crono=False):
    # Deze functie verwacht zwart-wit afbeeldingen
    if (crono):
        start = time.clock()
    if (roi):
        # Bovenste en onderste balk afknippen van afbeelding.
        img = img[int(roi_x * image_height):int(roi_y * image_height):1, 0:image_width:1]
    if (equalize_hist):
        # Normalisatie van de inputafbeelding door histogramequalisatie.
        img = cv2.equalizeHist(img)

    # Inputafbeelding in stukken knippen.
    img_sections = cut_img(img, sections_x, sections_y)
    cdf_array = []

    # Van elk stuk van de inputafbeelding het histgram bereken, daarvan de cdf nemen en opslaan.
    for i in range(len(img_sections)):
        img_hist = cv2.calcHist([img_sections[i]], [0], None, [256], [0, 256])
        cdf = img_hist.cumsum()
        cdf_array.append(cdf)

    cdf_array = np.array(cdf_array)

    if (crono):
        stop = time.clock()
        print("Elapsed time of calc_cdf: " + str(stop - start))

    return cdf_array
