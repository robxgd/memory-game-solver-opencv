import datetime
import os

import cv2
import imutils
import numpy as np

import compare3 as cb
import Solution as s

FILENAME = "Strand_1.MOV"
show = True

# Start timing
starttime = datetime.datetime.now()

''' PARAMETERS '''
PARAMS = [['Strand_1', 0, 20],
          ['Strand_2', 0, 20],
          ['Winnie_1', 0, 20],
          ['Crayon_2', 0, 20],
          ['VDK_1', 0, 15],
          ['VDK_2', 1, 20],
          ['Disney_1', 0, 35],
          ['Winnie_2', 0, 10]]


def handDetected(last):
    if np.mean(last) > 0.0022:
        return False
    else:
        return True


VIDEOS_PATH = 'Memory_Games/'
OUTPUT_PATH = 'fixed_frames/'
FILEPATH = FILENAME[:-4]
key_frames = np.array([])

'''Search key frames video'''
# Search if solution is known
solution = s.solution(FILENAME)
if solution is not None and not FILEPATH.endswith('240'):
    # Make output folders
    if not os.path.exists(OUTPUT_PATH[:-1]):
        os.makedirs(OUTPUT_PATH[:-1])
    if not os.path.exists(OUTPUT_PATH + FILEPATH):
        print(OUTPUT_PATH + FILEPATH)
        os.makedirs(OUTPUT_PATH + FILEPATH)

    # Search and set parameters for known solutions
    par = 0
    counter = 0
    for file in PARAMS:
        if FILEPATH.startswith(PARAMS[counter][0]):
            par = counter
        counter += 1
    print(PARAMS[par][0])

    handoutframe = PARAMS[par][1]
    length = PARAMS[par][2]
    game = PARAMS[par][0]

    # Start video stream and set path variables
    vs = cv2.VideoCapture(VIDEOS_PATH + FILENAME)
    path = VIDEOS_PATH + FILEPATH
    pad = OUTPUT_PATH + FILEPATH + '/0'
    print(path)

    # Create background subtraction
    fgbg = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=0)

    # Set parameters
    last = np.array([])
    counter = 0
    saved = False
    hand = False

    # Read and save first frame, needed to determine turned cards
    ret, frame = vs.read()
    cv2.imwrite(pad + '.jpg', frame)

    while ret:
        counter += 1

        # Resize image and apply background subtractor

        resized = imutils.resize(frame, width=600)
        fgmask = fgbg.apply(resized)

        # Dilate and erode to clean the fgmask
        fgmask = cv2.dilate(fgmask, (3, 3))
        fgmask = cv2.erode(fgmask, (3, 3))

        # Calculate the amount of pixels in the video
        pixels = fgmask.shape[0] * fgmask.shape[1]

        # Fill the array until length is 20
        # Ensure length is always 20
        if counter > 20:
            som = np.sum(fgmask) / pixels / 255
            if len(last) < length:
                last = np.append(last, som)
            else:
                last = np.delete(last, 0)
                last = np.append(last, som)

            # Calculate mean of last 20 fgmasks
            # If mean difference between the frames is larger than given threshold, a hand is detected
            # A frame has two images with the right side up if a hand has gone out of the frame any two times

            if handDetected(last):
                hand = True
                if saved == False:
                    handoutframe += 1
                    saved = True
                    if handoutframe % 2 == 0:
                        pad = OUTPUT_PATH + FILEPATH + '/frame' + str(counter)
                        cv2.imwrite(pad + '.jpg', frame)
            else:
                hand = False
                saved = False
            if show:
                cv2.putText(resized, str(not hand), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                cv2.putText(resized, str(handoutframe), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                #
                cv2.imshow(FILENAME, resized)
                cv2.imshow('fgbg', fgmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Read next frame
        ret, frame = vs.read()

    # Do cleanup
    cv2.destroyAllWindows()
    vs.release()

    # Find solution to found key frames
    try:
        cb.main_function(FILEPATH)
    except:
        print('Geen matching gevonden')

    # Print time passed for this video
    print(datetime.datetime.now() - starttime)
else:
    print('No known solution was found')
