import datetime
import os

import cv2
import imutils
import numpy as np

import compare3 as cb
import Solution as s


def handDetected(last):
    if np.mean(last) > 0.0022:
        return False
    else:
        return True


''' PARAMETERS '''
PARAMS = [['Strand_1', 0, 20],
          ['Strand_2', 0, 20],
          ['Winnie_1', 0, 20],
          ['Crayon_2', 0, 20],
          ['VDK_1', 0, 15],
          ['VDK_2', 1, 20],
          ['Disney_1', 0, 35],
          ['Winnie_2', 0, 10]]

# VIDEOS_PATH = '../Memory_Games/'
VIDEOS_PATH = 'Memory_Games/'
OUTPUT_PATH = 'fixed_frames/'
path = ""
show = True

'''Search key frames in all videos from folder'''
for dirpath, dirnames, filenames in os.walk(VIDEOS_PATH):
    for filename in filenames:

        # Search if solution is known
        solution = s.solution(os.path.splitext(filename)[0])
        if solution is not None and not os.path.splitext(filename)[0].endswith('240'):

            # Start timing and create output folder
            starttime = datetime.datetime.now()
            path = OUTPUT_PATH + os.path.splitext(filename)[0] + '/'
            print('path: ' + path)
            if not os.path.exists(OUTPUT_PATH + os.path.splitext(filename)[0]):
                print(OUTPUT_PATH + os.path.splitext(filename)[0])
                os.makedirs(OUTPUT_PATH + os.path.splitext(filename)[0])

            # # Search and set parameters for known solutions
            par = 0
            counter = 0
            for file in PARAMS:
                if os.path.splitext(filename)[0].startswith(PARAMS[counter][0]):
                    par = counter
                counter += 1

            handoutframe = PARAMS[par][1]
            length = PARAMS[par][2]
            game = PARAMS[par][0]

            # Start video stream
            vs = cv2.VideoCapture(VIDEOS_PATH + filename)
            print(path)

            # Create background subtractor
            fgbg = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=0)

            # Initialise needed parameters
            last = np.array([])
            counter = 0
            saved = False
            hand = False

            # Read first frame and save -> needed to select right tiles
            ret, frame = vs.read()
            pad = path + '0'
            cv2.imwrite(pad + '.jpg', frame)

            # Read video frame by frame
            while ret:
                counter += 1

                # Resize image and apply background subtractor
                resized = imutils.resize(frame, width=600)
                fgmask = fgbg.apply(resized)

                fgmask = cv2.dilate(fgmask, (3, 3))
                fgmask = cv2.erode(fgmask, (3, 3))

                # Calculate the amount of pixels in the video
                pixels = fgmask.shape[0] * fgmask.shape[1]

                # Give time to the background subtractor to warm up a little bit
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
                        if not saved:
                            handoutframe += 1
                            saved = True
                            if handoutframe % 2 == 0:
                                pad = path + 'frame' + str(counter)
                                print(pad)
                                cv2.imwrite(pad + '.jpg', frame)
                    else:
                        hand = False
                        saved = False
                if show:
                    cv2.putText(resized, str(not hand), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                    cv2.putText(resized, str(handoutframe), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                    #
                    cv2.imshow(filename, resized)
                    cv2.imshow('fgbg', fgmask)

                # Stop calculating this video and go to next
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Read next frame
                ret, frame = vs.read()

            # Do cleanup
            cv2.destroyAllWindows()
            vs.release()

            # Find solution to found key frames
            try:
                cb.main_function(filename[:-4])
            except:
                print('Geen matching gevonden')

            # Print time passed for each video
            print('Passed time for video ' + filename + ': ')
            print(datetime.datetime.now() - starttime)
