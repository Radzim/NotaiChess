import time
import os
from ChessFunctions import *
from GridDetect import *

def play(url):
    lastMoveTime = time.time()
    timeLeft, timeIncrement = time_control(AT_startTime, increment=AT_incrementTime)

    board = chess.Board()
    pngBoard = board_to_png(board)
    pgn = ''
    winner = ''

    capture = cv2.VideoCapture(url)


    centralPoint, squareRadius, angle = detect_grid(capture)

    referenceFrameWithBuffer = 0
    lastSquares = []
    stabilityCounter = 0

    while winner == '':
        ret, readFrame = capture.read()

        frameWithBuffer = rotate_crop_frame(readFrame, centralPoint, squareRadius, angle)

        scaling = AT_scaling
        shape = np.array(np.multiply(np.shape(frameWithBuffer)[:2], scaling), dtype=int)
        smallFrameWithBuffer = cv2.resize(frameWithBuffer, shape, interpolation=cv2.INTER_AREA)
        grayscaleFrameWithBuffer = smallFrameWithBuffer[:, :, 2]

        if np.sum(referenceFrameWithBuffer) == 0:
            referenceFrameWithBuffer = grayscaleFrameWithBuffer.astype(float)

        diffFrameWithBuffer = grayscaleFrameWithBuffer.astype(float) - referenceFrameWithBuffer
        diffThreshWithBuffer = np.array(np.where(np.absolute(diffFrameWithBuffer) > AT_changeThreshold*255, 255, 0), dtype=np.uint8)
        diffThreshWithBuffer = cv2.medianBlur(diffThreshWithBuffer, 5)
        bufferCrop = np.array([np.multiply(np.shape(frameWithBuffer)[0], scaling)//18, np.multiply(np.shape(frameWithBuffer)[0], scaling)*17//18], dtype=int)
        diffThresh = diffThreshWithBuffer[bufferCrop[0]:bufferCrop[1], bufferCrop[0]:bufferCrop[1]]
        gridFrame = draw_grid(grayscaleFrameWithBuffer[bufferCrop[0]:bufferCrop[1], bufferCrop[0]:bufferCrop[1]])

        squareCentres = square_centres(diffThresh)
        numpySquares = np.reshape(np.array(squareCentres, dtype=np.uint8), (-1, 8))
        squaresImage = cv2.resize(numpySquares, dsize=(540, 540), interpolation=cv2.INTER_NEAREST)
        twoSquares = two_squares(squareCentres)

        stabilityCounter = detect_stable_move(stabilityCounter, squareCentres, diffThreshWithBuffer, diffThresh, twoSquares, lastSquares)

        if stabilityCounter >= AT_stableFramesForMove:
            stabilityCounter = 0
            potential_uci_move = try_uci_move(twoSquares, board.turn, new_square_owner(board), squareNames)
            uci_move = uci_move_promotion_check(potential_uci_move, board.legal_moves)
            if uci_move != '':
                timeLeft[board.turn] = timeLeft[board.turn] - (time.time() - lastMoveTime) + timeIncrement
                lastMoveTime = time.time()

                board.push_uci(uci_move)
                pgn = update_pgn(pgn, board, uci_move)
                pngBoard = board_to_png(board)

                winner = analyse_position(board)

            referenceFrameWithBuffer = grayscaleFrameWithBuffer.astype(float)


        lastSquares = twoSquares
        liveTimeLeft = update_live_time(timeLeft, lastMoveTime, board.turn, time.time())
        winner += check_timeout(liveTimeLeft, board.turn)

        if AT_showEverything:
            cv2.imshow('Original', smallFrameWithBuffer)
            cv2.imshow('Gridboard', gridFrame)
            cv2.imshow('SquaresAvg', squaresImage)
            cv2.imshow('Digital Board', pngBoard)
        if AT_showAnything:
            cv2.imshow('Buffer2', diffThreshWithBuffer)
            printout = create_printout(liveTimeLeft, pgn, winner)
            cv2.imshow('Live Game Analysis', printout)

        if np.average(diffThreshWithBuffer) > AT_changeForQuit*255:
            break
        if cv2.waitKey(1) == ord("q"):
            break
        if cv2.waitKey(1) == ord("r"):
            break
        if cv2.waitKey(1) == ord("c"):
            referenceFrameWithBuffer = grayscaleFrameWithBuffer.astype(float)

    capture.release()
    print(pgn)
    cv2.destroyAllWindows()
    os.remove("board.png")
    os.remove("board.svg")
    return pgn
