import time
import os
from ChessFunctions import *
from GridDetect import *


# TODO: UPPERCASE VARIABLES
def play(url):
    return_code = ''
    return_value = ''
    # STARTING VIDEO STREAM FROM URL
    print('Getting video stream...')
    capture = cv2.VideoCapture(url)
    ret, readFrame = capture.read()
    if not ret:
        return_code = 'ERROR 1A: no video stream found'
        return return_code, return_value
    print('Video stream found.', np.shape(readFrame)[:2])
    # GRID LINEUP
    grid_detector_values = detect_grid(capture)
    if grid_detector_values[:5] == 'ERROR':
        return_code = grid_detector_values
        return return_code, return_value
    grid_central_point, grid_radius, grid_angle = grid_detector_values
    # TODO: too long lines
    print('Grid detected.', 'centre:', tuple(grid_central_point), 'size:', (2*grid_radius, 2*grid_radius) , 'a:', grid_angle)
    # INITIALISING
    lastMoveTime = time.time()
    timeLeft, timeIncrement = time_control(AT_startTime, increment=AT_incrementTime)
    board = chess.Board()
    pngBoard = board_to_png(board)
    pgn = ''
    referenceFrameWithBuffer = 0
    lastSquares = []
    stabilityCounter = 0
    # MAIN LOOP UNTIL GAME ENDS
    while True:
        # FRAME READING MECHANISM
        ret, readFrame = capture.read()
        cv2.imshow('ReadFrame', readFrame)
        if not ret:
            return_code = 'ERROR 1B: video stream stopped'
            break
        # USE GRID FINDING TO ISOLATE BOARD AND MARGIN
        frameWithBuffer = rotate_crop_frame(readFrame, grid_central_point, grid_radius, grid_angle)
        shape = np.array(np.multiply(np.shape(frameWithBuffer)[:2], AT_scaling), dtype=int)
        smallFrameWithBuffer = cv2.resize(frameWithBuffer, shape, interpolation=cv2.INTER_AREA)
        grayscaleFrameWithBuffer = smallFrameWithBuffer[:, :, 2]  # using red channel for better results than grayscale
        # EDGE CASE FIRST REFERENCE FRAME INITIALISATION
        if np.sum(referenceFrameWithBuffer) == 0:
            referenceFrameWithBuffer = grayscaleFrameWithBuffer.astype(float)
        # FIND FRAME DIFFERENCES
        diffFrameWithBuffer = grayscaleFrameWithBuffer.astype(float) - referenceFrameWithBuffer
        # TODO: too long lines
        diffThreshWithBuffer = np.array(np.where(np.absolute(diffFrameWithBuffer) > AT_changeThreshold * 255, 255, 0), dtype=np.uint8)
        diffThreshWithBuffer = cv2.medianBlur(diffThreshWithBuffer, 5)
        bufferCrop = np.array([shape[0] // 18, shape[1] * 17 // 18], dtype=int)
        diffThresh = diffThreshWithBuffer[bufferCrop[0]:bufferCrop[1], bufferCrop[0]:bufferCrop[1]]
        gridFrame = draw_grid(grayscaleFrameWithBuffer[bufferCrop[0]:bufferCrop[1], bufferCrop[0]:bufferCrop[1]])
        # CALCULATE AVERAGE SQUARE VALUE CHANGES
        squareCentres = square_centres(diffThresh)
        numpySquares = np.reshape(np.array(squareCentres, dtype=np.uint8), (-1, 8))
        squaresImage = cv2.resize(numpySquares, dsize=(540, 540), interpolation=cv2.INTER_NEAREST)
        twoSquares = two_squares(squareCentres)
        # DETECT STABLE DIFFERENCES THAT MAY BE MOVES
        # TODO: too long lines
        stabilityCounter = detect_stable_move(stabilityCounter, squareCentres, diffThreshWithBuffer, diffThresh, twoSquares, lastSquares)
        # IF A STABLE CHANGE IS DETECTED
        if stabilityCounter >= AT_stableFramesForMove:
            # CALCULATE MOVE
            potential_uci_move = try_uci_move(twoSquares, board.turn, new_square_owner(board), squareNames)
            uci_move = uci_move_promotion_check(potential_uci_move, board.legal_moves)
            # MAKE MOVE
            if uci_move != '':
                timeLeft[board.turn] = timeLeft[board.turn] - (time.time() - lastMoveTime) + timeIncrement
                lastMoveTime = time.time()
                # ADD TO CHESSBOARD OBJECT
                board.push_uci(uci_move)
                pgn = update_pgn(pgn, board, uci_move)
                pngBoard = board_to_png(board)
                # CASE IF GAME ENDS
                if analyse_position(board) != '':
                    return_code = analyse_position(board)
                    break
            # REFRESH REFERENCE FRAME TO NEW STABLE IMAGE
            stabilityCounter = 0
            referenceFrameWithBuffer = grayscaleFrameWithBuffer.astype(float)
        # TODO (maybe): redo lastSquares
        lastSquares = twoSquares
        liveTimeLeft = update_live_time(timeLeft, lastMoveTime, board.turn, time.time())
        if check_timeout(liveTimeLeft, board.turn) != '':
            return_code = check_timeout(liveTimeLeft, board.turn)
            break
        if AT_showEverything:
            # TODO: ONE SINGLE DISPLAY
            cv2.imshow('Original', smallFrameWithBuffer)
            cv2.imshow('Grid Board', gridFrame)
            cv2.imshow('SquaresAvg', squaresImage)
            cv2.imshow('Buffer', diffThreshWithBuffer)
        if AT_showAnything:
            # TODO: ONE SINGLE DISPLAY
            cv2.imshow('Original', smallFrameWithBuffer)
            printout = create_printout(liveTimeLeft, pgn, return_code)
            cv2.imshow('Digital Board', pngBoard)
            cv2.imshow('Live Game Analysis', printout)
            cv2.waitKey(1)
        if np.average(diffThreshWithBuffer) > AT_changeForQuit * 255:
            return_code = 'END 5: GAME ENDED BY RESIGNATION'
            break
    # FINISHING TOUCHES
    capture.release()
    cv2.destroyAllWindows()
    os.remove("board.png")
    os.remove("board.svg")
    return_value = pgn
    return return_code, return_value
