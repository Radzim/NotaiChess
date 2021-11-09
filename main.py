import cv2
import numpy as np
import chess
import chess.svg
import chess.pgn
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from stockfish import Stockfish
import webbrowser

import warnings
import winsound
import time

winsound.Beep(300, 150)
lastMoveTime = time.time()

moveCount = 1
pgn = ''
increment = 10
timeLeft = {'W': 5 * 60, 'B': 5 * 60}
liveTimeLeft = {}

warnings.filterwarnings("ignore")

board = chess.Board()
game = chess.pgn.Game()

engine = Stockfish("C:/Users/radzi/OneDrive/Documents/stockfish_14_win_x64_avx2/stockfish_14_win_x64_avx2/stockfish_14_x64_avx2.exe")

squareNames = ['h8', 'h7', 'h6', 'h5', 'h4', 'h3', 'h2', 'h1',
               'g8', 'g7', 'g6', 'g5', 'g4', 'g3', 'g2', 'g1',
               'f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1',
               'e8', 'e7', 'e6', 'e5', 'e4', 'e3', 'e2', 'e1',
               'd8', 'd7', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1',
               'c8', 'c7', 'c6', 'c5', 'c4', 'c3', 'c2', 'c1',
               'b8', 'b7', 'b6', 'b5', 'b4', 'b3', 'b2', 'b1',
               'a8', 'a7', 'a6', 'a5', 'a4', 'a3', 'a2', 'a1']
squareOwner = ['B', 'B', 'N', 'N', 'N', 'N', 'W', 'W',
               'B', 'B', 'N', 'N', 'N', 'N', 'W', 'W',
               'B', 'B', 'N', 'N', 'N', 'N', 'W', 'W',
               'B', 'B', 'N', 'N', 'N', 'N', 'W', 'W',
               'B', 'B', 'N', 'N', 'N', 'N', 'W', 'W',
               'B', 'B', 'N', 'N', 'N', 'N', 'W', 'W',
               'B', 'B', 'N', 'N', 'N', 'N', 'W', 'W',
               'B', 'B', 'N', 'N', 'N', 'N', 'W', 'W']

toMove = 'W'

print('before')


def draw_grid(img_in, color=(0, 255, 0), thickness=1):
    img = np.array(img_in)
    h, w = img.shape
    rows, cols = 8, 8
    dy, dx = h / rows, w / cols
    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)
    return img


def calculate_all_area_averages(flash_matrix, fragment_dimensions):
    m_y, m_x = flash_matrix.shape
    f_y, f_x = fragment_dimensions
    horizontal_sum = np.cumsum(flash_matrix, axis=0)
    rectangular_sum = np.cumsum(horizontal_sum, axis=1)
    big_rectangle = rectangular_sum[f_y - 1:m_y, f_x - 1:m_x]
    tall_rectangle = rectangular_sum[0:m_y - f_y + 1, f_x - 1:m_x]
    long_rectangle = rectangular_sum[f_y - 1:m_y, 0:m_x - f_x + 1]
    small_rectangle = rectangular_sum[0:m_y - f_y + 1, 0:m_x - f_x + 1]
    center_rectangle = big_rectangle - tall_rectangle - long_rectangle + small_rectangle
    return np.divide(center_rectangle, f_y * f_x)


def square_centres(image):
    area_averages = calculate_all_area_averages(image, (image.shape[0] // 16, image.shape[0] // 16))
    square_centres_list = []
    for i in range(8):
        for j in range(8):
            square_centres_list.append(area_averages[image.shape[0] * i * 2 // 15][image.shape[1] * j * 2 // 15])
    return np.maximum(0, np.array(square_centres_list, dtype=np.float))


def two_squares(square_centres_list):
    castles = {frozenset({0, 8, 16, 24}): {24, 8}, frozenset({24, 32, 40, 56}): {24, 40}, frozenset({7, 15, 23, 31}): {31, 15}, frozenset({31, 39, 47, 63}): {31, 47}}
    enpassants = {frozenset({2, 3, 11}): {2, 11}, frozenset({10, 11, 19}): {10, 19}, frozenset({18, 19, 27}): {18, 27}, frozenset({26, 27, 35}): {26, 35}, frozenset({34, 35, 43}): {34, 43}, frozenset({42, 43, 51}): {42, 51}, frozenset({50, 51, 59}): {50, 59}, frozenset({10, 3, 11}): {10, 3}, frozenset({18, 11, 19}): {18, 11}, frozenset({26, 19, 27}): {26, 19}, frozenset({34, 27, 35}): {34, 27}, frozenset({42, 35, 43}): {42, 35}, frozenset({50, 43, 51}): {50, 43}, frozenset({58, 51, 59}): {58, 51}, frozenset({5, 4, 12}): {5, 12}, frozenset({13, 12, 20}): {13, 20}, frozenset({21, 20, 28}): {21, 28}, frozenset({29, 28, 36}): {29, 36}, frozenset({37, 36, 44}): {37, 44}, frozenset({45, 44, 52}): {45, 52}, frozenset({53, 52, 60}): {53, 60}, frozenset({12, 4, 13}): {13, 4}, frozenset({20, 12, 21}): {21, 12}, frozenset({28, 20, 29}): {29, 20}, frozenset({36, 28, 37}): {37, 28}, frozenset({44, 36, 45}): {45, 36}, frozenset({52, 44, 53}): {53, 44}, frozenset({60, 52, 61}): {61, 52}}
    highest_two_pred = np.where(square_centres_list > np.partition(square_centres_list, -3)[-3], 1, 0)
    highest_three_pred = np.where(square_centres_list > np.partition(square_centres_list, -4)[-4], 1, 0)
    highest_four_pred = np.where(square_centres_list > np.partition(square_centres_list, -5)[-5], 1, 0)
    highest_two = frozenset([i for i, val in enumerate(highest_two_pred) if val == 1])
    highest_three = frozenset([i for i, val in enumerate(highest_three_pred) if val == 1])
    highest_four = frozenset([i for i, val in enumerate(highest_four_pred) if val == 1])
    if highest_four in castles.keys():
        return list(castles[highest_four])
    if highest_three in enpassants.keys():
        return list(enpassants[highest_three])
    else:
        return list(highest_two)


def board_to_png(board_chess):
    svg_data = chess.svg.board(board=board_chess, size=1800)
    f = open("board.svg", "w")
    f.write(svg_data)
    f.close()
    drawing = svg2rlg("board.svg")
    renderPM.drawToFile(drawing, "board.png", fmt="PNG")
    return cv2.resize(cv2.imread("board.png"), (900, 900))


def analyse_position(chessboard):
    if chessboard.is_stalemate():
        return 'DRAW BY STALEMATE'
    if chessboard.is_insufficient_material():
        return 'DRAW BY INSUFFICIENT MATERIAL'
    if chessboard.can_claim_threefold_repetition():
        return 'DRAW BY THREEFOLD REPETITION'
    if chessboard.can_claim_fifty_moves():
        return 'DRAW BY FIFTY MOVE RULE'
    if chessboard.is_checkmate():
        if toMove == 'W':
            return 'WHITE WON BY CHECKMATE'
        if toMove == 'B':
            return 'BLACK WON BY CHECKMATE'
    return ''


def create_printout(live_time_left, pgn_text, engine_instance, win_text):
    img = np.zeros((900, 1000, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_content = 'BLACK ' + str(int(live_time_left['B']) // 60) + ':' + ('00' + str(int(live_time_left['B']) % 60))[-2:] + '    ' + str(int(live_time_left['W']) // 60) + ':' + ('00' + str(int(live_time_left['W']) % 60))[-2:] + ' WHITE'
    engine_eval = engine_instance.get_evaluation()
    if engine_eval['type'] == 'cp':
        engine_text = 'eval=' + str(engine_eval['value'] / 100)
    else:
        engine_text = 'eval=' + 'M' + str(engine_eval['value'])
    bottom_left_corner_of_text = (10, 50)
    font_scale = 2
    font_color = (255, 255, 255)
    line_type = 2
    cv2.putText(img, text_content, bottom_left_corner_of_text, font, font_scale, font_color, line_type)
    cv2.putText(img, win_text, (10, 150), font, font_scale, font_color, line_type)
    cv2.putText(img, engine_text, (10, 250), font, 1, font_color, line_type)
    for i, line in enumerate(pgn_text.split('   ')):
        cv2.putText(img, line, (10, 350 + 60 * i), font, 0.8, font_color, line_type)
    return img

print('after')

capture = cv2.VideoCapture("http://10.249.161.212:8080/video")

print('cap')

referenceFrame = 0
referenceFrameWithBuffer = 0
lastMoves = []
lastSquares = []
lastMove = 0
winner = ''
pngBoard = board_to_png(board)

renumber = [56, 48, 40, 32, 24, 16, 8, 0, 	57, 49, 41, 33, 25, 17, 9, 1, 	58, 50, 42, 34, 26, 18, 10, 2, 	59, 51, 43, 35, 27, 19, 11, 3, 	60, 52, 44, 36, 28, 20, 12, 4, 	61, 53, 45, 37, 29, 21, 13, 5, 	62, 54, 46, 38, 30, 22, 14, 6, 	63, 55, 47, 39, 31, 23, 15, 7]
lichessLink = board.fen().replace('/', '').replace('1', ' ').replace('2', '  ').replace('3', '   ').replace('4', '    ').replace('5', '     ').replace('6', '      ').replace('7', '       ').replace('8', '        ')
for i in range(64):
    if lichessLink[i] == ' ':
        squareOwner[renumber[i]] = 'N'
    elif lichessLink[i].isupper():
        squareOwner[renumber[i]] = 'W'
    elif lichessLink[i].islower():
        squareOwner[renumber[i]] = 'B'

# browserInstance = webbrowser.get(using=None)
# browserInstance.open(lichessLink, new=2)

while winner == '':
    ret, frame = capture.read()
    scaling = 0.5
    smallFrame = cv2.resize(frame, (int(1920 * scaling), int(1080 * scaling)), interpolation=cv2.INTER_AREA)
    croppedFrame = smallFrame[int(0 * scaling):int(1080 * scaling), int(420 * scaling):int(1500 * scaling)]  #toDel
    croppedFrameWithBuffer = smallFrame[int(0 * scaling):int(1080 * scaling), int(320 * scaling):int(1600 * scaling)]
    grayscaleFrame = croppedFrame[:, :, 2]  #toDel
    grayscaleFrameWithBuffer = croppedFrameWithBuffer[:, :, 2]
    if np.sum(referenceFrame) == 0:
        referenceFrame = grayscaleFrame.astype(float) #toDel
        referenceFrameWithBuffer = grayscaleFrameWithBuffer.astype(float)
    diffFrame = grayscaleFrame.astype(float) - referenceFrame
    diffFrameWithBuffer = grayscaleFrameWithBuffer.astype(float) - referenceFrameWithBuffer
    diffThreshWithBuffer = np.array(np.where(np.absolute(diffFrameWithBuffer) > 20, 255, 0), dtype=np.uint8)
    diffThreshWithBuffer = cv2.medianBlur(diffThreshWithBuffer, 5)
    diffThresh = diffThreshWithBuffer[int(0 * scaling):int(1080 * scaling), int(100 * scaling):int(1180 * scaling)]
    gridFrame = draw_grid(grayscaleFrame)

    squareCentres = square_centres(diffThresh)
    numpySquares = np.reshape(np.array(squareCentres, dtype=np.uint8), (-1, 8))
    squaresImage = cv2.resize(numpySquares, dsize=(540, 540), interpolation=cv2.INTER_NEAREST)
    twoSquares = two_squares(squareCentres)
    if max(squareCentres) > 10 and np.sum(diffThreshWithBuffer)-np.sum(diffThresh) < 500000 and twoSquares == lastSquares and len(twoSquares) == 2:
        lastMove += 1
    else:
        lastMove = 0
    if lastMove == 5:
        uci_move = ''
        if squareOwner[twoSquares[0]] == toMove:
            uci_move = squareNames[twoSquares[0]] + squareNames[twoSquares[1]]
        elif squareOwner[twoSquares[1]] == toMove:
            uci_move = squareNames[twoSquares[1]] + squareNames[twoSquares[0]]
        else:
            print(squareNames[twoSquares[0]] + squareNames[twoSquares[1]])
            print('NOT YOUR MOVE')
        if uci_move != '':
            print(uci_move)
            if chess.Move.from_uci(uci_move) in board.legal_moves or chess.Move.from_uci(uci_move+'q') in board.legal_moves:
                if chess.Move.from_uci(uci_move+'q') in board.legal_moves:
                    uci_move = uci_move+'q'
                if toMove == 'W':
                    pgn += str(moveCount) + '. ' + uci_move
                    moveCount += 1
                else:
                    pgn += ' ' + uci_move + ' '
                    if (moveCount - 1) % 5 == 0:
                        pgn += '  '
                winsound.Beep(200, 100)
                timeLeft[toMove] -= time.time() - lastMoveTime
                lastMoveTime = time.time()
                board.push_uci(uci_move)
                engine.make_moves_from_current_position([uci_move])
                winner = analyse_position(board)
                pngBoard = board_to_png(board)
                empty = uci_move[0:2]
                taken = uci_move[2:4]
                lichessLink = board.fen().replace('/', '').replace('1', ' ').replace('2', '  ').replace('3',
                                                                                                        '   ').replace(
                    '4', '    ').replace('5', '     ').replace('6', '      ').replace('7', '       ').replace('8',
                                                                                                              '        ')
                for i in range(64):
                    if lichessLink[i] == ' ':
                        squareOwner[renumber[i]] = 'N'
                    elif lichessLink[i].isupper():
                        squareOwner[renumber[i]] = 'W'
                    elif lichessLink[i].islower():
                        squareOwner[renumber[i]] = 'B'
                print(squareOwner)
                timeLeft[toMove] += increment
                if toMove == 'W':
                    toMove = 'B'
                else:
                    toMove = 'W'
                referenceFrameWithBuffer = grayscaleFrameWithBuffer.astype(float)
                lichessLink = board.fen()
                # browserInstance.open(lichessLink, new=0, autoraise=True)
            else:
                print('ILLEGAL MOVE')
        lastMove = 0

    lastSquares = twoSquares

    cv2.imshow('Original', smallFrame)
    cv2.imshow('Gridboard', gridFrame)
    cv2.imshow('Diffboard', diffThresh)
    cv2.imshow('SquaresAvg', squaresImage)
    cv2.imshow('Digital Board', pngBoard)
    cv2.imshow('Buffer', diffThreshWithBuffer)

    liveTimeLeft['B'] = timeLeft['B']
    liveTimeLeft['W'] = timeLeft['W']
    liveTimeLeft[toMove] -= time.time() - lastMoveTime
    if liveTimeLeft[toMove] <= 0:
        if toMove == 'W':
            winner = "BLACK WON ON TIME"
        else:
            winner = "WHITE WON ON TIME"
    printout = create_printout(liveTimeLeft, pgn, engine, winner)
    cv2.imshow('Live Game Analysis', printout)

    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("r"):
        break
    if cv2.waitKey(1) == ord("c"):
        referenceFrameWithBuffer = grayscaleFrameWithBuffer.astype(float)

winsound.Beep(200, 100)
winsound.Beep(200, 300)

print(pgn)
time.sleep(100)
capture.release()
cv2.destroyAllWindows()
