import time
from ChessFunctions import *


lastMoveTime = time.time()

timeLeft, timeIncrement = time_control(5, increment=5)

pgn = ''
liveTimeLeft = {}
board = chess.Board()

referenceFrame = 0
referenceFrameWithBuffer = 0
lastSquares = []
stabilityCounter = 0
winner = ''
pngBoard = board_to_png(board)

while winner == '':
    twoSquaresText = input("squares: ").split(' ')
    twoSquares = [squareNames.index(square) for square in twoSquaresText]

    potential_uci_move = try_uci_move(twoSquares, board.turn, new_square_owner(board), squareNames)
    uci_move = uci_move_promotion_check(potential_uci_move, board.legal_moves)
    if uci_move != '':
        timeLeft[board.turn] = timeLeft[board.turn] - (time.time() - lastMoveTime) + timeIncrement
        lastMoveTime = time.time()

        board.push_uci(uci_move)
        pgn = update_pgn(pgn, board, uci_move)
        pngBoard = board_to_png(board)

        winner = analyse_position(board)
    else:
        print('ILLEGAL MOVE')

    lastSquares = twoSquares
    liveTimeLeft = update_live_time(timeLeft, lastMoveTime, board.turn, time.time())
    winner = check_timeout(liveTimeLeft, board.turn)


    cv2.imshow('Digital Board', pngBoard)
    # cv2.imshow('Buffer', diffThreshWithBuffer)

    printout = create_printout(liveTimeLeft, pgn, winner)
    cv2.imshow('Live Game Analysis', printout)

    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("r"):
        break


print(board.move_stack)
print(pgn)
time.sleep(100)
cv2.destroyAllWindows()
