import chess.pgn
import chess.svg
import cv2.cv2 as cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ChessData import *
from alanTuning import *


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
    img = np.zeros((900, 900, 3), np.uint8)
    img = 255-img
    board_string = str(board_chess)
    board_string = board_string.replace('K', '♔')
    board_string = board_string.replace('Q', '♕')
    board_string = board_string.replace('R', '♖')
    board_string = board_string.replace('B', '♗')
    board_string = board_string.replace('N', '♘')
    board_string = board_string.replace('P', '♙')
    board_string = board_string.replace('k', '♚')
    board_string = board_string.replace('q', '♛')
    board_string = board_string.replace('r', '♜')
    board_string = board_string.replace('b', '♝')
    board_string = board_string.replace('n', '♞')
    board_string = board_string.replace('p', '♟')
    board_string = board_string.replace('.', ' ')
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("DejaVuSansMono.ttf", 50)
    draw.text((50, 80), str(board_string), fill=(0, 0, 0, 255), font=font)
    img = np.array(img_pil)
    return img


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
        if chessboard.turn == chess.WHITE:
            return 'WHITE WON BY CHECKMATE'
        if chessboard.turn == chess.BLACK:
            return 'BLACK WON BY CHECKMATE'
    return ''


def create_printout(live_time_left, pgn_text, win_text):
    img = np.zeros((900, 1000, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_content = 'BLACK ' + str(int(live_time_left[chess.BLACK]) // 60) + ':' + ('00' + str(int(live_time_left[chess.BLACK]) % 60))[-2:] + '    ' + str(int(live_time_left[chess.WHITE]) // 60) + ':' + ('00' + str(int(live_time_left[chess.WHITE]) % 60))[-2:] + ' WHITE'
    bottom_left_corner_of_text = (10, 50)
    font_scale = 2
    font_color = (255, 255, 255)
    line_type = 2
    cv2.putText(img, text_content, bottom_left_corner_of_text, font, font_scale, font_color, line_type)
    cv2.putText(img, win_text, (10, 150), font, font_scale, font_color, line_type)
    for i, line in enumerate(pgn_text.split('   ')):
        cv2.putText(img, line, (10, 250 + 60 * i), font, 0.8, font_color, line_type)
    return img


def new_square_owner(board):
    one_to_one_fen = board.fen().replace('/', '').replace('1', '_' * 1).replace('2', '_' * 2).replace('3',
                                                                                                      '_' * 3).replace(
        '4', '_' * 4).replace('5', '_' * 5).replace('6', '_' * 6).replace('7', '_' * 7).replace('8', '_' * 8)
    square_owner = np.empty(64)
    for i in range(64):
        if one_to_one_fen[i] == '_':
            square_owner[renumber[i]] = None
        elif one_to_one_fen[i].isupper():
            square_owner[renumber[i]] = chess.WHITE
        elif one_to_one_fen[i].islower():
            square_owner[renumber[i]] = chess.BLACK
    return square_owner


def time_control(start, increment=0):
    return {chess.WHITE: start * 60, chess.BLACK: start * 60}, increment


# USE BOARD.STACK!
def update_pgn(pgn_in, board, uci_move):
    move_count = len(board.move_stack) // 2 + 1
    pgn = pgn_in
    if board.turn == chess.BLACK:
        pgn += str(move_count) + '. ' + uci_move
        move_count += 1
    else:
        pgn += ' ' + uci_move + ' '
        if (move_count - 1) % 5 == 0:
            pgn += '  '
    return pgn


def try_uci_move(two_squares_list, turn, live_square_owner, square_names):
    if live_square_owner[two_squares_list[0]] == turn:
        return square_names[two_squares_list[0]] + square_names[two_squares_list[1]]
    elif live_square_owner[two_squares_list[1]] == turn:
        return square_names[two_squares_list[1]] + square_names[two_squares_list[0]]
    else:
        return ''


def uci_move_promotion_check(uci_move, legal_moves):
    if uci_move != '':
        if chess.Move.from_uci(uci_move + 'q') in legal_moves:
            return uci_move + 'q'
        if chess.Move.from_uci(uci_move) in legal_moves:
            return uci_move
    return ''


def detect_stable_move(stable_count, square_centres_list, diff_thresh_wb, diff_thresh, two_squares_list, last_squares):
    condition_a = max(square_centres_list) > AT_minimumChangeForMove * 255
    condition_b = (np.sum(diff_thresh_wb) - np.sum(diff_thresh)) / (np.size(diff_thresh_wb)-np.size(diff_thresh)) < AT_bufferProtection * 255
    condition_c = two_squares_list == last_squares
    condition_d = len(two_squares_list) == 2
    if condition_a and condition_b and condition_c and condition_d:
        return stable_count + 1
    else:
        return 0


def update_live_time(time_left, last_move_time, turn, time_time):
    time_left_copy = dict(time_left)
    time_left_copy[chess.BLACK] = time_left[chess.BLACK]
    time_left_copy[chess.WHITE] = time_left[chess.WHITE]
    time_left_copy[turn] -= time_time - last_move_time
    return time_left_copy


def check_timeout(live_time_left, turn):
    if live_time_left[turn] <= 0:
        if turn == chess.WHITE:
            return "BLACK WON ON TIME"
        else:
            return "WHITE WON ON TIME"
    return ''
