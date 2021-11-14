# TODO (maybe): MAYBE A FUNCTION FOR THIS TOO (check if python.chess has that)
squareNames = ['h8', 'h7', 'h6', 'h5', 'h4', 'h3', 'h2', 'h1',
               'g8', 'g7', 'g6', 'g5', 'g4', 'g3', 'g2', 'g1',
               'f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1',
               'e8', 'e7', 'e6', 'e5', 'e4', 'e3', 'e2', 'e1',
               'd8', 'd7', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1',
               'c8', 'c7', 'c6', 'c5', 'c4', 'c3', 'c2', 'c1',
               'b8', 'b7', 'b6', 'b5', 'b4', 'b3', 'b2', 'b1',
               'a8', 'a7', 'a6', 'a5', 'a4', 'a3', 'a2', 'a1']

castles = {frozenset({0, 8, 16, 24}): {24, 8}, frozenset({24, 32, 40, 56}): {24, 40},
           frozenset({7, 15, 23, 31}): {31, 15}, frozenset({31, 39, 47, 63}): {31, 47}}


# TODO (maybe): MAYBE A FUNCTION FOR THIS
enpassants = {frozenset({2, 3, 11}): {2, 11}, frozenset({10, 11, 19}): {10, 19}, frozenset({18, 19, 27}): {18, 27},
              frozenset({26, 27, 35}): {26, 35}, frozenset({34, 35, 43}): {34, 43}, frozenset({42, 43, 51}): {42, 51},
              frozenset({50, 51, 59}): {50, 59}, frozenset({10, 3, 11}): {10, 3}, frozenset({18, 11, 19}): {18, 11},
              frozenset({26, 19, 27}): {26, 19}, frozenset({34, 27, 35}): {34, 27}, frozenset({42, 35, 43}): {42, 35},
              frozenset({50, 43, 51}): {50, 43}, frozenset({58, 51, 59}): {58, 51}, frozenset({5, 4, 12}): {5, 12},
              frozenset({13, 12, 20}): {13, 20}, frozenset({21, 20, 28}): {21, 28}, frozenset({29, 28, 36}): {29, 36},
              frozenset({37, 36, 44}): {37, 44}, frozenset({45, 44, 52}): {45, 52}, frozenset({53, 52, 60}): {53, 60},
              frozenset({12, 4, 13}): {13, 4}, frozenset({20, 12, 21}): {21, 12}, frozenset({28, 20, 29}): {29, 20},
              frozenset({36, 28, 37}): {37, 28}, frozenset({44, 36, 45}): {45, 36}, frozenset({52, 44, 53}): {53, 44},
              frozenset({60, 52, 61}): {61, 52}}

# TODO: ROTATE BOARD INSTEAD
renumber = [56, 48, 40, 32, 24, 16, 8, 0, 57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36, 28, 20, 12, 4, 61, 53, 45, 37, 29, 21, 13, 5, 62, 54, 46, 38, 30, 22, 14, 6, 63, 55, 47, 39, 31, 23, 15, 7]
