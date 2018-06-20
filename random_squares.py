from random import random, seed
from math import pi, sin, cos, sqrt
import matplotlib.pyplot as plt

pi_2 = pi / 2

MINX = MINY = 0
MAXX = MAXY = 100
DEFAULT_SIDE = 15
DEFAULT_SAFETY_MARGIN = DEFAULT_SIDE * sqrt(1)

__global_generation_counter = 0


def get_func_deg1(p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    if x0 == x1:
        return None
    a = (y0 - y1)/(x0 - x1)
    b = y0 - x0 * a
    return lambda x: a * x + b


def is_point_in_square(p, sq):
    x, y = p
    p0, p1, p2, p3 = sq
    side_func0 = get_func_deg1(p0, p1)
    side_func1 = get_func_deg1(p1, p2)
    side_func2 = get_func_deg1(p2, p3)
    side_func3 = get_func_deg1(p3, p0)
    if not side_func0 or not side_func1 or not side_func2 or not side_func3:
        xmin = min(p0[0], p2[0])
        xmax = max(p0[0], p2[0])
        ymin = min(p0[1], p2[1])
        ymax = max(p0[1], p2[1])
        return xmin <= x <= xmax and ymin <= y <= ymax
    return ((y - side_func0(x)) * (y - side_func2(x))) <= 0 and \
           ((y - side_func1(x)) * (y - side_func3(x))) <= 0


def squares_overlap(square0, square1):
    for p0 in square0:
        if is_point_in_square(p0, square1):
            return True
    for p1 in square1:
        if is_point_in_square(p1, square0):
            return True
    xc0 = (square0[0][0] + square0[2][0]) / 2
    yc0 = (square0[0][1] + square0[2][1]) / 2
    if is_point_in_square((xc0, yc0), square1):
        return True
    # The "reverse center check" not needed, since squares are congruent
    """
    xc1 = (square1[0][0] + square1[2][0]) / 2
    yc1 = (square1[0][1] + square1[2][1]) / 2
    if is_point_in_square((xc1, yc1), square0):
        return True
    """
    return False


def __generation_monitor():
    global __global_generation_counter
    __global_generation_counter += 1


def generate_random_point(minx=MINX, miny=MINY, maxx=MAXX, maxy=MAXY, safety_margin=DEFAULT_SAFETY_MARGIN):
    if maxx - minx < 2 * safety_margin or maxy - miny < 2 * safety_margin:
        print("MUEEE")
        safety_margin = 0
    x = safety_margin + random() * (maxx - minx - 2 * safety_margin)
    y = safety_margin + random() * (maxy - miny - 2 * safety_margin)
    __generation_monitor()
    return x, y


def generate_random_angle(max_val=pi_2):
    return random() * max_val


def generate_random_square(side=DEFAULT_SIDE, squares_to_avoid=()):
    while True:
        restart = False
        x0, y0 = generate_random_point()

        angle = generate_random_angle()
        x1 = x0 + side * cos(angle)
        y1 = y0 + side * sin(angle)

        angle += pi_2
        x2 = x1 + side * cos(angle)
        y2 = y1 + side * sin(angle)

        angle += pi_2
        x3 = x2 + side * cos(angle)
        y3 = y2 + side * sin(angle)

        ret = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        for square in squares_to_avoid:
            if squares_overlap(ret, square):
                restart = True
        if restart:
            continue
        return ret


def square_to_plot(square):
    xs, ys = zip(square[0], square[1], square[2], square[3])
    return xs + (xs[0],), ys + (ys[0],)


def main(MAX_SQUARES):
    seed()
    squares = list()
    allow_overlapping = True # CHANGE to True to allow square to overlap
    for _ in range(MAX_SQUARES):
        if allow_overlapping:
            square = generate_random_square()
        else:
            square = generate_random_square(squares_to_avoid=squares)
        squares.append(square)
    plot_squares = tuple()
    for sq in squares:
        plot_squares += square_to_plot(sq)
    print("STATS:\n    Squares: {}\n    Allow  overlapping: {}\n    Generated values: {}".format(MAX_SQUARES, allow_overlapping, __global_generation_counter))
    plt.close('all')
    plt.plot(*plot_squares)
    plt.axis([MINX, MAXX, MINY, MAXY])
    plt.axes().set_aspect('equal')
    plt.show()
    return squares

if __name__ == "__main__":
    generated_squares = main(MAX_SQUARES=15)