# This code is used for visualizing belief model constraints.

import numpy as np
import pkg.tools.Solver2 as s2
import matplotlib.ticker as mt
import matplotlib.pyplot as plt
import matplotlib.patches as pts
from pkg.tools.Solver2 import Plane

def print_poly(polys):
    for poly in polys:
        print(s2.edges_to_vertices(poly))

def feed_payment_matrix(matrix):
    ps = np.zeros((3,3,3))
    ps[0] = feed_payment_matrix_helper(0, matrix)
    ps[1] = feed_payment_matrix_helper(1, matrix)
    ps[2] = feed_payment_matrix_helper(2, matrix)

    return ps

def feed_payment_matrix_helper(report, matrix, belief_1=np.array([0.1, 0.2]),
                                               belief_2=np.array([0.1, 0.7]),
                                               belief_3=np.array([0.7, 0.1])):

    a = belief_1[0] * matrix[report, 0] + belief_1[1] * matrix[report, 1] + \
        (1 - belief_1[0] - belief_1[1]) * matrix[report, 2]
    b = belief_2[0] * matrix[report, 0] + belief_2[1] * matrix[report, 1] + \
        (1 - belief_2[0] - belief_2[1]) * matrix[report, 2]
    c = belief_3[0] * matrix[report, 0] + belief_3[1] * matrix[report, 1] + \
        (1 - belief_3[0] - belief_3[1]) * matrix[report, 2]

    a = np.append(belief_1, a)
    b = np.append(belief_2, b)
    c = np.append(belief_3, c)

    return np.array([a, b, c])

def generate_3_lines(points):

    [a, b, c] = points[0]
    plane_0 = Plane(a, b, c)

    [a, b, c] = points[1]
    plane_1 = Plane(a, b, c)

    [a, b, c] = points[2]
    plane_2 = Plane(a, b, c)

    z_o_line = np.around(s2.plane_inter_2d(plane_0, plane_1), decimals=6)
    z_t_line = np.around(s2.plane_inter_2d(plane_0, plane_2), decimals=6)
    o_t_line = np.around(s2.plane_inter_2d(plane_1, plane_2), decimals=6)

    return z_o_line, z_t_line, o_t_line

def reward(payment_matrix, report, belief):
    reward = belief[0] * payment_matrix[report, 0] + \
             belief[1] * payment_matrix[report, 1] + \
             (1 - belief[0] - belief[1]) * payment_matrix[report, 2]

    return reward

def produce_patches(pay_mat):

    ps = feed_payment_matrix(pay_mat)
    z_o_line, z_t_line, o_t_line = generate_3_lines(ps)

    prob_triangle = np.array([[[0, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, 0]]])
    piece_a, piece_b = s2.split_polygon(prob_triangle, z_o_line)
    point_in_piece_a = s2.sample_from_polygon(s2.edges_to_vertices(piece_a))

    if reward(pay_mat, 0, point_in_piece_a) > reward(pay_mat, 1, point_in_piece_a):
        piece_a_zero = True
    else:
        piece_a_zero = False

    if piece_a_zero:
        piece_1, piece_2 = s2.split_polygon(piece_a, z_t_line)
        piece_3, piece_4 = s2.split_polygon(piece_b, o_t_line)
    else:
        piece_1, piece_2 = s2.split_polygon(piece_b, z_t_line)
        piece_3, piece_4 = s2.split_polygon(piece_a, o_t_line)

    point_in_piece_1 = s2.sample_from_polygon(s2.edges_to_vertices(piece_1))
    point_in_piece_3 = s2.sample_from_polygon(s2.edges_to_vertices(piece_3))

    if reward(pay_mat, 0, point_in_piece_1) > reward(pay_mat, 2, point_in_piece_1):
        patch_of_zero = piece_1
    else:
        patch_of_zero = piece_2

    if reward(pay_mat, 1, point_in_piece_3) > reward(pay_mat, 2, point_in_piece_3):
        patch_of_one = piece_3
    else:
        patch_of_one = piece_4

    patches = [s2.edges_to_vertices(prob_triangle),
               s2.edges_to_vertices(patch_of_zero),
               s2.edges_to_vertices(patch_of_one)]

    return patches

# Draws belief model constraints based on payment matrix
# parameters: pay_mat - 3 by 3 ndarray
#             save - whether the simplex should be drawn or saved to a .png file
#             savename - used if save=True, defines savename of a file
def draw_simplex(pay_mat, save=False, savename='simplex'):

    patches = produce_patches(pay_mat)
    # print(patches)
    final_patches = list(map(s2.project_polygon, patches))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.18, 1)
    ax.xaxis.set_major_locator(mt.NullLocator())
    ax.yaxis.set_major_locator(mt.NullLocator())
    ax.text(-0.06, -0.05, '0')
    ax.text(1.03, -0.05, '1')
    ax.text(0.49, np.sqrt(3) / 2 + 0.03, '2')

    fig.subplots_adjust(bottom = 0.1)
    fig.subplots_adjust(top = 0.9)
    fig.subplots_adjust(left = 0.1)
    fig.subplots_adjust(right = 1)

    p_0 = pts.Polygon(final_patches[0], ec='black', fc='sandybrown')
    p_1 = pts.Polygon(final_patches[1], ec='black', fc='indianred')
    p_2 = pts.Polygon(final_patches[2], ec='black', fc='lightseagreen')
    ax.add_patch(p_0)
    ax.add_patch(p_1)
    ax.add_patch(p_2)
    plt.legend([p_1, p_2, p_0], ['TR0', 'TR1', 'TR2'])

    if save:
        plt.savefig(savename + '.png')
    else:
        plt.show()
