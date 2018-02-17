# This code contains helper functions used for simplex drawing.

import numpy as np

def linear_coeff(x1, y1, x2, y2):
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return [a, b, c]

def line_inter(l1, l2):
    tmp = np.array([l1, l2])
    mat = tmp[:, :2]
    vec = -1 * tmp[:, 2]
    return np.linalg.solve(mat, vec)

def plane_inter_2d(plane_1, plane_2):
    norm_diff = plane_1.norm - plane_2.norm
    [a, b, c] = norm_diff.tolist()
    d = plane_1.d - plane_2.d
    # plane_diff = Plane(a, b, c, d)
    return [a, b, -d]

def sample_from_polygon(polygon):
    polyg = polygon[:3, :]
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    p = (1 - np.sqrt(r1)) * polyg[0, :] + \
        (np.sqrt(r1) * (1 - r2)) * polyg[1, :] + \
        (np.sqrt(r1) * r2) * polyg[2, :]
    return p

def split_polygon(polygon, line):
    it = 0
    phase = 1
    n_edges = polygon.shape[0]
    new_poly_1 = []
    new_poly_2 = []

    while phase <= 3 and it <= 2 * n_edges:
        curr_edge = polygon[it % n_edges, :, :]
        [x_min, y_min] = np.min(curr_edge, axis=0).tolist()
        [x_max, y_max] = np.max(curr_edge, axis=0).tolist()
        [[x1, y1], [x2, y2]] = curr_edge

        it += 1

        parallel = False
        tmp_line = linear_coeff(x1, y1, x2, y2)

        try:
            int_point = line_inter(tmp_line, line)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                parallel = True
                pass
            else:
                raise

        if parallel:
            if phase == 2:
                new_poly_1.append(curr_edge)
            if phase == 3:
                new_poly_2.append(curr_edge)
            continue

        no_intersection = int_point[0] < x_min or \
                          int_point[0] > x_max or \
                          int_point[1] < y_min or \
                          int_point[1] > y_max

        if phase == 1:
            no_intersection = no_intersection or \
            np.linalg.norm(int_point - curr_edge[1, :]) == 0
        if phase == 3:
            no_intersection = no_intersection or \
            np.linalg.norm(int_point - curr_edge[0, :]) == 0

        if no_intersection:
            if phase == 2:
                new_poly_1.append(curr_edge)
            if phase == 3:
                new_poly_2.append(curr_edge)
            continue

        if phase == 1:
            to_app = np.array([int_point, curr_edge[1, :]])
            new_poly_1.append(to_app)

        if phase == 2:
            to_app_1 = np.array([curr_edge[0, :], int_point])
            to_app_2 = np.array([int_point, curr_edge[1, :]])
            new_poly_1.append(to_app_1)

            if np.linalg.norm(int_point - curr_edge[1, :]) != 0:
                new_poly_2.append(to_app_2)

        if phase == 3:
            to_app = np.array([curr_edge[0, :], int_point])
            new_poly_2.append(to_app)

        phase += 1

    new_poly_1 = np.array(new_poly_1)
    new_poly_2 = np.array(new_poly_2)

    ### IMPORTANT ###
    # This function fails if line touches only one vertex,
    # overlaps with one of the sides of polygon or
    # does not intersect polygon at all.

    return close_polygon_edges(new_poly_1), \
           close_polygon_edges(new_poly_2)

def close_polygon_edges(polygon):
    last_edge = np.array([[polygon[polygon.shape[0] - 1, 1, :], polygon[0, 0, :]]])
    closed_polygon = np.append(polygon, last_edge, axis=0)
    return closed_polygon

def edges_to_vertices(polygon):
    return polygon[:, 0, :]

def project_polygon(points):
    tripts = np.zeros((points.shape[0],2))
    for i in range(points.shape[0]):

        x = 1.0 / 2
        y = 1.0 / (2 * np.sqrt(3))

        p1 = points[i, 0]
        x = x - (p1 * np.cos(np.pi / 6) / np.sqrt(3))
        y = y - (p1 * np.sin(np.pi / 6) / np.sqrt(3))

        p2 = points[i, 1]
        x = x + (p2 * np.cos(np.pi / 6) / np.sqrt(3))
        y = y - (p2 * np.sin(np.pi / 6) / np.sqrt(3))

        p3 = 1 - points[i, 0] - points[i, 1]
        y = y + (p3 / np.sqrt(3))

        tripts[i,:] = (x,y)

    return tripts

class Plane:

    def __init__(self, *args):
        if len(args) == 3:
            (a, b, c) = args
            vec_1 = a - b
            vec_2 = c - b
            self.norm = np.cross(vec_1, vec_2)
            self.d = np.dot(self.norm, a)
        elif len(args) == 4:
            (a, b, c, d) = args
            self.norm = np.array([a, b, c])
            self.d = d
        else:
            print('Bad arguments for constructor')
