import numpy as np
from scipy.ndimage import filters
from bidict import bidict


def image_gradient(image, sigma):
    gx = filters.gaussian_filter(image, sigma, order=[0, 1])
    gy = filters.gaussian_filter(image, sigma, order=[1, 0])
    return gx, gy


def compute_edge_points(partial_gradients):
    gx, gy = partial_gradients
    rows, cols = gx.shape
    edges = []

    def mag(y, x):
        return np.hypot(gx[y, x], gy[y, x])

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):

            center_mag = mag(y, x)
            left_mag = mag(y, x - 1)
            right_mag = mag(y, x + 1)
            top_mag = mag(y - 1, x)
            bottom_mag = mag(y + 1, x)

            theta_x, theta_y = 0, 0
            if (left_mag < center_mag >= right_mag) and abs(gx[y, x]) >= abs(gy[y, x]):
                theta_x = 1
            elif (top_mag < center_mag >= bottom_mag) and abs(gx[y, x]) <= abs(gy[y, x]):
                theta_y = 1
            if theta_x != 0 or theta_y != 0:
                a = mag(y - theta_y, x - theta_x)
                b = mag(y, x)
                c = mag(y + theta_y, x + theta_x)
                lamda = (a - c) / (2 * (a - 2 * b + c))
                ex = x + lamda * theta_x
                ey = y + lamda * theta_y
                edges.append((ex, ey))
    return np.asarray(edges)


def chain_edge_points(edges, g):
    gx, gy = g

    def neighborhood(p, max_dist):
        px, py = p
        for e in edges:
            ex, ey = e
            if abs(ex - px) <= max_dist and abs(ey - py) <= max_dist:
                yield e

    def gval(p):
        px, py = [int(x) for x in p]
        return [gx[py, px], gy[py, px]]

    def envec(e, n):
        return np.asanyarray(n) - np.asanyarray(e)

    def perp(v):
        x, y = gval(e)
        return np.asanyarray([y, -x])

    def dist(a, b):
        return np.hypot(*(np.subtract(b, a)))

    def h(float_array):
        return tuple(x for x in float_array)

    links = bidict()
    for e in edges:
        nf = [n for n in neighborhood(e, 2) if
              np.dot(gval(e), gval(n)) > 0 and np.dot(envec(e, n), perp(gval(e))) > 0]
        nb = [n for n in neighborhood(e, 2) if
              np.dot(gval(e), gval(n)) > 0 and np.dot(envec(e, n), perp(gval(e))) < 0]

        f_idx = np.argmin([dist(e, n) for n in nf])
        f = h(nf[f_idx])
        b_idx = np.argmin([dist(e, n) for n in nb])
        b = h(nb[b_idx])
        if f not in links.inv:
            links[h(e)] = f
        else:
            a = links.inv[f]
            if dist(e, f) < dist(a, f):
                del links.inv[f]
                del links[h(e)]
                links[h(e)] = f
        if b not in links:
            links[b] = h(e)
        else:
            a = links[b]
            if dist(b, e) < dist(b, a):
                del links[b]
                del links.inv[h(e)]
                links[h(e)] = f
    return links


class CurvePoint(object):
    __slots__ = ['x', 'y', 'valid']

    def __init__(self, x, y, valid):
        self.x = x
        self.y = y
        self.valid = valid

    def __hash__(self):
        return hash((self.x,self.y))


def thresholds_with_hysteresis(edges, links, grads, high_threshold, low_threshold):
    edges = [CurvePoint(e[0], e[1], valid=False) for e in edges]
    gx, gy = grads

    def mag(x, y):
        x, y = int(x), int(y)
        return np.hypot(gx[y, x], gy[y, x])

    def h(float_array):
        return tuple(x for x in float_array)

    chains = []
    for e in edges:
        if not e.valid and mag(e.x,e.y) >= high_threshold:
            forward = []
            backward = []
            e.valid = True
            f = h([e.x,e.y])
            while h(f) in links:
                n = links[h(f)]
                if not n.valid and mag(n.x,n.y) >= low_threshold:
                    n.valid = True
                    f = n
                    forward.append(f)
            b = h([e.x,e.y])
            while h(b) in links.inv:
                n = links.inv[h(b)]
                if not n.valid and mag(n.x,n.y) >= low_threshold:
                    n.valid = True
                    b = n
                    backward.insert(0,b)
            chain = backward + [h([e.x,e.y])] + forward
            chains.append(np.asarray([(c.x,c.y) for c in chain ]))
    return chains

if __name__ == '__main__':
    import numpy as np
    from scipy.ndimage import filters
    import cv2

    pad = 20
    circle = cv2.imread("./kreis.png", 0)
    I = np.zeros((circle.shape[0] + 2 * pad, circle.shape[1] + 2 * pad), dtype=np.uint8) + 255
    I[pad:circle.shape[0] + pad, pad:circle.shape[1] + pad] = circle
    I = I.astype(np.float32)

    grads = image_gradient(I, 2.0)
    edgels = compute_edge_points(grads)
    links = chain_edge_points(edgels, grads)
    chains = thresholds_with_hysteresis(edgels, links, grads, 1, 0.1)
