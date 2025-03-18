import numpy as np
import matplotlib.pyplot as plt

stuff = np.load('../data/some_corresp.npz')
p1, p2 = stuff["pts1"], stuff["pts2"]
pic1 = plt.imread('../data/im1.png')
pic2 = plt.imread('../data/im2.png')

def camera2(E):
    a, b, c = np.linalg.svd(E)
    avg = b[:2].mean()
    E = a.dot(np.array([[avg,0,0], [0,avg,0], [0,0,0]])).dot(c)
    a, b, c = np.linalg.svd(E)
    w = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    if np.linalg.det(a.dot(w).dot(c))<0:
        w = -w
    res = np.zeros([3,4,4])
    res[:,:,0] = np.concatenate([a.dot(w).dot(c), a[:,2].reshape([-1, 1])/abs(a[:,2]).max()], axis=1)
    res[:,:,1] = np.concatenate([a.dot(w).dot(c), -a[:,2].reshape([-1, 1])/abs(a[:,2]).max()], axis=1)
    res[:,:,2] = np.concatenate([a.dot(w.T).dot(c), a[:,2].reshape([-1, 1])/abs(a[:,2]).max()], axis=1)
    res[:,:,3] = np.concatenate([a.dot(w.T).dot(c), -a[:,2].reshape([-1, 1])/abs(a[:,2]).max()], axis=1)
    return res

def refine_F(F):
    a, b, c = np.linalg.svd(F)
    b[-1] = 0
    return a @ np.diag(b) @ c

def eight_point(p1, p2, m):
    t = np.array([[1/m, 0, 0], [0, 1/m, 0], [0, 0, 1]])
    n1 = np.column_stack((p1, np.ones(len(p1)))) @ t
    n2 = np.column_stack((p2, np.ones(len(p2)))) @ t
    x1, y1 = n1[:, 0], n1[:, 1]
    x2, y2 = n2[:, 0], n2[:, 1]
    a = np.column_stack((x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, np.ones_like(x1)))
    _, _, v = np.linalg.svd(a)
    f = v[-1].reshape(3, 3)
    f = refine_F(f)
    return t.T @ f @ t

cam = np.load('../data/intrinsics.npz')
m = max(pic1.shape)
f = eight_point(p1, p2, m)
print(f)

def epipolar_correspondences(pic1, pic2, f, pts):
    res = []
    for pt in pts:
        x, y = pt
        l = f @ np.array([x, y, 1])
        a, b, c = l
        xr = np.arange(pic2.shape[1])
        yr = -(a * xr + c) / b
        v = (yr >= 0) & (yr < pic2.shape[0])
        xr, yr = xr[v], yr[v].astype(int)
        best = None
        err = float('inf')
        for x2, y2 in zip(xr, yr):
            if 0 <= y2 < pic2.shape[0] and 0 <= x2 < pic2.shape[1]:
                p1 = pic1[max(0, y-2):y+3, max(0, x-2):x+3]
                p2 = pic2[max(0, y2-2):y2+3, max(0, x2-2):x2+3]
                if p1.shape == p2.shape:
                    e = np.sum((p1.astype(float) - p2.astype(float)) ** 2)
                    if e < err:
                        err = e
                        best = (x2, y2)
        res.append(best if best else (x, y))
    return np.array(res)

temp = np.load('../data/temple_coords.npz')
p1_temp = temp["pts1"]
p2_temp = epipolar_correspondences(pic1, pic2, f, p1_temp)

k1 = cam['K1']
k2 = cam['K2']
e = k2.T @ f @ k1
print(e)

def triangulate(P1, pts1, P2, pts2):
    res = []
    for p1, p2 in zip(pts1, pts2):
        a = np.array([
            p1[0] * P1[2, :] - P1[0, :],
            p1[1] * P1[2, :] - P1[1, :],
            p2[0] * P2[2, :] - P2[0, :],
            p2[1] * P2[2, :] - P2[1, :]
        ])
        _, _, v = np.linalg.svd(a)
        x = v[-1, :4]
        x = x / x[3]
        res.append(x[:3])
    return np.array(res)

p1 = k1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
m = camera2(e)

best_M = None
max_depth = -float('inf')
best_pts = None

for i in range(4):
    p2 = k2 @ m[:, :, i]
    pts = triangulate(p1, p1_temp, p2, p2_temp)
    depth = 0
    for pt in pts:
        if (p1 @ np.append(pt, 1) > 0)[2] and (p2 @ np.append(pt, 1) > 0)[2]:
            depth += 1
    if depth > max_depth:
        max_depth = depth
        best_M = m[:, :, i]
        besti = i
        best_pts = pts

p2 = k2 @ best_M
pts = best_pts
print(besti)

def plot_3d_points(pts, angles):
    for e, a in angles:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
        ax.view_init(elev=e, azim=a)
        plt.title(f'View')
        plt.show()
