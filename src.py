import cv2
import numpy as np
from numpy.linalg import norm
import math

inputFile = "vidfin.mp4"    # input video
v = 4                       # speed


def CmList(fi, fjList):

    d = np.sqrt(fi.shape[0] * fi.shape[0] + fi.shape[1] * fi.shape[1])
    tc = 0.1 * d
    gamma = 0.5 * d

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(2 | 1, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(fi, mask=None, **feature_params)

    motionCostArr = np.zeros(fjList.size)
    it = 0

    for fj in fjList:

        try:
            p1, st, err = cv2.calcOpticalFlowPyrLK(fi, fj, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_old) > 10:
                h, status = cv2.findHomography(good_old, good_new)
                Cm = 0.0
                pt1 = np.ones([3, 1])
                pt2 = np.ones([3, 1])
                pt1 = np.asmatrix(pt1)
                pt2 = np.asmatrix(pt2)

                for x in xrange(0, good_old.shape[0]):
                    pt1[0, 0] = good_old[x, 0]
                    pt1[1, 0] = good_old[x, 1]
                    pt1[2, 0] = 1

                    pt1 = np.mat(h) * pt1
                    pt1[0, 0] /= pt1[2, 0]
                    pt1[1, 0] /= pt1[2, 0]
                    pt1[2, 0] = 1

                    pt2[0, 0] = good_new[x, 0]
                    pt2[1, 0] = good_new[x, 1]
                    pt2[2, 0] = 1

                    Cm += np.linalg.norm(pt2 - pt1)

                Cm /= good_old.shape[0]

                pt1[0, 0] = fi.shape[1] / 2
                pt1[1, 0] = fi.shape[0] / 2
                pt1[2, 0] = 1

                pt2 = np.mat(h) * pt1
                pt2[0, 0] /= pt2[2, 0]
                pt2[1, 0] /= pt2[2, 0]
                pt2[2, 0] = 1

                C0 = norm(pt2 - pt1)

                if Cm < tc:
                    motionCostArr[it] = C0
                else:
                    motionCostArr[it] = gamma
            else:
                motionCostArr[it] = gamma

            it += 1

        # Exception if homography not found
        except Exception, e:
            motionCostArr[it] = gamma
            it += 1
    return motionCostArr


def Cs(i, j, v):
    ts = 200
    return min(math.fabs((j - i) - v), ts)


def Ca(h, i, j):
    ta = 200
    return min(math.fabs((j - i) - (i - h)), ta)


def getMin(D, i, j, lamdaA):
    argmin = 0
    minVal = float("inf")
    l = []
    for k in xrange(1, D.shape[1] + 1):
        val = D[i - k, k - 1] + lamdaA * Ca(i - k, i, j)
        l.append(val)
        if val < minVal:
            minVal = val
            argmin = i - k
    return minVal, argmin

w = 25
g = 25
batchSize = 200
lamdaS = 200
lamdaA = 80
bss = 3

outptFile = 'hyper' + str(v) + '.mp4'
cap = cv2.VideoCapture(inputFile)
ret, frame = cap.read()
tempneigh = np.zeros([batchSize, frame.shape[0], frame.shape[1]], dtype=np.uint8)
fCount = 0
it = 0
MCList = []

while 1:
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if it == batchSize:
        motionCostArr = np.zeros(w)
        for i in xrange(0, batchSize - w):
            fi = tempneigh[i, :, :]
            fjList = tempneigh[range(i + 1, i + w + 1), :, :]
            MCList.append(CmList(fi, fjList))

        tempneigh[range(0, w), :, :] = tempneigh[range(batchSize - w, batchSize), :, :]
        it = w

    tempneigh[it, :, :] = frame
    it += 1
    temp = 0
    ret, frame = cap.read()
    while temp < bss and ret == False:
        ret, frame = cap.read()
        temp += 1
    print fCount
    if fCount > 2000:
        break
    fCount += 1

motionCostArr = np.zeros(w)
for i in xrange(0, it - w):
    fi = tempneigh[i, :, :]
    fjList = tempneigh[range(i + 1, i + w + 1), :, :]
    toAppend = CmList(fi, fjList)
    MCList.append(toAppend)

# Initialization
D = np.zeros([len(MCList), w])
T = np.zeros([len(MCList), w])

for i in xrange(0, g):
    for j in xrange(0, w):
        D[i, j] = MCList[i][j] + lamdaS * Cs(i, j + i + 1, v)

# First pass: populate Dv
for i in xrange(g, len(MCList)):
    for j in xrange(0, w):
        c = MCList[i][j] + lamdaS * Cs(i, j + i + 1, v)
        minVal, argmin = getMin(D, i, j + i + 1, lamdaA)
        D[i, j] = c + minVal
        T[i, j] = argmin

# Second pass: trace back min cost path
t = len(MCList)

s = -1
d = -1
minVal = float("inf")

for i in xrange(t - g, t):
    for j in xrange(0, w):
        if D[i, j] < minVal:
            minVal = D[i][j]
            s, d = i, j

p = [s + d + 1]
while s > g:
    p.append(s)
    b = T[s][d]
    d = int(s - (b + 1))
    s = int(b)

p.reverse()

cap = cv2.VideoCapture(inputFile)
ret, frame = cap.read()
fCount = 0
it = 0
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
size = (frame.shape[1], frame.shape[0])
vid = cv2.VideoWriter(outptFile, fourcc, 30, size, True)

while ret and it < len(p):
    if fCount == p[it]:
        vid.write(frame)
        it += 1
    temp = 0
    ret, frame = cap.read()
    while temp < bss and ret == False:
        ret, frame = cap.read()
        temp += 1
    print fCount
    fCount += 1

vid.release()
