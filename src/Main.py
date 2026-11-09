from PIL import Image
import numpy as np
import random


def get_strips(n):
    pixels = np.array(Image.open(f"out/strips/strip-{n:02d}.png").convert("RGB"))
    binary = (np.mean(pixels, axis=2) > 127).astype(np.uint8)
    return binary[:, 0], binary[:, -1]


def shuffle(arr):
    return random.sample(arr, len(arr))


def get_dists(dist, xs, ys):
    return [dist(r1, l2) for (_, r1), (l2, _) in zip(xs, ys)]


def bench(dist, strips):
    xs = get_dists(dist, strips, strips[1:])
    ys = get_dists(dist, strips, shuffle(strips))
    print(f"first: {sum(xs)} second: {sum(ys)}")
    return sum(xs) / sum(ys)


def dist2(left, right):
    def signature(arr):
        changes = np.where(np.diff(np.r_[0, arr, 0]) != 0)[0]
        return np.fromiter(
            (start + end for start, end in zip(changes[::2], changes[1::2])),
            dtype=int,
        )

    sl = signature(left)
    sr = signature(right)
    # k = min(sl.shape[0], sr.shape[0]) // 3 + 1
    k = 20
    difs = np.fromiter((np.min(np.abs(sl - x)) for x in sr), dtype=int)
    return sum(np.sort(difs)[:k])


def dist1(left, right):
    return np.count_nonzero(left - right)


# random.seed(645)
strips = [get_strips(n) for n in range(35)]
print(bench(dist1, strips))
print(bench(dist2, strips))
# print(np.unique(left, axis=0, return_counts=True))
# for x in left:
#     print(x)

# Image.fromarray(binary * 255).save("binary.png")
