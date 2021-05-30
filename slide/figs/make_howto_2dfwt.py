import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../implementation'))
import fwt
import numpy as np

HAAR_SCALING_COEF = [0.707106781186547, 0.707106781186547]

def _minmax_scale(vec, maxscale):
    """ 入力ベクトル値の範囲が[0, maxscale]になるように調整 """
    minval = np.min(vec)
    maxval = np.max(vec)
    # ゼロ除算対策のため、非ゼロ要素だけ除算
    div = np.divide(vec - minval, maxval - minval, where=((vec - minval) != 0))
    return maxscale * div


def _minmax_clip(vec, maxscale):
    return np.where(vec >= maxscale, maxscale, vec)

if __name__ == "__main__":
    from PIL import Image

    scaling_coef = HAAR_SCALING_COEF

    # 画像読み込み（簡略化のためグレスケ変換）
    img = Image.open('pattern.png').convert("L")
    original = np.asarray(img)
    pyramid = original.copy()
    imgpyramid = original.copy()

    # 入力を低域（左）と高域（右）に分解
    src_len = original.shape[0]
    half_src_len = src_len // 2
    for j in range(src_len):
        src_l, src_h = fwt.fwt1d(original[j, :], scaling_coef)
        pyramid[j, 0:half_src_len] = src_l
        pyramid[j, half_src_len:src_len] = src_h
        imgpyramid[j, 0:half_src_len] = _minmax_clip(src_l, 255)
        imgpyramid[j, half_src_len:src_len] = _minmax_clip(src_h, 255)

    Image.fromarray(imgpyramid.astype(np.uint8)).save("vertical_pyramid.png")

    for j in range(half_src_len):
        src_l, src_h = fwt.fwt1d(pyramid[:, j], scaling_coef)
        pyramid[0:half_src_len, j] = src_l
        pyramid[half_src_len:src_len, j] = src_h
        imgpyramid[0:half_src_len, j] = _minmax_clip(src_l, 255)
        imgpyramid[half_src_len:src_len, j] = _minmax_clip(src_h, 255)
        src_l, src_h = fwt.fwt1d(pyramid[:, half_src_len + j], scaling_coef)
        pyramid[0:half_src_len, half_src_len + j] = src_l
        pyramid[half_src_len:src_len, half_src_len + j] = src_h
        imgpyramid[0:half_src_len, half_src_len + j] = _minmax_clip(src_l, 255)
        imgpyramid[half_src_len:src_len, half_src_len + j] = _minmax_clip(src_h, 255)

    Image.fromarray(imgpyramid.astype(np.uint8)).save("pyramid.png")
    Image.fromarray(original.astype(np.uint8)).save("original.png")
