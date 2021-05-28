" 高速ウェーブレット変換（サンプル） "
import numpy as np
from scipy.ndimage import correlate1d
from scipy.ndimage import convolve1d

# スケーリング係数
HAAR_SCALING_COEF = [0.707106781186547, 0.707106781186547] 
DAUBECHIES2_SCALING_COEF = [0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551]
DAUBECHIES3_SCALING_COEF = [0.332670552950, 0.806891509311, 0.459877502118, -0.135011020010, -0.085441273882, 0.035226291882]
DAUBECHIES4_SCALING_COEF  = [0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417, -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785]

def _roundup_power_of_two(integer):
    """ 入力整数を2の冪数(2, 4, 8, 16, ...)に切り上げる """
    return 2 ** int(np.ceil(np.log2(integer)))


def _minmax_scale(vec, maxscale):
    """ 入力ベクトル値の範囲が[0, maxscale]になるように調整 """
    minval = np.min(vec)
    maxval = np.max(vec)
    # ゼロ除算対策のため、非ゼロ要素だけ除算
    div = np.divide(vec - minval, maxval - minval, where=((vec - minval) != 0))
    return maxscale * div


def calculate_wavelet_coef(scaling_coef):
    """ ウェーブレット係数からスケーリング係数を生成 """
    wavelet_coef = scaling_coef[::-1].copy() # 順序逆の配列
    for n in range(len(scaling_coef)):
        wavelet_coef[n] *= ((-1) ** n)
    return wavelet_coef


def fwt1d(src, scaling_coef):
    """ 1次元高速ウェーブレット変換 """
    # ウェーブレット係数計算
    wavelet_coef = calculate_wavelet_coef(scaling_coef)
    # 入力が整数だと丸め込まれるためfloatに変換
    src = src.astype(float)
    # correlate1dはフィルタカーネルの半分（中心）だけ出力が後ろにずれるので
    # 先に入力を前にずらしておく
    src = np.roll(src, -len(scaling_coef)//2)
    # 畳み込み 入力の端点は巡回
    # フィルタのインデックスが正方向に増加するためcorrelate1dを使用
    decomp_src = correlate1d(src, scaling_coef, mode='wrap')[::2]
    decomp_wav = correlate1d(src, wavelet_coef, mode='wrap')[::2]
    return [decomp_src, decomp_wav]


def ifwt1d(decomp_src, decomp_wav, scaling_coef):
    """ 1次元高速ウェーブレット逆変換 """
    # ウェーブレット係数計算
    wavelet_coef = calculate_wavelet_coef(scaling_coef)
    src_len = 2 * len(decomp_src)
    # 0値挿入
    scaling_interp = np.zeros(src_len)
    scaling_interp[::2] = decomp_src
    wavelet_interp = np.zeros(src_len)
    wavelet_interp[::2] = decomp_wav
    # 畳み込み 入力の端点は巡回
    src = convolve1d(scaling_interp, scaling_coef, mode='wrap')
    src += convolve1d(wavelet_interp, wavelet_coef, mode='wrap')
    # convolve1dはフィルタカーネルの半分（中心）だけ出力が前にずれるので
    # 入力を後ろにずらす
    src = np.roll(src, len(scaling_coef)//2)
    return src


def fwt2d(src2d, scaling_coef):
    """ 2次元高速ウェーブレット変換 """
    src_len = src2d.shape[0]
    half_src_len = src_len // 2
    src2d_ll = np.zeros((half_src_len, half_src_len))
    src2d_hl = np.zeros((half_src_len, half_src_len))
    src2d_lh = np.zeros((half_src_len, half_src_len))
    src2d_hh = np.zeros((half_src_len, half_src_len))
    src2d_l = np.zeros((src_len, half_src_len))
    src2d_h = np.zeros((src_len, half_src_len))
    # src2dを低域（左）と高域（右）に分解
    for j in range(src_len):
        sl, sh = fwt1d(src2d[j, :], scaling_coef)
        src2d_l[j, :] = sl
        src2d_h[j, :] = sh
    # src2d_l, src2d_hを更に左上(ll)、左下(hl)、右上(lh)、右下(hh)に分割
    for j in range(half_src_len):
        sl, sh = fwt1d(src2d_l[:, j], scaling_coef)
        src2d_ll[:, j] = sl
        src2d_hl[:, j] = sh
        sl, sh = fwt1d(src2d_h[:, j], scaling_coef)
        src2d_lh[:, j] = sl
        src2d_hh[:, j] = sh
    return [src2d_ll, src2d_hl, src2d_lh, src2d_hh]


def ifwt2d(src2d_ll, src2d_hl, src2d_lh, src2d_hh, scaling_coef):
    """ 2次元高速ウェーブレット逆変換 """
    src_len = src2d_ll.shape[0]
    twice_src_len = 2 * src_len
    src2d = np.zeros((twice_src_len, twice_src_len))
    src2d_l = np.zeros((twice_src_len, src_len))
    src2d_h = np.zeros((twice_src_len, src_len))
    # 左上(ll)、左下(hl)、右上(lh)、右下(hh)から左(l)、右(h)に合成
    for j in range(src_len):
        src2d_l[:, j] = ifwt1d(src2d_ll[:, j], src2d_hl[:, j], scaling_coef)
        src2d_h[:, j] = ifwt1d(src2d_lh[:, j], src2d_hh[:, j], scaling_coef)
    # 左(l)、右(h)から元を合成
    for j in range(twice_src_len):
        src2d[j, :] = ifwt1d(src2d_l[j, :], src2d_h[j, :], scaling_coef)
    return src2d


if __name__ == "__main__":
    import sys
    from PIL import Image

    maxlevel = 2

    # 画像読み込み（簡略化のためグレスケ変換）
    img = Image.open(sys.argv[1]).convert("L")

    # サイズを2の冪数に切り上げ、配列に画像データをロード
    # 余白は0埋め
    p2width = _roundup_power_of_two(max(img.height, img.width))
    original = np.zeros((p2width, p2width))
    original[0:img.height, 0:img.width] = np.asarray(img)
    image_pyramid = original.copy()

    # スケーリング係数
    scaling_coef = HAAR_SCALING_COEF
    # scaling_coef = DAUBECHIES2_SCALING_COEF
    # scaling_coef = DAUBECHIES3_SCALING_COEF
    # scaling_coef = DAUBECHIES4_SCALING_COEF

    # 分解
    ll = image_pyramid
    image_octabe = []
    for _ in range(maxlevel):
        ll, hl, lh, hh = fwt2d(ll, scaling_coef)
        width = ll.shape[0]
        image_pyramid[0:width, 0:width] = _minmax_scale(ll, 255)
        image_pyramid[0:width, width:2*width] = _minmax_scale(hl, 255)
        image_pyramid[width:2*width, 0:width] = _minmax_scale(lh, 255)
        image_pyramid[width:2*width, width:2*width] = _minmax_scale(hh, 255)
        image_octabe.insert(0, [ll, hl, lh, hh])

    # 合成
    for _ in range(maxlevel):
        ll, hl, lh, hh = image_octabe.pop(0)
        reconstract = ifwt2d(ll, hl, lh, hh, scaling_coef)

    print('Reconstract RMSE:', np.linalg.norm(original - reconstract))

    # 結果保存
    Image.fromarray(original.astype(np.uint8)).save("original.png")
    Image.fromarray(image_pyramid.astype(np.uint8)).save("pyramid.png")
    Image.fromarray(reconstract.astype(np.uint8)).save("reconstruct.png")

    sys.exit()
