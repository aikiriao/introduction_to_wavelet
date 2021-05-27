import unittest
import fwt
import numpy as np
import math
import random


def ifwt1d_ref(decomp_src, decomp_wav, scaling_coef):
    """ 1次元高速ウェーブレット逆変換（リファレンス） """
    wavelet_coef = fwt.calculate_wavelet_coef(scaling_coef)
    # 素朴な実装
    decomp_src_len = len(decomp_src)
    src = np.zeros(2 * decomp_src_len)
    for n in range(0, 2 * decomp_src_len, 2):
        for k in range(0, len(scaling_coef), 2):
            src[n] += scaling_coef[k] * decomp_src[(n - k) // 2]
            src[n] += wavelet_coef[k] * decomp_wav[(n - k) // 2]
            src[n + 1] += scaling_coef[k + 1] * decomp_src[(n - k) // 2]
            src[n + 1] += wavelet_coef[k + 1] * decomp_wav[(n - k) // 2]
    return src


def fwt1d_ref(src, scaling_coef):
    """ 1次元高速ウェーブレット変換（リファレンス） """
    wavelet_coef = fwt.calculate_wavelet_coef(scaling_coef)
    src_len = len(src)
    coef_len = len(scaling_coef)
    tmp_src = np.zeros(src_len)
    tmp_wav = np.zeros(src_len)
    for n in range(src_len):
        for k in range(coef_len):
            tmp_src[n] += scaling_coef[k] * src[(n + k) % src_len]
            tmp_wav[n] += wavelet_coef[k] * src[(n + k) % src_len]
    return [tmp_src[::2], tmp_wav[::2]]


class TestFWT(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_generate_scaling_coef(self):
        """ スケーリング係数の生成テスト """
        haar_scaling_coef = np.array([1, 1])
        haar_wavelet_coef = np.array([1, -1])
        haar_scaling_coef = haar_scaling_coef / np.linalg.norm(haar_scaling_coef)
        haar_wavelet_coef = haar_wavelet_coef / np.linalg.norm(haar_wavelet_coef)
        test_coef = fwt.calculate_wavelet_coef(haar_scaling_coef)
        self.assertEqual(haar_wavelet_coef.tolist(), test_coef.tolist())

    def test_ref_decomp_comp_by_haar(self):
        """ リファレンス実装・ハール基底による分解・再合成テスト """
        haar_scaling_coef = np.array([1, 1])
        haar_scaling_coef = haar_scaling_coef / np.linalg.norm(haar_scaling_coef)
        # 無音
        src = np.zeros(8)
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())
        # 直流
        src = np.ones(8)
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())
        # -1,1繰り返し振動
        src = np.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())
        # 長めの正弦波
        src = np.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())
        # ホワイトノイズ
        src = np.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())

    def test_ref_decomp_comp_by_daubechies(self):
        """ リファレンス実装・ドベシィ基底による分解・再合成テスト """
        scaling_coef = np.array([0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551])
        # 無音
        src = np.zeros(8)
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())
        # 直流
        src = np.ones(8)
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())
        # -1,1繰り返し振動
        src = np.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())
        # 長めの正弦波
        src = np.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())
        # ホワイトノイズ
        src = np.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt1d_ref(src, scaling_coef)
        comp_src = ifwt1d_ref(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, comp_src).all())

    def test_decomp_ref_by_haar(self):
        """ 分解結果の一致確認テスト """
        haar_scaling_coef = np.array([1, 1])
        haar_scaling_coef = haar_scaling_coef / np.linalg.norm(haar_scaling_coef)
        # 無音
        src = np.zeros(8)
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef)
        self.assertTrue(np.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(np.isclose(decomp_wav, decomp_wav_test).all())
        # 直流
        src = np.ones(8)
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef)
        self.assertTrue(np.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(np.isclose(decomp_wav, decomp_wav_test).all())
        # -1,1繰り返し振動
        src = np.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef)
        self.assertTrue(np.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(np.isclose(decomp_wav, decomp_wav_test).all())
        # 長めの正弦波
        src = np.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef)
        self.assertTrue(np.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(np.isclose(decomp_wav, decomp_wav_test).all())
        # ホワイトノイズ
        src = np.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt1d_ref(src, haar_scaling_coef)
        decomp_src_test, decomp_wav_test = fwt.fwt1d(src, haar_scaling_coef)
        self.assertTrue(np.isclose(decomp_src, decomp_src_test).all())
        self.assertTrue(np.isclose(decomp_wav, decomp_wav_test).all())

    def test_decomp_comp_by_haar(self):
        """ ハール基底による分解・再合成テスト """
        haar_scaling_coef = np.array([1, 1])
        haar_scaling_coef = haar_scaling_coef / np.linalg.norm(haar_scaling_coef)
        # 無音
        src = np.zeros(8)
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # 直流
        src = np.ones(8)
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # -1,1繰り返し振動
        src = np.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # 長めの正弦波
        src = np.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # ホワイトノイズ
        src = np.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, haar_scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())

    def test_decomp_comp_by_daubechies(self):
        """ ドベシィ基底による分解・再合成テスト """
        scaling_coef = np.array([0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551])
        # 無音
        src = np.zeros(8)
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # 直流
        src = np.ones(8)
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # -1,1繰り返し振動
        src = np.array([ (-1)**(i) for i in range(8) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # 長めの正弦波
        src = np.array([ math.sin(i) for i in range(128) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # ホワイトノイズ
        src = np.array([ random.random() for i in range(16) ])
        decomp_src, decomp_wav = fwt.fwt1d(src, scaling_coef)
        src_test = fwt.ifwt1d(decomp_src, decomp_wav, scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())

    def test_2d_decomp_comp_by_haar(self):
        """ ハール基底による2次元分解・再合成テスト """
        haar_scaling_coef = np.array([1, 1])
        haar_scaling_coef = haar_scaling_coef / np.linalg.norm(haar_scaling_coef)
        # 全て0
        src = np.zeros((4,4))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # 全て1
        src = np.ones((4,4))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # -1,1繰り返し振動
        src = np.array([(-1)**(i) for i in range(4 * 4)]).reshape((4,4))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # 長めの正弦波
        src = np.array([math.sin(i) for i in range(16 * 16)]).reshape((16, 16))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())
        # ホワイトノイズ
        src = np.array([random.random() for i in range(4 * 4)]).reshape((4, 4))
        ll, hl, lh, hh = fwt.fwt2d(src, haar_scaling_coef)
        src_test = fwt.ifwt2d(ll, hl, lh, hh, haar_scaling_coef)
        self.assertTrue(np.isclose(src, src_test).all())

if __name__ == '__main__':
    unittest.main()
