import pywt
import numpy as np
import matplotlib.pyplot as plt


def test_signal(f=None):

    if f is None:
        f = [0.3, 0.13, 0.05]
    aa = []
    for j in f:
        aa.extend([np.sin(j * np.pi * i) + np.cos(0.01*np.pi * i) for i in range(200)])

    return aa


def wavelet_cwt(data = None,
                fs = 1000,
                wavename = 'cgau8',
                totalscal = 128,
                ylim = None):

    if data is None:
        data = test_signal()

    t = np.arange(0, len(data) / fs, 1.0 / fs)

    fc = pywt.central_frequency(wavename)

    # 计算对应频率的小波尺度
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)

    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1 / fs)

    plt.figure(figsize=(20, 5))
    plt.subplot(211)
    plt.plot(t, data)
    plt.xlabel('time')

    plt.subplot(212)
    plt.pcolormesh(t, frequencies, abs(cwtmatr))
    plt.ylabel('f')
    plt.xlabel('t')
    plt.subplots_adjust(hspace=0.4)

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.show()

    return t, frequencies, cwtmatr

def wavelet_dwt(data = None,
                wavename = 'db5',
                mode = 'smooth'):

    if data is None:
        data = test_signal()

    cA, cD = pywt.dwt(data, wavename)

    ya = pywt.idwt(cA, None, wavename, mode)
    yd = pywt.idwt(None,cD, wavename, mode)

    t = range(len(data))

    plt.figure(figsize=(20, 5))
    plt.subplot(311)
    plt.plot(t, data)
    plt.xlabel('time')

    plt.subplot(312)
    plt.plot(t, ya)
    plt.xlabel('time')

    plt.subplot(313)
    plt.plot(t, yd)
    plt.xlabel('time')

    plt.show()

def wavelet_dec(data = None,
                wavename = 'Haar',
                level = 3):

    if data is None:
        data = test_signal()

    coeffs = pywt.wavedec(data, wavename, mode='symmetric', level = level)
    # Returns: (list) [cA_n, cD_n, cD_n-1, …, cD_2, cD_1]
    # 其中n表示分解的级别。结果的第一个元素(CA_N)是近似分量，下面的元素(CD_n—CD_1)是细节分量。

    # ya4 = pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0]).tolist(), 'db4')
    # yd4 = pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0]).tolist(), 'db4')
    # yd3 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0]).tolist(), 'db4')
    # yd2 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0]).tolist(), 'db4')
    # yd1 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1]).tolist(), 'db4')

    for i in coeffs:
        print(len(i))

    plt.subplot(len(coeffs) + 1, 1, 1)
    plt.plot(data)

    for i in range(len(coeffs)):
        zeros = [0 for _ in range(len(coeffs))]
        zeros[i] = 1
        y = pywt.waverec(np.multiply(np.array(coeffs, dtype=object), zeros).tolist(), wavename)

        plt.subplot(len(coeffs)+1, 1, i + 2)

        plt.plot(y)

    yd1 = pywt.waverec(np.multiply(np.array(coeffs, dtype=object), [1, 1, 1, 1]).tolist(), wavename)
    plt.figure()
    plt.plot(yd1)

    plt.show()

def waveletpacket(data = None,
                  wavename = 'db4',
                  maxlevel = 10,
                  fs = 1000):

    if data is None:
        data = test_signal()

    freq_gap = 10
    freq_num = 10
    iter_freqs = [{'name':str(freq_gap*i)+ '--'+str(freq_gap*i+freq_gap),
                   'fmin': freq_gap*i,
                   'fmax': freq_gap*i+freq_gap} for i in range(freq_num)]

    wp = pywt.WaveletPacket(data=data, wavelet=wavename, mode='symmetric', maxlevel=maxlevel)

    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]

    freqBand = fs / (2 ** maxlevel)

    print(freqBand)

    fig, axes = plt.subplots(len(iter_freqs) + 1, 1, figsize=(20, 20), sharex=True, sharey=False)

    axes[0].plot(data)

    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavename, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):

            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand



            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data

        # 绘制对应频率的数据

        #         print(new_wp.reconstruct(update=True))
        axes[iter + 1].plot(new_wp.reconstruct(update=True))
        # 设置图名
        axes[iter + 1].set_title(iter_freqs[iter]['name'])
    plt.show()

if __name__ == '__main__':
    # waveletpacket()
    wavelet_dec()

    # data = test_signal()
    # plt.plot(data)
    # plt.show()