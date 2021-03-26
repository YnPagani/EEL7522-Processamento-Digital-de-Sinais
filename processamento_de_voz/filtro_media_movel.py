from os.path import dirname, join as pjoin
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    plt.style.use('seaborn')
    wave_path = pjoin(dirname(__file__), "voice_test.wav")

    samplerate, data = wavfile.read(wave_path)
    length = data.shape[0] / samplerate

    print(f"Sample rate: {samplerate}")
    print(f"Audio length: {length}")

    # used to normalize the samples
    m = np.amax(np.abs(data))

    # changing from int16 to float32. Note that we need to normalize the
    # amplitudes since int16 has a range from (-32768, +32768) and float32
    # has (-1.0, +1.0)
    data = (data/m).astype(np.float32)

    # voice_test and voice_test_float32 should sound the same
    wavfile.write("voice_test_float.wav", samplerate, data)

    # y will be the output. We create y using data just to have the same shape
    y = np.copy(data)

    time = np.linspace(0., length, data.shape[0])

    # First window with plots for each M
    plt.figure(1)

    M = (50, 100, 1000)
    for n, m in enumerate(M, start=1):
        for i in range(1, data.shape[0]):
            if i < m:
                y[i] = y[i - 1] + (data[i]/m)
            else:
                y[i] = y[i - 1] + ((data[i] - data[i - m]) / m)

        # Saving wav file
        wav_name = f"voice_test_avg{m}.wav"
        wavfile.write(wav_name, samplerate, y)

        # Plotting audio wav for each M
        plt.subplot(len(M), 1, n)
        plt.title(f"M: {m}")
        plt.plot(time, y[:, 0], label="Left channel")
        plt.plot(time, y[:, 1], label="Right channel")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

    # Guarantee space between which subplot
    plt.tight_layout()

    # Second window with original signal
    plt.figure(2)
    plt.title("Original")
    plt.plot(time, data[:, 0], label="Left channel")
    plt.plot(time, data[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()



