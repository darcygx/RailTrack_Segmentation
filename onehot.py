import numpy as np

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()  # ravel拉长为向量
    buf.ravel()[nmsk-1] = 1
    return buf

def main():
    pass


if __name__ == '__main__':
    main()

