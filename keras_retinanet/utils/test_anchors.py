import numpy as np
import keras


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    ratios = np.array([0.5, 1, 2], keras.backend.floatx())
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())
    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    # np.tile 对 np.array 的内容进行复制, 第二个参数表示复制的次数
    # 如果第二个参数是一个 int, 那么表示沿 axis=-1 进行复制
    # 如果第二个参数是一个 tuple, 那么分别沿对应的 axis 进行复制
    # 参见 https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.tile.html
    # 假设 scales 和 ratios 都是默认值, 即为 np.array([1, 2^(1/3)=1.26, 2^(2/3)=1.59]) 和 np.array([0.5, 1, 2])
    # np.tile 之后就变为 np.array([[1,1.26,1.59,1,1.26,1.59,1,1.26,1.59],[1,1.26,1.59,1,1.26,1.59,1,1.26,1.59]])
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    # np.repeat 的第三个参数为 axis=None, 默认是先铺平再对元素重复
    # 参见 https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.repeat.html
    # 如 ratios 为 np.array([0.5,1,2]),len(scales)=3, repeat 之后返回 np.array([0.5,0.5,0.5,1,1,1,2,2,2])
    # FIXME: 存储图片的 height?
    anchors[:, 2] = np.sqrt(areas * np.repeat(ratios, len(scales)))
    # FIXME: 存储图片的 width?
    anchors[:, 3] = anchors[:, 2] / np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


generate_anchors()
