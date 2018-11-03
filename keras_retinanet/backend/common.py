"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras.backend
from .dynamic import meshgrid


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args:
        shape  : Shape of feature map to shift the anchors over. (feature_map_height, feature_map_width)
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
                接收的 anchors 的中心点都在 (0,0),而真正 anchors 的起点是 (0.5 * stride, 0.5 * stride)
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    # 假设 shift_x 的 shape 为 (m,),shift_y 的 shape 为 (n,),那么 meshgrid 会生成两个 shape 为 (m,n) 的数组
    # 第一个数组的内容相当于沿着 axis=0 复制了 n 次, 第二个数组的内容相当于把每个元素沿着 axis=1 复制了 m 次
    # 假设 shape=(4,3),stride=1 shift_x=np.array([0.5,1.5,2.5,3.5]), shift_y=np.array([0.5,1.5,2.5])
    # 那么 np.meshgrid(a,b) 生成两个数组
    # 第一个数组为 np.array([[0.5,1.5,2.5,3.5],[0.5,1.5,2.5,3.5],[0.5,1.5,2.5,3.5]]) * stride
    # 第二个数组为 np.array([[0.5,0.5,0.5,0.5],[1.5,1.5,1.5,1.5],[2.5,2.5,2.5,2.5]]) * stride
    shift_x, shift_y = meshgrid(shift_x, shift_y)
    # reshape 后的 shift_x 变成 np.array([0.5,1.5,2.5,3.5,0.5,1.5,2.5,3.5,0.5,1.5,2.5,3.5]) * stride
    shift_x = keras.backend.reshape(shift_x, [-1])
    # reshape 后的 shift_y 变成 np.array([0.5,0.5,0.5,0.5,1.5,1.5,1.5,1.5,2.5,2.5,2.5,2.5]) * stride
    shift_y = keras.backend.reshape(shift_y, [-1])
    # shifts 为 np.array([[0.5,1.5,2.5,3.5,0.5,1.5,2.5,3.5,0.5,1.5,2.5,3.5],
    #                    [0.5,0.5,0.5,0.5,1.5,1.5,1.5,1.5,2.5,2.5,2.5,2.5],
    #                    [0.5,1.5,2.5,3.5,0.5,1.5,2.5,3.5,0.5,1.5,2.5,3.5],
    #                    [0.5,0.5,0.5,0.5,1.5,1.5,1.5,1.5,2.5,2.5,2.5,2.5]]) * stride
    # 可以依次视为 x1,y1,x2,y2 的 shift
    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)
    # keras.backend.transpose 用来转置
    # 参见 https://www.tensorflow.org/api_docs/python/tf/keras/backend/transpose
    # shape 为 (k,4)
    shifts = keras.backend.transpose(shifts)
    num_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w
    # num_anchors 个 anchor 分别做 k 个 shift
    # 这一步应该是用了 broadcast,太复杂了 , 返回的 shape (k, num_anchors, 4)
    # TODO 改掉这样的实现
    shifted_anchors = keras.backend.reshape(anchors, [1, num_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * num_anchors, 4])

    return shifted_anchors
