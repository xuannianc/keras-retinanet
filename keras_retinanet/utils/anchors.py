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

import numpy as np
import keras

from ..utils.compute_overlap import compute_overlap


class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """

    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    # NOTE: 32 表示在 P3 上 anchor 的大小, P3 相比与原图长和宽都缩小了 8 倍, 那么原图上 32 * 32, 到 P3 上就是 4 * 4
    # 且 P3 上每一个像素相当于原图上 8 * 8 个像素, stride=8, 相当于在 P3 上调整一个像素
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    # floatx() 返回 keras 默认使用的浮点数类型 'float32'
    ratios=np.array([0.5, 1, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def anchor_targets_bbox(
        anchors,
        image_group,
        annotations_group,
        num_classes,
        negative_overlap=0.4,
        positive_overlap=0.5
):
    """ Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert (len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert (len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert ('bboxes' in annotations), "Annotations should contain bboxes."
        assert ('labels' in annotations), "Annotations should contain labels."

    batch_size = len(image_group)

    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_indices = compute_gt_annotations(anchors,
                                                                                            annotations['bboxes'],
                                                                                            negative_overlap,
                                                                                            positive_overlap)

            labels_batch[index, ignore_indices, -1] = -1
            labels_batch[index, positive_indices, -1] = 1

            regression_batch[index, ignore_indices, -1] = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute target class labels
            # [argmax_overlaps_indices[positive_indices] 得到是 annotation 的下标
            # 设置 positive 的 anchor 的相应的 class 为 1
            labels_batch[
                index, positive_indices, annotations['labels'][argmax_overlaps_indices[positive_indices]].astype(int)] = 1
            #
            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_indices, :])

        # ignore annotations outside of image
        # 忽略中心点在图像外面的 anchor
        if image.shape:
            # vstack 之后 shape 为 (2, num_anchors), 转置之后的 shape 为 (num_anchors, 2)
            # 第一个元素是中心点 x 的坐标, 第二个元素是中心点 y 的坐标
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            # shape 为 (num_anchors,)
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1] = -1
            regression_batch[index, indices, -1] = -1

    return regression_batch, labels_batch


def compute_gt_annotations(
        anchors,
        annotations,
        negative_overlap=0.4,
        positive_overlap=0.5
):
    """ Obtain indices of gt annotations with the greatest overlap.

    Args:
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns:
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """
    # shape 为 (num_anchors, num_bboxes)
    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    # shape 为 (num_anchors,)
    argmax_overlaps_indices = np.argmax(overlaps, axis=1)
    # shape 为 (num_anchors,)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_indices]

    # 判断 positive indices 和 ignore indices
    # shape 为 (num_anchors,)
    positive_indices = max_overlaps >= positive_overlap
    # shape 为 (num_anchors,)
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_indices


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)

    return shape


def make_shapes_callback(model):
    """ Make a function for getting the shape of the pyramid levels.
    """

    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        anchor_params=None,
        shapes_callback=None,
):
    """ Generators anchors for a given shape.

    Args:
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns:
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    # guess_shapes 得到的每一个 feature map 的尺寸分别是原图的 [1/8, 1/16, 1/32, 1/64, 1/128]
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.append.html
        # np.append(a,b,axis=0) 相当于 np.concatenate([a,b],axis=0)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args:
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location. shape 为 (num_scales * num_ratios,4).
                 (-w/2,-h/2,w/2,h/2), 中心点在 (0,0)
    """

    # create a grid starting from half stride from the top left corner
    # 如原 image 的 shape 为 (512, 1024), C3 的 shape 为 (64, 128) 缩小了 8 倍
    # C3 上的一个像素相当于原来的 8 * 8 个像素, stride=8 表示每次移动一个像素
    # 那么 shift_x 为 np.array([0.5 * 8, 1.5 * 8,...,127.5 * 8])
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    # 那么 shift_y 为 np.array((0.5 * 8, 1.5 * 8,...,63.5 * 8])
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    # shift_x 为 np.array([[0.5 * 8, 1.5 * 8,...,127.5 * 8],...62 个...,[0.5 * 8, 1.5 * 8,...,127.5 * 8]])
    # shift_y 为 np.array([[0.5 * 8,...126 个...,0.5 * 8],...62 个...,[63.5 * 8,...126 个...,63.5 * 8]])
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shifts 为 np.array([[0.5 * 8, 1.5 * 8,...,127.5 * 8,...62 个...,0.5 * 8, 1.5 * 8,...,127.5 * 8],
    #                     [0.5 * 8,...126 个...,0.5 * 8,...62 个...,63.5 * 8,...126 个...,63.5 * 8],
    #                     [0.5 * 8, 1.5 * 8,...,127.5 * 8,...62 个...,0.5 * 8, 1.5 * 8,...,127.5 * 8],
    #                     [0.5 * 8,...126 个...,0.5 * 8,...62 个...,63.5 * 8,...126 个...,63.5 * 8]].T
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to cell K shifts (K, 1, 4) to get shift anchors (K, A, 4)
    # then reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    # numpy broadcast
    # 按照上面的举例: shape 为 (64 * 128, 9, 4)
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    # 按照上面的举例: shape 为 (64 * 128 * 9, 4)
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.

    Args:
        base_size: anchor 的大小
        ratios: aspect ratios w/h
        scales: 以 base_size 为基础对 anchor 进行适当的缩放

    Returns:
        shape 为 (num_ratios * num_scales, 4)
        生成 num_ratios * num_scales 个中心点在 (0,0) 的 anchor
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales
    # ratios 和 scales 都是 np.array, len() 是可以作用于 np.array 的
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
    # shape 为 (num_anchors, )
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    # np.repeat 的第三个参数为 axis=None, 默认是先铺平再对元素重复
    # 参见 https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.repeat.html
    # 如 ratios 为 np.array([0.5,1,2]),len(scales)=3, repeat 之后返回 np.array([0.5,0.5,0.5,1,1,1,2,2,2])
    # FIXME: 修改后宽小于长的 anchors 放在前面
    # 获取宽
    anchors[:, 2] = np.sqrt(areas * np.repeat(ratios, len(scales)))
    # 获取高
    anchors[:, 3] = anchors[:, 2] / np.repeat(ratios, len(scales))
    # anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    # anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (0, 0, w, h) -> (x1, y1, x2, y2)
    # anchors[:,2] * 0.5 把所有的 width 乘以 0.5, shape 是 (num_anchors,)
    # np.tile 之后, shape 变为 (2, num_anchors),然后再转置 (num_anchors,2)
    # anchors 的每一行元素变成 (-w/2,-h/2,w/2,h/2), 以 (0,0) 作为中心点
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights
    # shape 为 (4, num_anchors)
    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    # shape 为 (num_anchors, 4)
    targets = targets.T

    targets = (targets - mean) / std

    return targets
