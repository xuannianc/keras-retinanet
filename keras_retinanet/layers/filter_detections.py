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

import keras
from .. import backend


def filter_detections(
        boxes,
        classification,
        other=[],
        class_specific_filter=True,
        nms=True,
        score_threshold=0.05,
        max_detections=300,
        nms_threshold=0.5
):
    """ Filter detections using the boxes and classification values.

    Args:
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.

    Returns:
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores, labels):
        # threshold based on score
        # scores 的 shape 为 (num_boxes,)
        # labels 的 shape 为 (num_boxes,)
        # indices 的 shape 为 (num_greater_boxes,1), 第二维的元素为 box_id
        filtered_indices = backend.where(keras.backend.greater(scores, score_threshold))

        if nms:
            # shape 为 (num_greater_boxes,4)
            filtered_boxes = backend.gather_nd(boxes, filtered_indices)
            # gather 返回值的 shape 是 (num_greater_boxes,1)
            # 那么 fitered_scores 的 shape 为 (num_greater_boxes,)
            filtered_scores = keras.backend.gather(scores, filtered_indices)[:, 0]

            # perform NMS
            # shape 为 (num_nms,), 每一个元素为 filtered_box 的 id
            nms_indices = backend.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                      iou_threshold=nms_threshold)

            # filter indices based on NMS
            # shape 为 (num_nms,1), 每一个元素为 box_id
            filtered_indices = keras.backend.gather(filtered_indices, nms_indices)

        # add indices to list of all indices
        labels = backend.gather_nd(labels, filtered_indices)
        # shape 为 (num_nms, 2) 第二维的第一个元素表示 box_id, 第二个元素表示 class_id
        indices = keras.backend.stack([filtered_indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        # FIXME: 这种方式相比下一种会, 出现同一个 box 包含了多个 label 的情况
        all_indices = []
        # perform per class filtering
        for label in range(int(classification.shape[1])):
            # 所有 boxes 属于某一个 label 的 score
            scores_of_label = classification[:, label]
            labels = label * backend.ones((keras.backend.shape(scores_of_label)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores_of_label, labels))

        # concatenate indices to single tensor
        # shape 为 (total_num_nms, 2)
        detection_indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        # 每个 box 最大的 score
        max_score_of_boxes = keras.backend.max(classification, axis=1)
        max_score_labels_of_boxes = keras.backend.argmax(classification, axis=1)
        detection_indices = _filter_detections(max_score_of_boxes, max_score_labels_of_boxes)

    # select top k
    # shape 为 (total_num_nms,)
    detection_scores = backend.gather_nd(classification, detection_indices)
    # shape 为 (total_num_nms,)
    detection_labels = detection_indices[:, 1]
    # shape 为 (total_num_nms,)
    detection_boxes = detection_indices[:, 0]
    # tf.nn.top_k 参考 https://www.tensorflow.org/api_docs/python/tf/nn/top_k
    # 对 scores 是一维的情况, 返回的两个值的 shape 都是 (k,)
    # top_k_scores[i] 表示第 i 大的 score, top_k_score_indices[i] 表示第 i 大的 score 在 detection_scores 数组中的 idx
    top_k_scores, top_k_score_indices = backend.top_k(detection_scores, k=keras.backend.minimum(max_detections,
                                                                                  keras.backend.shape(detection_scores)[
                                                                                      0]))

    # filter input using the final set of indices
    # indices[:, 0] 表示的是 total_num_nms 个条目中的 box_ids
    top_k_box_indices = keras.backend.gather(detection_boxes, top_k_score_indices)
    top_k_boxes = keras.backend.gather(boxes, top_k_box_indices)
    top_k_labels = keras.backend.gather(detection_labels, top_k_score_indices)
    other_ = [keras.backend.gather(o, top_k_box_indices) for o in other]

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(top_k_scores)[0])
    boxes = backend.pad(top_k_boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = backend.pad(top_k_scores, [[0, pad_size]], constant_values=-1)
    labels = backend.pad(top_k_labels, [[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(labels, 'int32')
    other_ = [backend.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in
              other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    # tf.int_shape 返回 tensor 的 shape 为 a tuple of int
    # 参见 https://www.tensorflow.org/api_docs/python/tf/keras/backend/int_shape
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_


class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.05,
            max_detections=300,
            parallel_iterations=32,
            **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args:
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        other = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes = args[0]
            classification = args[1]
            other = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [(input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections), ] + \
               [tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in
                range(2, len(input_shape))]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })

        return config
