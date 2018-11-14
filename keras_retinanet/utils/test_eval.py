import numpy as np


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# 参考 https://github.com/rafaelpadilla/Object-Detection-Metrics
recall = [0.0666, 0.0666, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.2, 0.2, 0.2666, 0.3333, 0.4, 0.4,
          0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4666, 0.4666]
precision = [1, 0.5, 0.6666, 0.5, 0.4, 0.3333, 0.2857, 0.25, 0.2222, 0.3, 0.2727, 0.3333, 0.3846, 0.4285, 0.4, 0.375,
             0.3529, 0.3333, 0.3157, 0.3, 0.2857, 0.2727, 0.3043, 0.2916]
print(len(recall))
print(len(precision))
print(_compute_ap(recall, precision))
