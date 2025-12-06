from itertools import product as product

import numpy as np
import torch


def nms_(dets, thresh):
    """Applies non-maximum suppression (CPU implementation).

    Courtesy of Ross Girshick:
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py

    Args:
        dets: Detection array of shape [N, 5] where each row is
            [x1, y1, x2, y2, score].
        thresh: IoU threshold for suppression.

    Returns:
        np.ndarray: Indices of kept detections.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep).astype(int)


def decode(loc, priors, variances):
    """Decodes locations from predictions using priors.

    Reverses the encoding applied for offset regression at train time.

    Args:
        loc: Location predictions for loc layers. Shape: [num_priors, 4].
        priors: Prior boxes in center-offset form. Shape: [num_priors, 4].
        variances: Variances of priorboxes.

    Returns:
        torch.Tensor: Decoded bounding box predictions.
    """

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Applies non-maximum suppression at test time.

    Prevents detecting too many overlapping bounding boxes for a given object.

    Args:
        boxes: The location preds for the img. Shape: [num_priors, 4].
        scores: The class pred scores for the img. Shape: [num_priors].
        overlap: The overlap thresh for suppressing unnecessary boxes.
        top_k: The maximum number of box preds to consider.

    Returns:
        tuple: (keep, count) - The indices of kept boxes and their count.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


class Detect(object):
    """Post-processor for object detection that applies NMS to predictions.

    Args:
        num_classes: Number of classes to detect.
        top_k: Maximum number of detections to keep per class.
        nms_thresh: IoU threshold for non-maximum suppression.
        conf_thresh: Confidence threshold for filtering detections.
        variance: Prior box variances for decoding predictions.
        nms_top_k: Maximum detections to consider before NMS.
    """

    def __init__(
        self,
        num_classes=2,
        top_k=750,
        nms_thresh=0.3,
        conf_thresh=0.05,
        variance=[0.1, 0.2],
        nms_top_k=5000,
    ):
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.nms_top_k = nms_top_k

    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
                count = count if count < self.top_k else self.top_k

                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1
                )

        return output


class PriorBox(object):
    """Generates prior boxes (anchors) for object detection.

    Args:
        input_size: Input image size as (height, width).
        feature_maps: List of feature map sizes for each detection layer.
        variance: Variances for encoding/decoding box coordinates.
        min_sizes: Minimum box sizes for each feature map layer.
        steps: Stride/step size for each feature map layer.
        clip: Whether to clip prior boxes to [0, 1] range.
    """

    def __init__(
        self,
        input_size,
        feature_maps,
        variance=[0.1, 0.2],
        min_sizes=[16, 32, 64, 128, 256, 512],
        steps=[4, 8, 16, 32, 64, 128],
        clip=False,
    ):
        super(PriorBox, self).__init__()

        self.imh = input_size[0]
        self.imw = input_size[1]
        self.feature_maps = feature_maps

        self.variance = variance
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip

    def forward(self):
        mean = []
        for k, fmap in enumerate(self.feature_maps):
            feath = fmap[0]
            featw = fmap[1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh

                mean += [cx, cy, s_kw, s_kh]

        output = torch.FloatTensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)

        return output
