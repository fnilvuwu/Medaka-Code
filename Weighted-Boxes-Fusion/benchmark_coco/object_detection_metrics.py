from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes

with open('tests/sample/groundtruths_coco.json') as fp:
    gold_dataset = coco_decoder.load_true_object_detection_dataset(fp)
with open('tests/sample/detections_coco.json') as fp:
    pred_dataset = coco_decoder.load_pred_object_detection_dataset(fp, gold_dataset)

gt_BoundingBoxes = get_bounding_boxes(gold_dataset)
pd_BoundingBoxes = get_bounding_boxes(pred_dataset)
results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)

for cls, metric in results.items():
    label = metric.label
    print('ap', metric.ap)
    print('precision', metric.precision)
    print('interpolated_recall', metric.interpolated_recall)
    print('interpolated_precision', metric.interpolated_precision)
    print('tp', metric.tp)
    print('fp', metric.fp)
    print('num_groundtruth', metric.num_groundtruth)
    print('num_detection', metric.num_detection)

from podm.metrics import MetricPerClass
mAP = MetricPerClass.mAP(results)

from podm.box import Box, intersection_over_union

box1 = Box.of_box(0., 0., 10., 10.)
box2 = Box.of_box(1., 1., 11., 11.)
intersection_over_union(box1, box2)