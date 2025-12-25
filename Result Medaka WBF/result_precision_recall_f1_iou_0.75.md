# YOLO-style Precision / Recall / F1

- Confidence threshold: **0.25**
- IoU threshold: **0.75**

## WBF - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_5_fold-test.csv | 0.891 | 0.935 | 0.913 |

## WBF - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_diff_size-test.csv | 0.867 | 0.902 | 0.884 |

## WBF - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_diff_version-test.csv | 0.896 | 0.911 | 0.903 |

## NMS - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_5_fold-test.csv | 0.747 | 0.935 | 0.830 |
| ensemble_soft-nms_5_fold-test.csv | 0.753 | 0.943 | 0.838 |

## NMS - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_diff_size-test.csv | 0.687 | 0.911 | 0.783 |
| ensemble_soft-nms_diff_size-test.csv | 0.653 | 0.919 | 0.764 |

## NMS - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_diff_version-test.csv | 0.727 | 0.911 | 0.809 |
| ensemble_soft-nms_diff_version-test.csv | 0.708 | 0.927 | 0.803 |

## NMW - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_5_fold-test.csv | 0.753 | 0.943 | 0.838 |

## NMW - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_diff_size-test.csv | 0.681 | 0.902 | 0.776 |

## NMW - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_diff_version-test.csv | 0.734 | 0.919 | 0.816 |

## Single Models - 5 Fold

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8-fold1-test.csv | 0.862 | 0.911 | 0.885 |
| YOLOv8-fold2-test.csv | 0.879 | 0.943 | 0.910 |
| YOLOv8-fold3-test.csv | 0.883 | 0.919 | 0.900 |
| YOLOv8-fold4-test.csv | 0.844 | 0.927 | 0.884 |
| YOLOv8-fold5-test.csv | 0.833 | 0.894 | 0.863 |

## Single Models - Different Sizes (YOLOv8n,YOLOv8s,YOLOv8m,YOLOv8l,YOLOv8x)

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8n-test.csv | 0.826 | 0.886 | 0.855 |
| YOLOv8s-test.csv | 0.812 | 0.878 | 0.844 |
| YOLOv8m-test.csv | 0.821 | 0.894 | 0.856 |
| YOLOv8l-test.csv | 0.835 | 0.902 | 0.867 |
| YOLOv8x-test.csv | 0.854 | 0.902 | 0.877 |

## Single YOLO - Different Version (YOLOv8n,YOLOv9t,YOLOv10n,YOLOv11n)

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8n-test.csv | 0.826 | 0.886 | 0.855 |
| YOLOv9t-test.csv | 0.811 | 0.870 | 0.839 |
| YOLOv10n-test.csv | 0.845 | 0.886 | 0.865 |
| YOLOv11n-test.csv | 0.821 | 0.894 | 0.856 |

