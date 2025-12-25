# YOLO-style Precision / Recall / F1

- Confidence threshold: **0.25**
- IoU threshold: **0.7**

## WBF - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_5_fold-test.csv | 0.915 | 0.959 | 0.937 |

## WBF - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_diff_size-test.csv | 0.891 | 0.927 | 0.908 |

## WBF - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_diff_version-test.csv | 0.920 | 0.935 | 0.927 |

## NMS - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_5_fold-test.csv | 0.766 | 0.959 | 0.852 |
| ensemble_soft-nms_5_fold-test.csv | 0.779 | 0.976 | 0.866 |

## NMS - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_diff_size-test.csv | 0.730 | 0.967 | 0.832 |
| ensemble_soft-nms_diff_size-test.csv | 0.682 | 0.959 | 0.797 |

## NMS - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_diff_version-test.csv | 0.766 | 0.959 | 0.852 |
| ensemble_soft-nms_diff_version-test.csv | 0.733 | 0.959 | 0.831 |

## NMW - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_5_fold-test.csv | 0.779 | 0.976 | 0.866 |

## NMW - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_diff_size-test.csv | 0.724 | 0.959 | 0.825 |

## NMW - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_diff_version-test.csv | 0.773 | 0.967 | 0.859 |

## Single Models - 5 Fold

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8-fold1-test.csv | 0.892 | 0.943 | 0.917 |
| YOLOv8-fold2-test.csv | 0.902 | 0.967 | 0.933 |
| YOLOv8-fold3-test.csv | 0.906 | 0.943 | 0.924 |
| YOLOv8-fold4-test.csv | 0.859 | 0.943 | 0.899 |
| YOLOv8-fold5-test.csv | 0.856 | 0.919 | 0.886 |

## Single Models - Different Sizes (YOLOv8n,YOLOv8s,YOLOv8m,YOLOv8l,YOLOv8x)

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8n-test.csv | 0.871 | 0.935 | 0.902 |
| YOLOv8s-test.csv | 0.865 | 0.935 | 0.898 |
| YOLOv8m-test.csv | 0.843 | 0.919 | 0.879 |
| YOLOv8l-test.csv | 0.850 | 0.919 | 0.883 |
| YOLOv8x-test.csv | 0.862 | 0.911 | 0.885 |

## Single YOLO - Different Version (YOLOv8n,YOLOv9t,YOLOv10n,YOLOv11n)

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8n-test.csv | 0.871 | 0.935 | 0.902 |
| YOLOv9t-test.csv | 0.833 | 0.894 | 0.863 |
| YOLOv10n-test.csv | 0.860 | 0.902 | 0.881 |
| YOLOv11n-test.csv | 0.843 | 0.919 | 0.879 |

