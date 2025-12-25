# YOLO-style Precision / Recall / F1

- Confidence threshold: **0.5**
- IoU threshold: **0.7**

## WBF - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_5_fold-test.csv | 0.934 | 0.919 | 0.926 |

## WBF - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_diff_size-test.csv | 0.949 | 0.902 | 0.925 |

## WBF - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_diff_version-test.csv | 0.965 | 0.886 | 0.924 |

## NMS - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_5_fold-test.csv | 0.830 | 0.951 | 0.886 |
| ensemble_soft-nms_5_fold-test.csv | 0.844 | 0.967 | 0.902 |

## NMS - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_diff_size-test.csv | 0.830 | 0.951 | 0.886 |
| ensemble_soft-nms_diff_size-test.csv | 0.765 | 0.951 | 0.848 |

## NMS - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_diff_version-test.csv | 0.848 | 0.951 | 0.897 |
| ensemble_soft-nms_diff_version-test.csv | 0.801 | 0.951 | 0.870 |

## NMW - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_5_fold-test.csv | 0.844 | 0.967 | 0.902 |

## NMW - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_diff_size-test.csv | 0.823 | 0.943 | 0.879 |

## NMW - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_diff_version-test.csv | 0.855 | 0.959 | 0.904 |

## Single Models - 5 Fold

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8-fold1-test.csv | 0.935 | 0.935 | 0.935 |
| YOLOv8-fold2-test.csv | 0.921 | 0.943 | 0.932 |
| YOLOv8-fold3-test.csv | 0.927 | 0.927 | 0.927 |
| YOLOv8-fold4-test.csv | 0.913 | 0.943 | 0.928 |
| YOLOv8-fold5-test.csv | 0.875 | 0.911 | 0.892 |

## Single Models - Different Sizes (YOLOv8n,YOLOv8s,YOLOv8m,YOLOv8l,YOLOv8x)

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8n-test.csv | 0.912 | 0.927 | 0.919 |
| YOLOv8s-test.csv | 0.902 | 0.902 | 0.902 |
| YOLOv8m-test.csv | 0.889 | 0.911 | 0.900 |
| YOLOv8l-test.csv | 0.897 | 0.919 | 0.908 |
| YOLOv8x-test.csv | 0.910 | 0.902 | 0.906 |

## Single YOLO - Different Version (YOLOv8n,YOLOv9t,YOLOv10n,YOLOv11n)

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8n-test.csv | 0.912 | 0.927 | 0.919 |
| YOLOv9t-test.csv | 0.880 | 0.894 | 0.887 |
| YOLOv10n-test.csv | 0.922 | 0.862 | 0.891 |
| YOLOv11n-test.csv | 0.896 | 0.911 | 0.903 |

