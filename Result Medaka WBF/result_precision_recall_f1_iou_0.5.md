# YOLO-style Precision / Recall / F1

- Confidence threshold: **0.25**
- IoU threshold: **0.5**

## WBF - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_5_fold-test.csv | 0.930 | 0.976 | 0.952 |

## WBF - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_diff_size-test.csv | 0.922 | 0.959 | 0.940 |

## WBF - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_wbf_diff_version-test.csv | 0.944 | 0.959 | 0.952 |

## NMS - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_5_fold-test.csv | 0.799 | 1.000 | 0.888 |
| ensemble_soft-nms_5_fold-test.csv | 0.799 | 1.000 | 0.888 |

## NMS - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_diff_size-test.csv | 0.742 | 0.984 | 0.846 |
| ensemble_soft-nms_diff_size-test.csv | 0.699 | 0.984 | 0.818 |

## NMS - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nms_diff_version-test.csv | 0.786 | 0.984 | 0.874 |
| ensemble_soft-nms_diff_version-test.csv | 0.752 | 0.984 | 0.852 |

## NMW - 5 Fold Ensemble

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_5_fold-test.csv | 0.799 | 1.000 | 0.888 |

## NMW - Different Model Sizes

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_diff_size-test.csv | 0.742 | 0.984 | 0.846 |

## NMW - Different Model Version

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| ensemble_nmw_diff_version-test.csv | 0.786 | 0.984 | 0.874 |

## Single Models - 5 Fold

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8-fold1-test.csv | 0.900 | 0.951 | 0.925 |
| YOLOv8-fold2-test.csv | 0.932 | 1.000 | 0.965 |
| YOLOv8-fold3-test.csv | 0.937 | 0.976 | 0.956 |
| YOLOv8-fold4-test.csv | 0.896 | 0.984 | 0.938 |
| YOLOv8-fold5-test.csv | 0.902 | 0.967 | 0.933 |

## Single Models - Different Sizes (YOLOv8n,YOLOv8s,YOLOv8m,YOLOv8l,YOLOv8x)

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8n-test.csv | 0.909 | 0.976 | 0.941 |
| YOLOv8s-test.csv | 0.895 | 0.967 | 0.930 |
| YOLOv8m-test.csv | 0.873 | 0.951 | 0.911 |
| YOLOv8l-test.csv | 0.887 | 0.959 | 0.922 |
| YOLOv8x-test.csv | 0.892 | 0.943 | 0.917 |

## Single YOLO - Different Version (YOLOv8n,YOLOv9t,YOLOv10n,YOLOv11n)

| Model | Precision | Recall | F1 |
|------|-----------|--------|----|
| YOLOv8n-test.csv | 0.909 | 0.976 | 0.941 |
| YOLOv9t-test.csv | 0.871 | 0.935 | 0.902 |
| YOLOv10n-test.csv | 0.899 | 0.943 | 0.921 |
| YOLOv11n-test.csv | 0.881 | 0.959 | 0.918 |

