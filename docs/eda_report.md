# Báo Cáo Phân Tích Khám Phá Dữ Liệu (Exploratory Data Analysis Report)
**Project:** Ứng dụng Khoa học dữ liệu đa yếu tố trong phân tích, dự đoán và cảnh báo sớm nguy cơ học tập kém của sinh viên

**Branch:** `feature/eda` *(đổi theo nhánh của bạn)*

**Vai trò:** Data Analyst / Visualization Engineer

---

## 1. Tổng quan Quá trình Khám phá Dữ liệu (Executive Summary)

Dữ liệu phân tích được tiếp nhận từ bước làm sạch (`dataset_with_risk.csv`) gồm **6,606 dòng** và **26 cột** (20 cột gốc sau mã hóa, cùng `Risk_Level`, `Study_Sleep_Ratio`, `Attendance_Category`, `Cluster`). EDA nhằm hiểu phân phối, mối quan hệ giữa các yếu tố học tập, sinh hoạt, gia đình và nhãn nguy cơ.

### Các phát hiện chính:
1. **Chuyên cần là yếu tố nổi bật nhất:** `Attendance` có tương quan mạnh nhất với `Exam_Score` (r = 0.58), tiếp theo là `Hours_Studied` (r = 0.45). Sinh viên nhóm `High` risk có chuyên cần trung bình 69.2%, thấp hơn nhóm `Low` (88.9%) gần 20 điểm phần trăm.
2. **Giờ ngủ gần như không liên quan đến điểm:** `Sleep_Hours` có r = -0.02; trung bình giờ ngủ của 3 nhóm nguy cơ gần như bằng nhau (7.04 / 7.03 / 7.02).
3. **Các yếu tố gia đình, trường học có ảnh hưởng nhỏ nhưng cùng chiều:** `Access_to_Resources` (r = 0.17), `Parental_Involvement` (r = 0.16), `Tutoring_Sessions` (r = 0.15).
4. **K-Means chia sinh viên thành 3 cụm** với tỷ lệ nguy cơ cao khác biệt rõ (4.7% đến 36.2%), chủ yếu phân biệt bởi chuyên cần và giờ ngủ.
5. **Các biến độc lập gần như không tương quan với nhau** (chỉ `Study_Sleep_Ratio` tương quan với `Hours_Studied` do được tính từ biến này).

---

## 2. Chi tiết Phân tích Đặc trưng & Biến Mục Tiêu

### 2.1 Phân tích Đơn biến (Univariate Analysis)
- **Biến số (Numerical):**

| Biến | Trung bình | Trung vị | Độ lệch chuẩn | Min – Max | Skewness |
|---|---|---|---|---|---|
| `Hours_Studied` | 19.97 | 20 | 5.99 | 1 – 44 | 0.01 |
| `Attendance` | 79.97 | 80 | 11.55 | 60 – 100 | 0.01 |
| `Sleep_Hours` | 7.03 | 7 | 1.47 | 4 – 10 | -0.02 |
| `Previous_Scores` | 75.07 | 75 | 14.40 | 50 – 100 | 0.00 |
| `Tutoring_Sessions` | 1.49 | 1 | 1.23 | 0 – 8 | 0.81 |
| `Physical_Activity` | 2.97 | 3 | 1.03 | 0 – 6 | -0.03 |
| `Exam_Score` | 67.23 | 67 | 3.87 | 55 – 100 | 1.58 |

  Hầu hết biến có phân bố **đối xứng** (skew ≈ 0). Chỉ `Exam_Score` (lệch phải 1.58) và `Tutoring_Sessions` (lệch phải 0.81) lệch rõ; điểm thi tập trung quanh 65–69, đuôi dài về phía điểm cao.
- **Biến mục tiêu `Risk_Level`:** High 1,452 (21.98%), Medium 2,906 (43.99%), Low 2,248 (34.03%). Tỷ lệ lớp lớn nhất/nhỏ nhất ≈ 2:1, mất cân bằng nhẹ.

### 2.2 Tương quan & Đa cộng tuyến (Correlation & Multicollinearity)
- Sử dụng **Ma trận tương quan (Heatmap)** cho các biến số. Tương quan với `Exam_Score`, sắp theo độ lớn:

| Hạng | Feature | r |
|---|---|---|
| 1 | `Attendance` | 0.582 |
| 2 | `Hours_Studied` | 0.447 |
| 3 | `Study_Sleep_Ratio` (biến mới) | 0.358 |
| 4 | `Previous_Scores` | 0.174 |
| 5 | `Access_to_Resources` | 0.171 |
| 6 | `Parental_Involvement` | 0.160 |
| 7 | `Tutoring_Sessions` | 0.154 |
| … | `Sleep_Hours`, `School_Type`, `Gender` | -0.016, 0.010, 0.000 |

- **Đa cộng tuyến:** Giữa các biến gốc gần như không có (|r| < 0.03). Cặp tương quan cao duy nhất là `Study_Sleep_Ratio` với `Hours_Studied` (r = 0.77) và `Sleep_Hours` (r = -0.58), vì biến này được tính từ hai biến kia.
- **Quyết định xử lý:** Khi dùng mô hình tuyến tính (Logistic Regression), cân nhắc chọn giữa `Study_Sleep_Ratio` hoặc cặp `Hours_Studied` + `Sleep_Hours`, không dùng cả ba. Mô hình cây (Random Forest, XGBoost) ít bị ảnh hưởng.

### 2.3 Phân tích Feature ↔ Target
**Trung bình các biến số theo nhóm nguy cơ:**

| Biến | High | Medium | Low |
|---|---|---|---|
| `Hours_Studied` | 15.91 | 19.59 | 23.10 |
| `Attendance` (%) | 69.23 | 78.42 | 88.92 |
| `Previous_Scores` | 71.37 | 74.52 | 78.16 |
| `Tutoring_Sessions` | 1.19 | 1.48 | 1.71 |
| `Sleep_Hours` | 7.04 | 7.03 | 7.02 |
| `Physical_Activity` | 2.89 | 2.99 | 2.98 |

**Điểm thi trung bình theo các yếu tố phân loại (mã hóa 0/1/2 = Low/Medium/High):**

| Yếu tố | Nhóm 0 | Nhóm 1 | Nhóm 2 |
|---|---|---|---|
| `Parental_Involvement` | 66.33 | 67.10 | 68.09 |
| `Access_to_Resources` | 66.20 | 67.12 | 68.09 |
| `Motivation_Level` | 66.73 | 67.33 | 67.70 |
| `Family_Income` | 66.85 | 67.33 | 67.82 |
| `Teacher_Quality` | 66.75 | 67.10 | 67.66 |

- Các yếu tố có **quan hệ tuyến tính đơn điệu**: mức càng cao, điểm càng cao, nhưng chênh lệch chỉ khoảng **1–2 điểm** giữa nhóm thấp và cao nhất.
- `Learning_Disabilities` = Yes: điểm 66.27 so với 67.34 (No). `Internet_Access` = Yes: 67.29 so với 66.47. `Gender` và `School_Type` gần như không khác biệt (67.23 / 67.23 và 67.21 / 67.29).

### 2.4 Phân tích Phân phối Lệch & Ngoại lệ
- `Exam_Score` lệch phải (skew 1.58) do các điểm cao; đây là **giá trị thật**, được giữ nguyên (xem `data_quality_report.md`).
- Nếu dùng mô hình nhạy với phân phối, có thể thử `np.log1p` cho `Tutoring_Sessions`. Các mô hình dựa trên cây không cần biến đổi này.

---

## 3. Feature Engineering & Phân nhóm

### 3.1 Đặc trưng mới
| Feature | Công thức | Ý nghĩa | Tương quan với `Exam_Score` |
|---|---|---|---|
| `Study_Sleep_Ratio` | `Hours_Studied / Sleep_Hours` | Cân bằng giữa học và nghỉ | 0.358 |
| `Attendance_Category` | Low: < 70; Medium: 70–84; High: ≥ 85 | Nhóm chuyên cần | — |

**Tỷ lệ nguy cơ theo nhóm chuyên cần:**

| `Attendance_Category` | Số sinh viên | Điểm TB | % High risk | % Low risk |
|---|---|---|---|---|
| Low | 1,573 | 64.21 | **56.1%** | 4.5% |
| Medium | 2,523 | 66.68 | 20.1% | 21.4% |
| High | 2,510 | 69.68 | 2.5% | 65.3% |

Đây là kết quả rõ nhất của toàn bộ EDA: chuyên cần dưới 70% khiến hơn một nửa sinh viên rơi vào nhóm nguy cơ cao.

### 3.2 Phân cụm K-Means (k = 3)
- **Feature dùng để phân cụm:** `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Physical_Activity` (đã chuẩn hóa bằng `StandardScaler`, `random_state=42`).
- **Chọn k:** Đồ thị Elbow (k = 1 đến 7) giảm đều, **không có "khuỷu" rõ rệt**; chọn k = 3 để cho ra số nhóm dễ diễn giải.
- **Chất lượng cụm:** Silhouette = 0.174, tức các cụm **tách nhau yếu**, có phần chồng lấn. Nên xem đây là phân nhóm để mô tả, không phải cấu trúc tự nhiên rõ ràng của dữ liệu.

| Cụm | Số SV | Chuyên cần | Giờ ngủ | Giờ học | Điểm thi TB | High | Medium | Low |
|---|---|---|---|---|---|---|---|---|
| 0 | 2,243 | 71.1 | **6.03** | 20.3 | 65.60 | **36.2%** | 49.9% | 13.9% |
| 1 | 2,503 | **91.5** | 6.71 | 19.2 | 69.27 | 4.7% | 36.0% | **59.3%** |
| 2 | 1,860 | 75.3 | **8.67** | 20.5 | 66.45 | 28.0% | 47.6% | 24.4% |

**Diễn giải:**
- **Cụm 0 "Chuyên cần thấp, ngủ ít"**: nguy cơ cao nhất (36.2% High).
- **Cụm 1 "Chuyên cần cao"**: nguy cơ thấp nhất (4.7% High), điểm cao nhất.
- **Cụm 2 "Chuyên cần trung bình, ngủ nhiều"**: nguy cơ trung gian (28.0% High).
- Giờ học (~19–21) và `Previous_Scores` (~75) gần như giống nhau ở cả 3 cụm, nên **chuyên cần** là yếu tố phân biệt chính.

---

## 4. Đánh Giá Chất Lượng Đặc Trưng (Feature Evaluation Summary)

| Tiêu chí | Kết quả | Đề xuất cho bước Modeling |
|---|---|---|
| **1. Missing Ratio** | 0% sau khi làm sạch | Không cần xử lý thêm. |
| **2. Skewness** | Chỉ `Exam_Score` (1.58) và `Tutoring_Sessions` (0.81) lệch rõ | Không bắt buộc biến đổi với mô hình cây. |
| **3. Outliers** | Xem `data_quality_report.md`; đều là giá trị hợp lệ | Không loại bỏ; có thể dùng `RobustScaler` nếu cần. |
| **4. Target Correlation** | Mạnh: `Attendance`, `Hours_Studied`; yếu: `Sleep_Hours`, `Physical_Activity`, `Gender`, `School_Type` (|r| < 0.03) | Cân nhắc loại các biến |r| < 0.03 khi Feature Selection. |
| **5. Multicollinearity** | Chỉ `Study_Sleep_Ratio` ↔ `Hours_Studied` (0.77) | Không đưa cả 3 biến vào mô hình tuyến tính. |

---

## 5. Kết Luận & Định Hướng Mô Hình Hóa

### Các lưu ý quan trọng
1. **Data leakage:** `Risk_Level` được tính trực tiếp từ `Exam_Score`. Khi huấn luyện mô hình dự đoán `Risk_Level`, **không đưa `Exam_Score` vào features**. `Cluster` cũng nên xem xét kỹ vì được tạo từ dữ liệu toàn bộ tập.
2. **Giới hạn của dữ liệu:** Các yếu tố lối sống (giờ ngủ, vận động) gần như không giải thích được điểm số, trong khi đề cương kỳ vọng nhiều yếu tố hành vi (Social Media, Gaming, Stress) mà dataset không có. Nên nêu rõ giới hạn này trong báo cáo cuối.
3. **Sức mạnh dự đoán có thể bị giới hạn:** Tương quan mạnh nhất cũng chỉ ở mức trung bình (0.58); các biến khác chỉ 0.1–0.2. Kết quả mô hình cần được đánh giá thận trọng.

### Khuyến nghị cho nhóm Feature Engineering & Modeling
1. **Chia dữ liệu:** dùng `stratify` theo `Risk_Level` khi tách Train/Test và Cross Validation.
2. **Mất cân bằng nhẹ:** dùng `class_weight='balanced'`; SMOTE chỉ cần nếu Recall của lớp High thấp.
3. **Đánh giá:** ưu tiên **Recall và F1 của lớp High** (bỏ sót sinh viên nguy cơ cao là lỗi tốn kém nhất), kèm F1-Macro và ROC-AUC; không chỉ dựa vào Accuracy.
4. **Giải thích (SHAP):** kỳ vọng `Attendance` và `Hours_Studied` là hai yếu tố dẫn đầu, phù hợp với kết quả tương quan ở trên.
