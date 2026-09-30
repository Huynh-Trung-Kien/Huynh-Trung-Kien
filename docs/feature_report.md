# Feature Engineering & Feature Selection

## 1. Overview

Feature Engineering chịu trách nhiệm chuyển đổi dữ liệu sinh viên đã làm sạch thành tập đặc trưng có thể đưa trực tiếp vào các mô hình Machine Learning.

Bài toán mục tiêu:

```text
Risk_Level  (dự đoán nguy cơ học tập kém)
```

Các class mục tiêu gồm:

```text
High     (nguy cơ cao)
Medium   (nguy cơ trung bình)
Low      (nguy cơ thấp)
```

Dataset đầu vào là `dataset_with_risk.csv` (6,606 dòng) với các nhóm thông tin: học tập (`Hours_Studied`, `Attendance`, `Previous_Scores`, `Tutoring_Sessions`), sinh hoạt (`Sleep_Hours`, `Physical_Activity`), gia đình (`Parental_Involvement`, `Family_Income`, ...), trường học (`Teacher_Quality`, `Access_to_Resources`, ...) và hành vi (`Motivation_Level`, `Peer_Influence`, ...).

Workflow tổng quát:

```text
                 Clean Dataset
                      │
                      ▼
            Create target: Risk_Level
                      │
                      ▼
            Feature Engineering
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
Study_Sleep_Ratio  Attendance_    Cluster
                   Category       (K-Means)
      └───────────────┼───────────────┘
                      ▼
              Feature Selection
              (|r| > 0.05)
                      │
                      ▼
            Train / Test Split
              (stratified)
                      │
                      ▼
          Scaling (StandardScaler)
                      │
                      ▼
               Machine Learning
```

---

# 2. Các thành phần xử lý

Toàn bộ pipeline hiện được cài đặt trong notebook `DATA.ipynb` (chưa tách thành module riêng). Vai trò của từng bước:

| Bước | Responsibility | Artifact lưu lại |
| --- | --- | --- |
| Tạo target | Tạo `Risk_Level` từ `Exam_Score` | `dataset_with_risk.csv` |
| Tạo đặc trưng mới | `Study_Sleep_Ratio`, `Attendance_Category`, `Cluster` | `eda_dataset_final.csv`, `kmeans_model.pkl` |
| Feature Selection | Chọn đặc trưng theo tương quan | `selected_features.pkl` |
| Chia dữ liệu | Train/Test có phân tầng | — |
| Encoding nhãn | `LabelEncoder` cho `Risk_Level` | `label_encoder.pkl` |
| Scaling | `StandardScaler` | `scaler.pkl` |
| Lưu cột đầu vào | Giữ đúng thứ tự cột cho Dashboard | `feature_columns.pkl` |

> Khuyến nghị: nếu mở rộng dự án, có thể tách thành `src/features/` (tạo feature, chọn feature) và `tests/test_features.py` như cấu trúc của các dự án modular.

---

# 3. Target Handling

Target không có sẵn trong dataset gốc mà được **tạo từ `Exam_Score`** theo quartile:

```text
Exam_Score < 65    →  High     (nguy cơ cao)
65 ≤ Exam_Score < 69  →  Medium
Exam_Score ≥ 69    →  Low
```

| Lớp | Số dòng | Tỷ lệ | Khoảng `Exam_Score` |
| --- | ---: | ---: | --- |
| High | 1,452 | 21.98% | 55 – 64 |
| Medium | 2,906 | 43.99% | 65 – 68 |
| Low | 2,248 | 34.03% | 69 – 100 |

**Quan trọng:** vì target được tính trực tiếp từ `Exam_Score`, cột này **bị loại hoàn toàn khỏi tập features** (xem mục 9).

Nhãn được mã hóa bằng `LabelEncoder`: `High = 0`, `Low = 1`, `Medium = 2`. Encoder được lưu để giải mã kết quả khi dự đoán.

---

# 4. Đặc trưng mới (Feature Creation)

## 4.1. `Study_Sleep_Ratio`

```text
Study_Sleep_Ratio = Hours_Studied / Sleep_Hours
```

Đo cân bằng giữa thời gian học và nghỉ ngơi. Giá trị từ 0.125 đến 9.5, trung bình 2.98. Tương quan với `Exam_Score`: **r = 0.358**.

*Lưu ý:* biến này được tính từ `Hours_Studied` (r = 0.77) và `Sleep_Hours` (r = -0.58) nên chứa thông tin trùng lặp. Trong Logistic Regression, hệ số của nó gần 0 (chạy lại), tức đóng góp thực tế thấp.

## 4.2. `Attendance_Category`

```text
Attendance < 70        →  Low
70 ≤ Attendance < 85   →  Medium
Attendance ≥ 85        →  High
```

| Nhóm | Số SV | % High risk |
| --- | ---: | ---: |
| Low | 1,573 | 56.1% |
| Medium | 2,523 | 20.1% |
| High | 2,510 | 2.5% |

Biến này phục vụ EDA và trực quan hóa; **không đưa vào mô hình** (là biến chữ, và trùng thông tin với `Attendance`).

## 4.3. `Cluster` (K-Means)

- Thuật toán: `KMeans(n_clusters=3, random_state=42, n_init=10)`.
- Feature phân cụm: `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Physical_Activity` (đã chuẩn hóa bằng `StandardScaler`).
- Kết quả: cụm 0 (2,243 SV), cụm 1 (2,503 SV), cụm 2 (1,860 SV). Silhouette = 0.174, tức các cụm tách nhau yếu.

---

# 5. Encoding các biến phân loại

Việc mã hóa được thực hiện ở bước làm sạch dữ liệu, nên các feature đã là số khi vào mô hình:

| Nhóm | Cột | Cách mã hóa |
| --- | --- | --- |
| Ordinal | `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality` | Low=0, Medium=1, High=2 |
| Ordinal | `Distance_from_Home` | Near=0, Moderate=1, Far=2 |
| Binary | `Extracurricular_Activities`, `Internet_Access`, `Learning_Disabilities` | Yes=1, No=0 |
| Binary | `School_Type`, `Gender` | Public/Male=0, Private/Female=1 |
| One-Hot (`drop_first=True`) | `Peer_Influence`, `Parental_Education_Level` | Cột dummy True/False |

---

# 6. Feature Selection

## 6.1. Phương pháp

Chọn các feature có **tương quan tuyệt đối với `Exam_Score` lớn hơn 0.05**:

```python
correlation_with_exam = df.corr(numeric_only=True)['Exam_Score'].abs()
selected_features = correlation_with_exam[correlation_with_exam > 0.05].index.tolist()
selected_features.remove('Exam_Score')
```

Kết quả: **18 feature** được giữ trên 26 cột.

```text
26 columns
   ├── Exam_Score, Risk_Level        (target / nguồn của target)
   ├── Attendance_Category           (biến chữ, bị bỏ khi numeric_only=True)
   ├── 5 feature |r| ≤ 0.05          (bị loại)
   ▼
18 selected features
```

## 6.2. Feature bị loại do tương quan yếu

| Feature | r với `Exam_Score` |
| --- | ---: |
| `Physical_Activity` | 0.028 |
| `Sleep_Hours` | -0.016 |
| `School_Type` | 0.010 |
| `Peer_Influence_Neutral` | -0.007 |
| `Gender` | 0.000 |

## 6.3. Feature được giữ (sắp theo |r|)

| Hạng | Feature | r | Nhóm |
| ---: | --- | ---: | --- |
| 1 | `Attendance` | 0.582 | Học tập |
| 2 | `Hours_Studied` | 0.447 | Học tập |
| 3 | `Study_Sleep_Ratio` | 0.358 | Đặc trưng mới |
| 4 | `Previous_Scores` | 0.174 | Học tập |
| 5 | `Access_to_Resources` | 0.171 | Trường học |
| 6 | `Parental_Involvement` | 0.160 | Gia đình |
| 7 | `Tutoring_Sessions` | 0.154 | Học tập |
| 8 | `Cluster` | 0.109 | Đặc trưng mới |
| 9 | `Parental_Education_Level_Postgraduate` | 0.095 | Gia đình |
| 10 | `Family_Income` | 0.093 | Gia đình |
| 11 | `Distance_from_Home` | -0.090 | Trường học |
| 12 | `Motivation_Level` | 0.089 | Hành vi |
| 13 | `Parental_Education_Level_High School` | -0.089 | Gia đình |
| 14 | `Learning_Disabilities` | -0.085 | Cá nhân |
| 15 | `Peer_Influence_Positive` | 0.080 | Hành vi |
| 16 | `Teacher_Quality` | 0.075 | Trường học |
| 17 | `Extracurricular_Activities` | 0.064 | Hành vi |
| 18 | `Internet_Access` | 0.056 | Trường học |

## 6.4. Hạn chế của phương pháp

- Ngưỡng 0.05 là chọn thủ công; `Extracurricular_Activities` (0.064) và `Internet_Access` (0.056) chỉ vừa đủ qua ngưỡng.
- Chỉ đo **tương quan tuyến tính với `Exam_Score`**, không phải với `Risk_Level` và không bắt được quan hệ phi tuyến.
- Tương quan được tính trên **toàn bộ dữ liệu trước khi chia Train/Test** (xem mục 9).
- Đề cương còn nêu Feature Importance; kết quả này có thể bổ sung từ SHAP (mục 10).

---

# 7. Train/Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

| Tập | Số dòng |
| --- | ---: |
| Train | 5,284 |
| Test | 1,322 |

Phân tầng (`stratify`) giữ nguyên tỷ lệ 3 lớp: tập test có High 291 (22.0%), Medium 581 (43.9%), Low 450 (34.0%), khớp phân bố gốc.

---

# 8. Scaling & Class Imbalance

## 8.1. Scaling

Dùng `StandardScaler`, **chỉ `fit` trên tập train** rồi `transform` cho tập test:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

Mô hình cuối (`best_model`) là `make_pipeline(StandardScaler(), LogisticRegression(...))`, nên khi dự đoán trên dữ liệu mới không cần chuẩn hóa thủ công. Scaler được lưu riêng trong `scaler.pkl` cho Dashboard.

## 8.2. Class Imbalance

Tỷ lệ lớp lớn nhất/nhỏ nhất chỉ khoảng **2:1** (Medium 43.99% so với High 21.98%), nên **không áp dụng SMOTE hay `class_weight`**. Kết quả trên tập test cho thấy recall giữa các lớp cân bằng (0.92 – 0.95), xác nhận quyết định này hợp lý.

---

# 9. Data Leakage Prevention

| Nguy cơ | Trạng thái | Ghi chú |
| --- | --- | --- |
| `Exam_Score` trong features | **Đã loại** | Target được tính trực tiếp từ cột này |
| Scaler fit trên toàn bộ dữ liệu | **Đã tránh** | Scaler chỉ fit trên train |
| Feature Selection tính trên toàn bộ dữ liệu | **Còn tồn tại** | Tương quan tính trước khi chia Train/Test; ảnh hưởng nhỏ vì chỉ dùng để lọc 5 biến tương quan rất thấp |
| K-Means fit trên toàn bộ dữ liệu | **Còn tồn tại** | Cụm được tạo từ cả tập test |
| `Cluster` chứa thông tin của biến đã loại | **Cần lưu ý** | Tạo từ `Sleep_Hours`, `Physical_Activity` (đã bị loại khỏi features); tương quan `Cluster`–`Sleep_Hours` = 0.69 |

**Kiểm chứng (chạy lại):** khi bỏ `Cluster` khỏi tập features, Logistic Regression đạt 93.6% so với 93.5% khi có, tức `Cluster` không đóng góp thêm. Khuyến nghị **bỏ `Cluster` khỏi tập features** để tránh rò rỉ, chỉ dùng cho EDA.

Cách làm chuẩn: `fit` K-Means và tính tương quan trên tập train, sau đó áp dụng lên tập test.

---

# 10. Kiểm chứng Đặc trưng qua Mô hình

Mô hình cuối: **Logistic Regression** (đã tuning bằng `GridSearchCV`: `C ∈ {0.01, 0.1, 1, 10, 100}`, `solver ∈ {lbfgs, liblinear}`, `cv=5`, `scoring='recall_macro'`). Kết quả trên tập test 1,322 dòng:

| Lớp | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| High | 0.931 | 0.928 | 0.929 |
| Medium | 0.927 | 0.923 | 0.925 |
| Low | 0.943 | 0.951 | 0.947 |

**Accuracy = 93.34%, F1-Macro = 0.934.** Không có sinh viên High nào bị dự đoán thành Low; lỗi chủ yếu xảy ra giữa các lớp liền kề.

**Mức độ ảnh hưởng của feature** (hệ số chuẩn hóa của Logistic Regression cho lớp High, *chạy lại*; dấu âm = giá trị càng cao thì nguy cơ càng thấp):

| Hạng | Feature | Hệ số |
| ---: | --- | ---: |
| 1 | `Attendance` | -6.20 |
| 2 | `Hours_Studied` | -4.79 |
| 3 | `Parental_Involvement` | -1.94 |
| 4 | `Access_to_Resources` | -1.93 |
| 5 | `Previous_Scores` | -1.82 |
| 6 | `Tutoring_Sessions` | -1.65 |
| 7 | `Motivation_Level` | -1.04 |

`Attendance` và `Hours_Studied` vượt trội hẳn, khớp với thứ hạng tương quan ở mục 6.3.

**Đầu ra dùng các feature này:**
- **Risk Score** = xác suất lớp High × 100 (thang 0–100).
- **Early Warning:** ≥ 70 → 🔴 Cảnh báo cao; 40–69 → 🟡 Cần theo dõi; < 40 → 🟢 Bình thường.
- **Recommendation:** lấy feature có SHAP value dương lớn nhất (đẩy nguy cơ High lên nhiều nhất) rồi ánh xạ sang lời khuyên; 5 feature có câu khuyến nghị riêng (`Attendance`, `Hours_Studied`, `Study_Sleep_Ratio`, `Motivation_Level`, `Previous_Scores`), còn lại dùng câu chung "Cần chú ý cải thiện yếu tố: …".

Trên tập test: 250/259 (96.5%) sinh viên bị cảnh báo đỏ thực sự thuộc nhóm High; 275/291 (94.5%) sinh viên High được gắn cảnh báo đỏ hoặc vàng; 16 sinh viên High (5.5%) bị xếp vào "Bình thường".

---

# 11. Output Artifacts

```text
01-Datasheet/
├── clean_dataset.csv
├── dataset_with_risk.csv
├── eda_dataset_final.csv
├── final_results_with_risk_score.csv
├── best_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── selected_features.pkl
├── feature_columns.pkl
├── kmeans_model.pkl
└── all_models_results.pkl
```

Khi dự đoán trên sinh viên mới:

```text
New Student
    ↓
Tạo feature (Study_Sleep_Ratio, ...)
    ↓
Chọn đúng cột theo feature_columns.pkl
    ↓
Scaler / Pipeline đã lưu
    ↓
best_model
    ↓
Risk Score → Early Warning → Recommendation
```

---

# 12. Testing

Nên kiểm thử tại `tests/test_features.py`:

```text
Target creation
    ↓
Feature creation
    ↓
Feature selection
    ↓
Scaling
    ↓
Complete pipeline
```

Các trường hợp cần kiểm tra:

```text
Exam_Score = 64  → Risk_Level = High
Exam_Score = 65  → Risk_Level = Medium
Exam_Score = 69  → Risk_Level = Low

Sleep_Hours > 0
    → Study_Sleep_Ratio không bị lỗi chia cho 0

Attendance = 69 / 70 / 84 / 85
    → Attendance_Category = Low / Medium / Medium / High

Exam_Score
    → không có trong selected_features

Tập test
    → scaler không được fit lại
```

---

# 13. Summary

Feature Engineering của dự án biến dữ liệu sinh viên đã làm sạch thành tập đặc trưng cho mô hình dự đoán nguy cơ học tập kém thông qua các bước:

```text
1. Tạo target Risk_Level từ Exam_Score
2. Tạo feature mới: Study_Sleep_Ratio, Attendance_Category, Cluster
3. Chọn 18 feature có |r| > 0.05 với Exam_Score
4. Chia Train/Test có phân tầng, chuẩn hóa bằng StandardScaler
```

Kết quả:

```text
26 columns
     ↓
18 selected features
     ↓
Logistic Regression (tuned)
     ↓
Accuracy 93.3% · F1-Macro 0.934
     ↓
Risk Score → Early Warning → Recommendation
```

Hai yếu tố quyết định nhất là **`Attendance`** và **`Hours_Studied`**. Ba điểm cần cải thiện nếu có thời gian: bỏ `Cluster` khỏi tập features, tính tương quan chỉ trên tập train, và bổ sung SHAP summary plot để xác nhận thứ hạng feature.
