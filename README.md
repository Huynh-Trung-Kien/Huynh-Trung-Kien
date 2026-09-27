# Student Academic Risk Prediction — Technical Documentation

## Ứng dụng Khoa học dữ liệu đa yếu tố trong phân tích, dự đoán và cảnh báo sớm nguy cơ học tập kém của sinh viên

---

## 1. Data Understanding & Data Cleaning

### 1.1 Data Collection

- Verify dataset source: Kaggle – Student Performance / Student Habits Dataset.
- Store original data in `data/raw/`.
- Never directly modify raw data.
- Record dataset information:
  - Source
  - Number of records
  - Number of features
  - Dataset scope (thời gian, đối tượng thu thập)

### 1.2 Data Understanding

Phân tích:
- Shape (số dòng, số cột)
- Column names
- Data types
- Missing values
- Unique values
- Cardinality
- Invalid values
- Numerical và categorical features

**Nhóm đặc trưng quan trọng:**

```
HỌC TẬP
├── Study_Hours
├── Attendance
├── Previous_Score
├── Assignment_Completion
└── Exam_Score

SINH HOẠT
├── Sleep_Hours
├── Exercise
└── Part_time_Job

CÔNG NGHỆ
├── Social_Media
├── Gaming
└── Internet_Usage

HÀNH VI
├── Stress
└── Motivation
```

### 1.3 Data Quality

Kiểm tra:
- Missing values
- Duplicate records
- Incorrect data types
- Invalid categories (ví dụ giá trị không thuộc danh sách hợp lệ)
- Inconsistent categories (ví dụ "Yes"/"yes"/"Y" cùng nghĩa nhưng viết khác nhau)
- Negative hoặc impossible values (Study_Hours < 0, Attendance > 100)
- Extreme values

Mỗi vấn đề nên ghi chú lại:

| Trường | Nội dung |
|---|---|
| Problem | Mô tả vấn đề |
| Affected Records | Số bản ghi bị ảnh hưởng |
| Affected Feature | Cột bị ảnh hưởng |
| Impact | Ảnh hưởng nếu không xử lý |
| Treatment | Cách xử lý đã áp dụng |
| Reason | Lý do chọn cách xử lý đó |

### 1.4 Output

- Clean Dataset
- Data Quality Report
- Data Understanding Notebook
- Data Cleaning Notebook
- Reusable Cleaning Functions

---

## 2. EDA, Statistics & Visualization

### 2.1 Univariate Analysis

**Numerical Features** (Study_Hours, Attendance, Sleep_Hours, Exam_Score...)

Thống kê:
- Mean, Median, Standard Deviation
- Min/Max, Quartiles, IQR
- Skewness
- Distribution
- Outliers

Trực quan hóa: Histogram, KDE, Boxplot

**Categorical Features** (Part_time_Job, Gender, Attendance_Category...)

Phân tích: Frequency, Percentage, Cardinality, Class imbalance

Trực quan hóa: Bar Chart, Count Plot, Percentage Chart

### 2.2 Feature ↔ Feature Analysis

**Numerical ↔ Numerical**

Ví dụ: Study_Hours ↔ Exam_Score, Sleep_Hours ↔ Stress, Social_Media ↔ Exam_Score

Phương pháp: Pearson Correlation, Spearman Correlation, Correlation Matrix, Heatmap, Scatter Plot

Mục tiêu: xác định tương quan, phát hiện dư thừa (redundancy), phát hiện đa cộng tuyến (multicollinearity)

### 2.3 Feature ↔ Target Analysis

Target: `Risk_Level` (Low / Medium / High)

Phân tích:
- Numerical Feature ↔ Risk Level (ví dụ: Study_Hours ↔ Risk Level)
- Categorical Feature ↔ Risk Level (ví dụ: Part_time_Job ↔ Risk Level)

Phương pháp thống kê có thể dùng: ANOVA, Kruskal-Wallis, Chi-square, Effect Size, Group Comparison

> Tương quan hoặc liên hệ không đồng nghĩa với quan hệ nhân quả (correlation does not imply causation).

### 2.4 Categorical ↔ Numerical Analysis

Ví dụ: Risk_Level ↔ Study_Hours, Risk_Level ↔ Attendance, Part_time_Job ↔ Exam_Score

Trực quan hóa: Boxplot, Violin Plot, Bar Chart, Grouped Bar Chart

### 2.5 Categorical ↔ Categorical Analysis

Ví dụ: Risk_Level ↔ Part_time_Job, Risk_Level ↔ Attendance_Category

Phương pháp: Crosstab, Percentage, Chi-square test

Trực quan hóa: Stacked Bar Chart, Grouped Bar Chart

### 2.6 Multivariate Analysis & Clustering

- Phân tích nhiều biến số cùng lúc, phát hiện tương tác (interaction) và đa cộng tuyến
- **K-Means Clustering:** phân nhóm sinh viên theo đặc điểm học tập/sinh hoạt (unsupervised learning), không cần biết trước nhãn
- Phân tích đặc điểm từng Cluster, so sánh sự khác biệt giữa các nhóm

### 2.7 Outlier Analysis

Phân tích outlier cho: Study_Hours, Attendance, Sleep_Hours, Exam_Score...

Phương pháp: IQR, Z-score

> Outlier không nên tự động bị xóa. Cần phân biệt giữa **Data Error** (lỗi nhập liệu) và **Genuine Extreme Value** (giá trị cực đoan nhưng có thật, ví dụ sinh viên thực sự học rất ít giờ).

### 2.8 Class Imbalance

Phân tích phân bố các lớp: Low / Medium / High

Kết quả này được dùng làm đầu vào cho bước Modeling (Bước 5) — quyết định có cần xử lý mất cân bằng lớp hay không.

### 2.9 Business Insights

Trả lời các câu hỏi như:
- Mức độ nguy cơ nào (Low/Medium/High) xuất hiện nhiều nhất?
- Yếu tố nào tương quan mạnh nhất với Exam Score?
- Có hiện tượng đa cộng tuyến giữa các biến sinh hoạt không?
- Nhóm sinh viên nào (theo Cluster) có nguy cơ cao nhất?
- Việc làm thêm (Part-time Job) có liên hệ với nguy cơ học tập kém không?
- Thời gian sử dụng mạng xã hội/game có khác biệt rõ rệt giữa nhóm Low và High risk không?

---

## 3. Feature Engineering & Preprocessing

### 3.1 Numerical Feature Engineering

Các đặc trưng có thể tạo:

```
Study_Sleep_Ratio     = Study_Hours / Sleep_Hours
Digital_Usage         = Social_Media + Gaming + Internet_Usage
Attendance_Category   = Low / Medium / High (dựa trên khoảng Attendance)
```

### 3.2 Categorical Feature Engineering

Kỹ thuật có thể dùng:
- Label Encoding (cho biến nhị phân như Part_time_Job)
- One-Hot Encoding (cho biến đa lớp không có thứ tự)
- Ordinal Encoding (cho biến có thứ tự như Attendance_Category)

### 3.3 Feature Selection

Dựa trên EDA và phân tích thống kê để xác định:
- Biến hữu ích
- Biến dư thừa
- Biến tương quan cao (multicollinearity)

Kỹ thuật có thể dùng: Correlation, Variance, Mutual Information, SelectKBest, Feature Importance

### 3.4 Data Leakage Prevention

Cần đặc biệt lưu ý: **không dùng `Exam_Score` làm feature đầu vào** khi target (`Risk_Level`) được tạo trực tiếp từ `Exam_Score` — đây là dạng target leakage rõ ràng nhất trong đề tài này.

Kiểm tra thêm:
- Train/Test Leakage
- Temporal Leakage (nếu dữ liệu có yếu tố thời gian)

### 3.5 Preprocessing

Thực hiện: Scaling (StandardScaler), Encoding, Imputation

Nguyên tắc bắt buộc:

```
Fit      → chỉ trên Training Data
Transform → áp dụng cho Training / Validation / Test
```

---

## 4. Modeling & Evaluation

### 4.1 Baseline Model

Bắt đầu với **Logistic Regression** để thiết lập baseline — mốc so sánh cho các mô hình phức tạp hơn.

### 4.2 Candidate Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM

Tất cả các mô hình cần được so sánh bằng cùng 1 chiến lược đánh giá (cùng tập Train/Test, cùng metric).

### 4.3 Class Imbalance

Phương pháp có thể dùng: `class_weight`, Random Over Sampling, SMOTE

> Chỉ áp dụng xử lý mất cân bằng lớp trên tập Training, không áp dụng lên tập Test.

### 4.4 Cross Validation

Dùng để đánh giá:
- Độ ổn định của mô hình
- Variance
- Khả năng tổng quát hóa (generalization)

Ghi lại: CV Mean, CV Standard Deviation, Scores by Fold

### 4.5 Hyperparameter Tuning

Phương pháp: GridSearchCV, RandomizedSearchCV

Tham số tiềm năng: Regularization, Tree Depth, Number of Estimators, Learning Rate

### 4.6 Bias & Variance Analysis

**High Bias (Underfitting):**
```
Training Performance   → Thấp
Validation Performance → Thấp
Test Performance       → Thấp
```

**High Variance (Overfitting):**
```
Training Performance   → Rất cao
Validation Performance → Thấp
Test Performance       → Thấp
```

### 4.7 Evaluation

Không chỉ dựa vào Accuracy. Cần đánh giá:
- Accuracy
- Precision, Recall
- F1-score, Macro-F1, Weighted-F1
- Confusion Matrix

> Khi dữ liệu mất cân bằng lớp đáng kể, **Macro-F1** cần được chú trọng đặc biệt.

### 4.8 Model Comparison

| Model | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | CV Mean | CV Std |
|---|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | | |
| Decision Tree | | | | | | | |
| Random Forest | | | | | | | |
| XGBoost | | | | | | | |
| SVM | | | | | | | |

Mô hình cuối cùng nên được chọn dựa trên tiêu chí đánh giá tổng thể của đề tài (ưu tiên Recall/Macro-F1 vì bỏ sót sinh viên nguy cơ cao nghiêm trọng hơn cảnh báo nhầm), chứ không chỉ dựa vào 1 chỉ số duy nhất.

### 4.9 Model Interpretation (SHAP)

Trả lời câu hỏi: **Mô hình dựa vào yếu tố nào để xác định mức độ nguy cơ?**

Kỹ thuật: Feature Importance, Permutation Importance, SHAP, Coefficient Analysis

### 4.10 Model Artifacts

Lưu lại các thành phần tái sử dụng:
- Best Model (`best_model.pkl`)
- Scaler (`scaler.pkl`)
- Encoder
- Label Mapping (Low/Medium/High ↔ số)

Các artifact này phục vụ cho workflow dự đoán (prediction) sau này.

---

## 5. Integration, Pipeline & Documentation

### 5.1 End-to-End Pipeline

```
Raw Dataset
      ↓
Validation
      ↓
Cleaning
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
Training
      ↓
Evaluation
      ↓
Save Model
      ↓
Risk Score → Early Warning → Recommendation
      ↓
Dashboard & What-if Analysis
```

### 5.2 Configuration Files (nếu tách pipeline thành module)

```
configs/
├── data.yaml       # đường dẫn dữ liệu, tham số cleaning, tham số split
├── features.yaml   # tham số feature engineering
├── model.yaml       # tham số mô hình, hyperparameter, tham số đánh giá
└── pipeline.yaml    # cấu hình pipeline, đường dẫn output
```

### 5.3 Testing

```
tests/
├── test_data.py       # kiểm tra Data Cleaning
├── test_features.py   # kiểm tra Feature Engineering
├── test_model.py       # kiểm tra Modeling
└── test_pipeline.py    # kiểm tra toàn bộ pipeline
```

---

## Ghi chú áp dụng cho đồ án cá nhân

Tài liệu này được điều chỉnh từ cấu trúc phân công 5 vai trò của một dự án nhóm (Data, Visualization, Feature Engineering, Modeling, Pipeline). Nếu bạn làm đồ án **cá nhân**, bạn vẫn thực hiện đầy đủ các phần trên nhưng theo trình tự thời gian (Bước 1 → 5 như đề tài gốc) thay vì chia theo người — mỗi mục lớn (1 đến 5) ở trên tương ứng với 1 giai đoạn bạn tự thực hiện tuần tự.
