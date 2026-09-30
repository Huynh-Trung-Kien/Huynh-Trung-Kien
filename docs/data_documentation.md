# Data Documentation — Dự đoán và cảnh báo sớm nguy cơ học tập kém của sinh viên

**Project:** Ứng dụng Khoa học dữ liệu đa yếu tố trong phân tích, dự đoán và cảnh báo sớm nguy cơ học tập kém của sinh viên
**Branch:** `feature/data` *(đổi theo nhánh của bạn)*
**Vai trò:** Data Engineer / Data Analyst

> Số liệu được tính trực tiếp từ `StudentPerformanceFactors.csv` (raw) và `dataset_with_risk.csv` (đã làm sạch).

---

## 1. Dataset ban đầu (Data Collection)

- **Nguồn dữ liệu:** Kaggle — Student Performance Factors (tải dưới dạng CSV).
- **Tên file:** `StudentPerformanceFactors.csv`
- **Dữ liệu raw:** lưu nguyên vẹn tại `data/raw/` (không sửa trực tiếp).
- **Công cụ:** Python/Pandas (đọc, kiểm tra, làm sạch), MySQL (lưu trữ bảng `students`), SQL (truy vấn).

### Tổng quan bảng dữ liệu

| Bảng dữ liệu | Số dòng | Số cột | Khóa chính / Liên kết | Mô tả vai trò |
|---|---|---|---|---|
| `StudentPerformanceFactors.csv` (raw) | 6,607 | 20 | Không có khóa định danh | Bảng duy nhất: thông tin học tập, gia đình, môi trường của sinh viên |
| `clean_dataset.csv` | 6,606 | 22 | — | Dữ liệu sau làm sạch và mã hóa |
| `dataset_with_risk.csv` | 6,606 | 26 | — | Clean dataset + `Risk_Level` + `Study_Sleep_Ratio`, `Attendance_Category`, `Cluster` |

---

## 2. Phân loại Feature (Data Understanding)

1. **ACADEMIC (Học tập):**
   - `Hours_Studied`: số giờ tự học
   - `Attendance`: tỷ lệ chuyên cần (%)
   - `Previous_Scores`: điểm kỳ trước
   - `Tutoring_Sessions`: số buổi học thêm
   - `Exam_Score`: điểm thi (dùng để tạo nhãn rủi ro)
2. **LIFESTYLE (Sinh hoạt):**
   - `Sleep_Hours`: số giờ ngủ
   - `Physical_Activity`: mức độ vận động
   - `Extracurricular_Activities`: hoạt động ngoại khóa (Yes/No)
3. **BEHAVIOR / MOTIVATION:**
   - `Motivation_Level`: động lực học (Low/Medium/High)
   - `Peer_Influence`: ảnh hưởng từ bạn bè (Positive/Neutral/Negative)
4. **FAMILY & SOCIAL:**
   - `Parental_Involvement`, `Family_Income`, `Parental_Education_Level`
5. **SCHOOL & RESOURCES:**
   - `Teacher_Quality`, `Access_to_Resources`, `School_Type`, `Distance_from_Home`, `Internet_Access`
6. **PERSONAL:**
   - `Gender`, `Learning_Disabilities`
7. **TARGET (tạo thêm):**
   - `Risk_Level`: Low / Medium / High, suy ra từ `Exam_Score`

### Bảng thuộc tính chi tiết (20 cột của dữ liệu gốc)

| # | Thuộc tính | Nhóm | Kiểu dữ liệu | Miền giá trị / Các giá trị | Missing | Mô tả |
|---|---|---|---|---|---|---|
| 1 | `Hours_Studied` | Academic | Số nguyên | 1 – 44 (TB 19.98) | 0 | Số giờ tự học mỗi tuần |
| 2 | `Attendance` | Academic | Số nguyên | 60 – 100 (TB 79.98) | 0 | Tỷ lệ chuyên cần (%) |
| 3 | `Previous_Scores` | Academic | Số nguyên | 50 – 100 (TB 75.07) | 0 | Điểm kỳ trước |
| 4 | `Tutoring_Sessions` | Academic | Số nguyên | 0 – 8 (TB 1.49) | 0 | Số buổi học thêm mỗi tháng |
| 5 | `Exam_Score` | Academic / Target | Số nguyên | 55 – 101 (TB 67.24) | 0 | Điểm thi cuối kỳ, dùng để tạo `Risk_Level` |
| 6 | `Sleep_Hours` | Lifestyle | Số nguyên | 4 – 10 (TB 7.03) | 0 | Số giờ ngủ mỗi ngày |
| 7 | `Physical_Activity` | Lifestyle | Số nguyên | 0 – 6 (TB 2.97) | 0 | Số giờ vận động mỗi tuần |
| 8 | `Extracurricular_Activities` | Lifestyle | Phân loại (2) | Yes, No | 0 | Có tham gia hoạt động ngoại khóa |
| 9 | `Motivation_Level` | Behavior | Phân loại thứ bậc (3) | Low, Medium, High | 0 | Mức độ động lực học tập |
| 10 | `Peer_Influence` | Behavior | Phân loại danh nghĩa (3) | Positive, Neutral, Negative | 0 | Ảnh hưởng từ bạn bè |
| 11 | `Parental_Involvement` | Family | Phân loại thứ bậc (3) | Low, Medium, High | 0 | Mức độ quan tâm của phụ huynh |
| 12 | `Family_Income` | Family | Phân loại thứ bậc (3) | Low, Medium, High | 0 | Mức thu nhập gia đình |
| 13 | `Parental_Education_Level` | Family | Phân loại danh nghĩa (3) | High School, College, Postgraduate | 90 (1.36%) | Trình độ học vấn phụ huynh |
| 14 | `Teacher_Quality` | School | Phân loại thứ bậc (3) | Low, Medium, High | 78 (1.18%) | Chất lượng giáo viên |
| 15 | `Access_to_Resources` | School | Phân loại thứ bậc (3) | Low, Medium, High | 0 | Mức độ tiếp cận tài liệu, học liệu |
| 16 | `School_Type` | School | Phân loại (2) | Public, Private | 0 | Loại hình trường |
| 17 | `Distance_from_Home` | School | Phân loại thứ bậc (3) | Near, Moderate, Far | 67 (1.01%) | Khoảng cách từ nhà đến trường |
| 18 | `Internet_Access` | School | Phân loại (2) | Yes, No | 0 | Có truy cập Internet |
| 19 | `Gender` | Personal | Phân loại (2) | Male, Female | 0 | Giới tính |
| 20 | `Learning_Disabilities` | Personal | Phân loại (2) | Yes, No | 0 | Có khó khăn trong học tập |

**Thuộc tính thêm ở bước xử lý sau:** `Risk_Level` (Low/Medium/High, nhãn dự đoán), `Study_Sleep_Ratio` (= `Hours_Studied` / `Sleep_Hours`), `Attendance_Category` (nhóm chuyên cần), `Cluster` (nhóm K-Means).

> Lưu ý: dataset này **không có** một số biến nêu trong đề cương như `Social Media`, `Gaming`, `Stress`, `Part-time Job`. Cần ghi rõ giới hạn này trong báo cáo.

---

## 3. Data Quality Issues phát hiện & Cách xử lý

| # | Vấn đề phát hiện | Feature ảnh hưởng | Số record | Mức độ | Cách xử lý | Lý do xử lý |
|---|---|---|---|---|---|---|
| 1 | Missing value | `Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home` | 78 (1.18%), 90 (1.36%), 67 (1.01%); tổng 235 ô | Trung bình | Impute bằng mode (giá trị xuất hiện nhiều nhất) | Biến phân loại nên dùng mode; giữ record để không mất dữ liệu |
| 2 | Dòng trùng lặp (Duplicate) | Toàn bộ dòng | 0 | Không có | Đã kiểm tra, không cần xử lý | Dataset không có bản ghi trùng |
| 3 | `Exam_Score` > 100 (ngoài thang điểm) | `Exam_Score` | 1 (điểm = 101) | Cao | Loại bỏ record | Điểm vượt thang là lỗi nhập liệu, ảnh hưởng nhãn `Risk_Level` |
| 4 | `Attendance` > 100, `Hours_Studied` < 0, `Sleep_Hours` < 0 | 3 cột tương ứng | 0 | Không có | Đã kiểm tra, không cần xử lý | Không có giá trị bất khả thi |
| 5 | Dữ liệu dạng chữ | Các cột object | — | Bắt buộc xử lý | Encoding (xem mục 3.1) | Mô hình cần dữ liệu số |
| 6 | Outlier `Exam_Score` (IQR, Boxplot) | `Exam_Score` | 104 (ngưỡng hợp lệ [59, 75]) | Thấp | Giữ nguyên, không loại | Điểm cao/thấp là kết quả thật của sinh viên; loại bỏ sẽ làm mất chính nhóm nguy cơ cần dự đoán |

### 3.1 Mã hóa dữ liệu (Encoding)

| Nhóm | Cột | Cách mã hóa |
|---|---|---|
| Ordinal Low/Medium/High | `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality` | Low=0, Medium=1, High=2 |
| Ordinal riêng | `Distance_from_Home` | Near=0, Moderate=1, Far=2 |
| Nhị phân | `Extracurricular_Activities`, `Internet_Access`, `Learning_Disabilities` | Yes=1, No=0 |
| Nhị phân | `School_Type` | Public=0, Private=1 |
| Nhị phân | `Gender` | Male=0, Female=1 |
| Nominal | `Peer_Influence`, `Parental_Education_Level` | One-Hot Encoding (`drop_first=True`) |

`StandardScaler` được áp dụng riêng ở bước Modeling (không lưu vào clean dataset).

### 3.2 Biến mục tiêu `Risk_Level`

- Tạo từ `Exam_Score` theo quartile: `< Q1 (65)` → **High**, `< Q3 (69)` → **Medium**, còn lại → **Low**.
- Khoảng điểm: High 55–64, Medium 65–68, Low 69–100.
- Phân bố lớp: High 1,452 (21.98%) / Medium 2,906 (43.99%) / Low 2,248 (34.03%).

---

## 4. Số record trước / sau cleaning

| Bảng dữ liệu | Số dòng Trước | Số dòng Sau | Số dòng Bị loại | Tỷ lệ bảo toàn |
|---|---|---|---|---|
| `StudentPerformanceFactors.csv` | 6,607 | 6,606 | 1 | 99.98% |

> **Nhận xét:** Ưu tiên giữ lại record khi thông tin cốt lõi vẫn dùng được (missing được impute thay vì xóa). Chỉ loại 1 dòng có `Exam_Score` = 101 (ngoài thang điểm). Không có dòng trùng lặp. 104 outlier theo IQR được giữ lại.

---

## 5. Column được giữ lại / Khuyến nghị xử lý tiếp theo

- **Giữ lại:** toàn bộ 22 cột của `clean_dataset.csv` (sau đó thêm `Risk_Level`, `Study_Sleep_Ratio`, `Attendance_Category`, `Cluster`).
- **Khuyến nghị cho bước EDA / Feature Engineering:**
  - Tạo `Study_Sleep_Ratio`, `Attendance_Category`, `Extra_Engagement`.
  - Chọn feature theo tương quan với `Exam_Score` (|r| > 0.05) và Feature Importance.
  - Không đưa `Exam_Score` vào tập feature khi huấn luyện dự đoán `Risk_Level` (tránh data leakage).

---

## 6. Output bàn giao

1. **Dữ liệu:** `clean_dataset.csv`, `dataset_with_risk.csv`, `eda_dataset_final.csv`, `final_results_with_risk_score.csv`.
2. **Bảng MySQL:** `students` (đẩy từ `clean_dataset.csv` bằng SQLAlchemy/PyMySQL).
3. **Model & artifact:** `best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `selected_features.pkl`, `feature_columns.pkl`, `all_models_results.pkl`.
4. **Notebook:** `DATA.ipynb` *(nên tách thành `01_data_understanding`, `02_data_cleaning`, `03_eda`, `04_modeling`)*.
5. **Dashboard:** `app.py` (Streamlit: tổng quan + What-if Analysis).
6. **Báo cáo:** `docs/data_documentation.md` (tài liệu này), `docs/data_quality_report.md`.
