# Data Documentation — Dự đoán và cảnh báo sớm nguy cơ học tập kém của sinh viên

**Đề tài:** Ứng dụng Khoa học dữ liệu đa yếu tố trong phân tích, dự đoán và cảnh báo sớm nguy cơ học tập kém của sinh viên

> **Mục đích tài liệu:** mô tả dữ liệu, các lỗi phát hiện, cách xử lý (kèm lý do) và bàn giao cho các bước sau (EDA, Feature Engineering, Modeling). Chi tiết phân tích chất lượng xem `docs/data_quality_report.md`.

---

## 1. Dataset ban đầu (Data Collection)

- **Nguồn dữ liệu:** Kaggle — Student Performance Factors (tải dưới dạng CSV).
- **Link / ngày tải / giấy phép (license):** `[TODO: điền link Kaggle, ngày tải, license ghi trên trang dataset]`
- **Tên file:** `StudentPerformanceFactors.csv`
- **Dữ liệu raw:** lưu nguyên vẹn tại `data/raw/` (không sửa trực tiếp).
- **Công cụ:** Python/Pandas (đọc, kiểm tra, làm sạch), MySQL (lưu bảng `students`), SQL (truy vấn).
- **Đặc điểm dữ liệu:**
  - Dữ liệu cắt ngang (mỗi sinh viên một dòng, không có chiều thời gian).
  - Không có cột định danh sinh viên (không có ID).
  - `[TODO: kiểm tra mô tả trên Kaggle xem dataset có phải dữ liệu tổng hợp (synthetic) hay không và ghi rõ ở đây]`

### Tổng quan bảng dữ liệu

| Bảng dữ liệu | Số dòng | Số cột | Khóa chính / Liên kết | Mô tả vai trò |
|---|---|---|---|---|
| `StudentPerformanceFactors.csv` (raw) | 6,607 | 20 | Không có khóa định danh | Bảng duy nhất: thông tin học tập, gia đình, môi trường của sinh viên |
| `clean_dataset.csv` | 6,606 | 22 | — | Dữ liệu sau làm sạch và mã hóa |
| `dataset_with_risk.csv` | 6,606 | 26 | — | Clean dataset + `Risk_Level`, `Study_Sleep_Ratio`, `Attendance_Category`, `Cluster` |

> Cách tính số cột: 20 cột gốc → 22 cột sau mã hóa (`Peer_Influence` và `Parental_Education_Level` mỗi cột có 3 giá trị, one-hot với `drop_first=True` thành 2 cột mới, tức +1 cột mỗi biến) → 26 cột sau khi thêm 4 cột (`Risk_Level`, `Study_Sleep_Ratio`, `Attendance_Category`, `Cluster`).

---

## 2. Phân loại Feature (Data Understanding)

1. **ACADEMIC (Học tập):**
   - `Hours_Studied`: số giờ tự học mỗi tuần
   - `Attendance`: tỷ lệ chuyên cần (%)
   - `Previous_Scores`: điểm kỳ trước
   - `Tutoring_Sessions`: số buổi học thêm mỗi tháng
   - `Exam_Score`: điểm thi (dùng để tạo nhãn rủi ro)
2. **LIFESTYLE (Sinh hoạt):**
   - `Sleep_Hours`: số giờ ngủ mỗi ngày
   - `Physical_Activity`: số giờ vận động mỗi tuần
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
| 13 | `Parental_Education_Level` | Family | Phân loại (3) | High School, College, Postgraduate | 90 (1.36%) | Trình độ học vấn phụ huynh |
| 14 | `Teacher_Quality` | School | Phân loại thứ bậc (3) | Low, Medium, High | 78 (1.18%) | Chất lượng giáo viên |
| 15 | `Access_to_Resources` | School | Phân loại thứ bậc (3) | Low, Medium, High | 0 | Mức độ tiếp cận tài liệu, học liệu |
| 16 | `School_Type` | School | Phân loại (2) | Public, Private | 0 | Loại hình trường |
| 17 | `Distance_from_Home` | School | Phân loại thứ bậc (3) | Near, Moderate, Far | 67 (1.01%) | Khoảng cách từ nhà đến trường |
| 18 | `Internet_Access` | School | Phân loại (2) | Yes, No | 0 | Có truy cập Internet |
| 19 | `Gender` | Personal | Phân loại (2) | Male, Female | 0 | Giới tính |
| 20 | `Learning_Disabilities` | Personal | Phân loại (2) | Yes, No | 0 | Có khó khăn trong học tập |

**Thuộc tính thêm ở bước xử lý sau:** `Risk_Level` (Low/Medium/High, nhãn dự đoán), `Study_Sleep_Ratio` (= `Hours_Studied` / `Sleep_Hours`), `Attendance_Category` (nhóm chuyên cần), `Cluster` (nhóm K-Means).

> **Giới hạn biến:** dataset **không có** một số biến nêu trong đề cương như `Social Media`, `Gaming`, `Stress`, `Part-time Job`. Giới hạn này được ghi rõ ở mục 5.

---

## 3. Data Quality Issues phát hiện & Cách xử lý

| # | Vấn đề phát hiện | Feature ảnh hưởng | Số record | Mức độ | Cách xử lý | Lý do xử lý |
|---|---|---|---|---|---|---|
| 1 | Missing value | `Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home` | 78 (1.18%), 90 (1.36%), 67 (1.01%); tổng 235 ô | Trung bình | Impute bằng mode (giá trị xuất hiện nhiều nhất) | Biến phân loại nên dùng mode; tỷ lệ thiếu thấp (~1%) nên impute không làm lệch phân phối đáng kể; giữ record để không mất dữ liệu |
| 2 | Dòng trùng lặp (Duplicate) | Toàn bộ dòng | 0 | Không có | Đã kiểm tra, không cần xử lý | Không có bản ghi trùng hoàn toàn. Lưu ý: dữ liệu không có ID nên không phân biệt được "trùng thật" với hai sinh viên có giá trị giống nhau |
| 3 | `Exam_Score` > 100 (ngoài thang điểm) | `Exam_Score` | 1 (điểm = 101) | Cao | Loại bỏ record | Điểm vượt thang là lỗi nhập liệu, ảnh hưởng trực tiếp nhãn `Risk_Level`. Chỉ 1 dòng (0.02%) nên loại bỏ không ảnh hưởng phân phối. Phương án thay thế: giới hạn về 100 |
| 4 | `Attendance` > 100, `Hours_Studied` < 0, `Sleep_Hours` < 0 | 3 cột tương ứng | 0 | Không có | Đã kiểm tra, không cần xử lý | Không có giá trị bất khả thi |
| 5 | Dữ liệu dạng chữ | Các cột object | — | Bắt buộc xử lý | Encoding (xem mục 3.1) | Mô hình cần dữ liệu số |
| 6 | Outlier `Exam_Score` (IQR, Boxplot) | `Exam_Score` | 104 (ngưỡng hợp lệ [59, 75]) | Thấp | Giữ nguyên, không loại | Điểm cao/thấp là kết quả thật của sinh viên; loại bỏ sẽ làm mất chính nhóm nguy cơ cần dự đoán |

### 3.1 Mã hóa dữ liệu (Encoding)

| Nhóm | Cột | Cách mã hóa | Lý do |
|---|---|---|---|
| Ordinal Low/Medium/High | `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality` | Low=0, Medium=1, High=2 | Có thứ tự rõ ràng, giữ được quan hệ lớn/nhỏ |
| Ordinal riêng | `Distance_from_Home` | Near=0, Moderate=1, Far=2 | Có thứ tự theo khoảng cách |
| Nhị phân | `Extracurricular_Activities`, `Internet_Access`, `Learning_Disabilities` | Yes=1, No=0 | Chỉ có 2 giá trị |
| Nhị phân | `School_Type` | Public=0, Private=1 | Chỉ có 2 giá trị |
| Nhị phân | `Gender` | Male=0, Female=1 | Chỉ có 2 giá trị |
| Nominal | `Peer_Influence`, `Parental_Education_Level` | One-Hot Encoding (`drop_first=True`) | Xem ghi chú bên dưới |

**Ghi chú:**
- `Parental_Education_Level` (High School < College < Postgraduate) và `Peer_Influence` (Negative < Neutral < Positive) về bản chất có thể coi là thứ bậc. Nhóm chọn one-hot để mô hình không giả định khoảng cách đều giữa các mức. `[TODO: nếu đổi sang ordinal thì cập nhật bảng này và số cột ở mục 1]`
- `StandardScaler` được áp dụng riêng ở bước Modeling (không lưu vào clean dataset).
- Dữ liệu sau mã hóa (`clean_dataset.csv`) phục vụ mô hình. Nếu cần bản dễ đọc cho EDA/dashboard, dùng lại dữ liệu trước mã hóa hoặc ánh xạ ngược từ bảng trên.

### 3.2 Biến mục tiêu `Risk_Level`

- Tạo từ `Exam_Score` theo quartile: `< Q1 (65)` → **High**, `< Q3 (69)` → **Medium**, còn lại → **Low**.
- Khoảng điểm: High 55–64, Medium 65–68, Low 69–100.
- Phân bố lớp: High 1,452 (21.98%) / Medium 2,906 (43.99%) / Low 2,248 (34.03%).

**Lý giải và giới hạn của cách đặt nhãn:**
- Nhãn chia theo quartile nên mang tính **tương đối**: "High risk" nghĩa là nằm trong nhóm điểm thấp nhất so với toàn bộ mẫu, không phải rớt môn theo thang điểm tuyệt đối. Điểm trung bình toàn mẫu là 67.24 và điểm thấp nhất là 55.
- Cách này được chọn vì dataset không có ngưỡng đạu/rớt chính thức, và cho phân bố lớp đủ lớn để huấn luyện.
- Do có nhiều sinh viên trùng điểm tại ngưỡng, tỷ lệ lớp lệch nhẹ so với 25/50/25 lý thuyết (lớp High chiếm ~22%). Ở bước Modeling cần dùng `class_weight` hoặc đánh giá bằng F1/recall của lớp High, không chỉ accuracy.

### 3.3 Nhật ký làm sạch (Cleaning Log)

| Bước | Hành động | Số dòng/ô bị tác động | Notebook |
|---|---|---|---|
| 1 | Đọc dữ liệu raw, kiểm tra kích thước, kiểu dữ liệu | 6,607 dòng × 20 cột | `01_data_understanding.ipynb` |
| 2 | Kiểm tra duplicate | 0 | `03_data_cleaning.ipynb` |
| 3 | Kiểm tra giá trị bất khả thi (`Attendance`, `Hours_Studied`, `Sleep_Hours`) | 0 | `03_data_cleaning.ipynb` |
| 4 | Loại dòng `Exam_Score` = 101 | 1 dòng | `03_data_cleaning.ipynb` |
| 5 | Impute mode cho 3 cột thiếu | 235 ô | `03_data_cleaning.ipynb` |
| 6 | Mã hóa ordinal / nhị phân / one-hot | 22 cột sau mã hóa | `03_data_cleaning.ipynb` |
| 7 | Phát hiện outlier `Exam_Score` bằng IQR (giữ nguyên) | 104 dòng | `03_data_cleaning.ipynb` |

> `[TODO: đối chiếu số bước và tên notebook với thực tế trong repo; nếu có file cleaning_log.csv thì ghi đường dẫn ở mục 7]`

---

## 4. Số record trước / sau cleaning

| Bảng dữ liệu | Số dòng Trước | Số dòng Sau | Số dòng Bị loại | Tỷ lệ bảo toàn |
|---|---|---|---|---|
| `StudentPerformanceFactors.csv` | 6,607 | 6,606 | 1 | 99.98% |

> **Nhận xét:** Ưu tiên giữ lại record khi thông tin cốt lõi vẫn dùng được (missing được impute thay vì xóa). Chỉ loại 1 dòng có `Exam_Score` = 101 (ngoài thang điểm). Không có dòng trùng lặp. 104 outlier theo IQR được giữ lại vì là giá trị thật.

---

## 5. Giới hạn của dữ liệu và rủi ro cần lưu ý

1. **Thiếu biến so với đề cương:** không có `Social Media`, `Gaming`, `Stress`, `Part-time Job`, nên phân tích đa yếu tố chỉ giới hạn trong 19 biến đầu vào hiện có.
2. **Dữ liệu cắt ngang, không có ID:** không theo dõi được sinh viên theo thời gian. "Cảnh báo sớm" ở đây nghĩa là dự đoán nguy cơ dựa trên các yếu tố có sẵn trước kỳ thi (như `Previous_Scores`, `Attendance`, `Hours_Studied`), không phải theo dõi diễn biến.
3. **Nguồn dữ liệu:** `[TODO: ghi rõ dữ liệu có phải tổng hợp hay không]`. Kết quả mô hình chỉ nên diễn giải trong phạm vi dataset, không suy rộng sang sinh viên Việt Nam.
4. **Nhãn tương đối:** xem mục 3.2.
5. **Rò rỉ dữ liệu (data leakage):**
   - `Exam_Score` là nguồn tạo `Risk_Level` nên **không đưa vào feature** khi huấn luyện.
   - `Cluster` (K-Means) cần được tạo **không dùng `Exam_Score`**. `[TODO: xác nhận trong notebook]`
   - Impute mode nên fit trên tập train rồi áp dụng cho tập test khi huấn luyện mô hình. `[TODO: xác nhận bước hiện tại làm trước hay sau khi chia train/test]`
6. **Mất cân bằng lớp nhẹ:** xem mục 3.2.

---

## 6. Column được giữ lại / Khuyến nghị xử lý tiếp theo

- **Giữ lại:** toàn bộ 22 cột của `clean_dataset.csv` (sau đó thêm `Risk_Level`, `Study_Sleep_Ratio`, `Attendance_Category`, `Cluster`).
- **Khuyến nghị cho bước EDA / Feature Engineering** (xem `docs/eda_report.md` và `docs/feature_report.md`):
  - Tạo `Study_Sleep_Ratio`, `Attendance_Category`, `Extra_Engagement`.
  - Chọn feature theo tương quan với `Exam_Score` (|r| > 0.05) và Feature Importance.
  - Không đưa `Exam_Score` vào tập feature khi huấn luyện dự đoán `Risk_Level` (tránh data leakage).
  - Khi đánh giá mô hình, dùng F1-macro và recall lớp High, không chỉ accuracy.

---

## 7. Output bàn giao

**1. Dữ liệu**
- `clean_dataset.csv`, `dataset_with_risk.csv`, `eda_dataset_final.csv`, `final_results_with_risk_score.csv`
- Bảng MySQL `students` (đẩy từ `clean_dataset.csv` bằng SQLAlchemy/PyMySQL)

**2. Notebook** (thư mục `notebooks/`)
- `01_data_understanding.ipynb`
- `02_data_collection.ipynb`
- `03_data_cleaning.ipynb`
- `04_eda.ipynb`
- `DATA.ipynb` (bản gộp ban đầu ở thư mục gốc) `[TODO: chuyển vào notebooks/ hoặc xóa để tránh trùng]`

**3. Model & artifact**
- `best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `selected_features.pkl`, `feature_columns.pkl`, `all_models_results.pkl`

**4. Dashboard**
- `app.py` (Streamlit: tổng quan + What-if Analysis)

**5. Báo cáo** (thư mục `docs/`)
- `data_documentation.md` (tài liệu này)
- `data_quality_report.md` (phân tích chất lượng chi tiết)
- `eda_report.md` (khám phá dữ liệu)
- `feature_report.md` (đặc trưng tạo thêm và được chọn)
