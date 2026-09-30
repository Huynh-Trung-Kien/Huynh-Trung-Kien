# Báo Cáo Phân Tích Chất Lượng Dữ Liệu (Data Quality Report)
**Project:** Ứng dụng Khoa học dữ liệu đa yếu tố trong phân tích, dự đoán và cảnh báo sớm nguy cơ học tập kém của sinh viên
**Branch:** `feature/data` *(đổi theo nhánh của bạn)*
**Vai trò:** Data Engineer / Data Analyst

---

## 1. Tổng quan Đánh giá Chất lượng Dữ liệu

Dataset Student Performance Factors gồm **1 bảng** (`StudentPerformanceFactors.csv`) với **6,607 dòng, 20 cột**. Sau khi kiểm tra và làm sạch, dữ liệu còn **6,606 dòng** và đạt mức **Tốt - Sẵn sàng cho phân tích và mô hình hóa**.

### Các điểm sáng của Dataset:
1. **Tính duy nhất (Uniqueness):** 0 dòng trùng lặp trong 6,607 dòng.
2. **Tính hợp lệ (Validity):** Các biến số nằm trong miền hợp lý: `Attendance` 60–100%, `Sleep_Hours` 4–10, `Hours_Studied` 1–44, `Previous_Scores` 50–100. Không có giá trị âm. Chỉ có 1 dòng `Exam_Score` = 101 (vượt thang 100).
3. **Tính đầy đủ (Completeness):** 17/20 cột không thiếu dữ liệu. Chỉ 3 cột phân loại có missing, tổng 235 ô (~0.18% toàn bộ 132,140 ô).
4. **Tính nhất quán (Consistency):** Các cột phân loại không có khoảng trắng thừa hay khác biệt hoa/thường; mỗi cột chỉ có 2–3 giá trị chuẩn.

---

## 2. Chi tiết Các Vấn đề Chất Lượng Dữ Liệu Phát Hiện & Phương Án Xử Lý

Mỗi vấn đề được trình bày theo 6 tiêu chí: vấn đề là gì, bao nhiêu record, feature nào, mức độ ảnh hưởng, cách xử lý, lý do.

---

### Vấn đề 1: Giá trị khuyết thiếu (Missing Values)
- **Vấn đề:** Ba cột phân loại có ô trống.
- **Feature bị ảnh hưởng:**

| Feature | Số ô thiếu | Tỷ lệ | Giá trị phổ biến nhất (mode) |
|---|---|---|---|
| `Parental_Education_Level` | 90 | 1.36% | High School (49.5%) |
| `Teacher_Quality` | 78 | 1.18% | Medium (60.1%) |
| `Distance_from_Home` | 67 | 1.01% | Near (59.4%) |

- **Số record bị ảnh hưởng:** 229 dòng có ít nhất một ô thiếu (3.47%); trong đó chỉ 6 dòng thiếu từ 2 cột trở lên.
- **Mức độ ảnh hưởng:** **Trung bình.** Tỷ lệ thiếu thấp (~1% mỗi cột) nhưng mô hình như Logistic Regression, SVM không chấp nhận giá trị trống.
- **Cách xử lý:** Điền (Impute) bằng **mode** của từng cột, không xóa record.
- **Tại sao chọn cách xử lý đó?**
  - Đây là biến phân loại nên mode là lựa chọn tự nhiên, không tạo ra giá trị "lạ" như mean.
  - Nếu xóa 229 dòng sẽ mất 3.47% dữ liệu, trong khi các cột còn lại của những dòng này vẫn đầy đủ và có giá trị.
  - Tỷ lệ thiếu thấp nên việc điền mode ít làm lệch phân phối. *Hạn chế:* mode làm tăng nhẹ tần suất của giá trị phổ biến; nếu cần chặt chẽ hơn có thể so sánh với cách tạo nhóm "Unknown".

---

### Vấn đề 2: Điểm thi vượt thang điểm (Exam_Score > 100)
- **Vấn đề:** Có bản ghi `Exam_Score = 101`, vượt thang điểm tối đa 100.
- **Feature bị ảnh hưởng:** `Exam_Score` (cột dùng để tạo nhãn `Risk_Level`).
- **Số record bị ảnh hưởng:** **1 bản ghi** (0.015%). Bản ghi này có `Attendance` = 98, `Previous_Scores` = 93, `Hours_Studied` = 27.
- **Mức độ ảnh hưởng:** **Cao về nguyên tắc** (giá trị bất khả thi), **rất thấp về số lượng**.
- **Cách xử lý:** Loại bỏ bản ghi này (6,607 → 6,606 dòng).
- **Tại sao chọn cách xử lý đó?** Điểm > 100 là lỗi nhập liệu, nhưng không thể biết điểm đúng là bao nhiêu (có thể là 100 hoặc thấp hơn). Vì chỉ 1 dòng, việc loại bỏ không ảnh hưởng đến mẫu, và tránh phải đoán giá trị cho biến mục tiêu.

---

### Vấn đề 3: Ngoại lai theo IQR (Outliers)
- **Vấn đề:** Một số cột có giá trị nằm ngoài khoảng [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
- **Feature bị ảnh hưởng và số record (trên dữ liệu gốc):**

| Feature | Khoảng hợp lệ theo IQR | Số ngoại lai | Độ lệch (skew) |
|---|---|---|---|
| `Exam_Score` | 59 – 75 | 104 | 1.64 |
| `Tutoring_Sessions` | -0.5 – 3.5 | 430 | 0.82 |
| `Hours_Studied` | 4 – 36 | 43 | 0.01 |
| `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Physical_Activity` | — | 0 | ≈ 0 |

- **Mức độ ảnh hưởng:** **Thấp.** Đây đều là giá trị nằm trong miền hợp lệ về nghiệp vụ.
- **Cách xử lý:** **Giữ nguyên 100%**, chỉ ghi nhận (dùng Boxplot để quan sát).
- **Tại sao chọn cách xử lý đó?**
  - `Exam_Score` lệch phải (skew 1.64): các điểm cao (>75) là sinh viên học tốt thật sự, không phải lỗi.
  - `Tutoring_Sessions` là biến đếm nhỏ (0–8), 430 dòng "ngoại lai" thực chất là những sinh viên học thêm nhiều.
  - Nếu loại 104 dòng ngoại lai của `Exam_Score` sẽ còn 6,503 dòng, nhưng làm mất các điểm cực trị và thu hẹp nhân tạo phân phối điểm, ảnh hưởng đến việc xác định ngưỡng `Risk_Level`.

---

### Vấn đề 4: Dữ liệu dạng chữ cần mã hóa (Categorical Encoding)
- **Vấn đề:** 13 cột dạng chữ không thể đưa trực tiếp vào mô hình.
- **Feature bị ảnh hưởng:** Toàn bộ 13 cột phân loại (xem bảng thuộc tính trong `data_documentation.md`).
- **Số record bị ảnh hưởng:** Toàn bộ 6,606 dòng.
- **Mức độ ảnh hưởng:** **Bắt buộc xử lý** (không xử lý thì không huấn luyện được).
- **Cách xử lý:**
  - **Ordinal Encoding** cho biến có thứ bậc: Low/Medium/High → 0/1/2; `Distance_from_Home` Near/Moderate/Far → 0/1/2.
  - **Label nhị phân** cho biến 2 giá trị (Yes/No, Public/Private, Male/Female).
  - **One-Hot Encoding (`drop_first=True`)** cho `Peer_Influence`, `Parental_Education_Level` (biến danh nghĩa, không có thứ tự).
- **Tại sao chọn cách xử lý đó?** Ordinal Encoding giữ được thông tin thứ bậc và không làm tăng số cột. One-Hot cho biến danh nghĩa tránh mô hình hiểu nhầm có thứ tự. `drop_first=True` tránh đa cộng tuyến. Kết quả: 20 cột gốc → 22 cột.

---

### Vấn đề 5: Phân bố nhãn `Risk_Level` (Class Distribution)
- **Vấn đề:** Nhãn mục tiêu được tạo từ `Exam_Score` theo quartile (`< 65` High, `< 69` Medium, còn lại Low), phân bố các lớp không đều hoàn toàn.
- **Feature bị ảnh hưởng:** `Risk_Level`.
- **Số record bị ảnh hưởng:**

| Lớp | Số dòng | Tỷ lệ | Khoảng `Exam_Score` |
|---|---|---|---|
| High | 1,452 | 21.98% | 55 – 64 |
| Medium | 2,906 | 43.99% | 65 – 68 |
| Low | 2,248 | 34.03% | 69 – 100 |

- **Mức độ ảnh hưởng:** **Thấp đến trung bình.** Lớp nhỏ nhất (High) vẫn chiếm ~22%, chưa phải mất cân bằng nghiêm trọng, nhưng đây lại là lớp quan trọng nhất cần phát hiện.
- **Cách xử lý:** Giữ nguyên ở bước làm sạch. Ở bước Modeling dùng chia Train/Test có phân tầng (`stratify`) và theo dõi **Recall của lớp High**; áp dụng `class_weight` hoặc SMOTE nếu cần.
- **Tại sao chọn cách xử lý đó?** Bước cleaning không nên tự thay đổi phân bố nhãn. Quyết định cân bằng lớp thuộc về nhóm Modeling.
- **Lưu ý quan trọng:** Vì nhãn được tính trực tiếp từ `Exam_Score`, **không được đưa `Exam_Score` vào tập features** khi dự đoán `Risk_Level` (data leakage).

---

## 3. Phân Biệt: Data Error vs Genuine Extreme Value

Nguyên tắc: **không tự động xóa dữ liệu chỉ vì nó là ngoại lai.**

| Hiện tượng | Bản chất | Quyết định xử lý | Cơ sở |
|---|---|---|---|
| `Exam_Score = 101` | **Data Error** | **LOẠI BỎ** (1 dòng) | Vượt thang điểm 100, không thể xác định giá trị đúng. |
| 104 dòng `Exam_Score` ngoài [59, 75] | **Genuine Extreme Value** | **GIỮ NGUYÊN** | Điểm cao/thấp thật của sinh viên; là chính đối tượng cần phân tích và dự đoán. |
| 430 dòng `Tutoring_Sessions` > 3 | **Genuine Extreme Value** | **GIỮ NGUYÊN** | Biến đếm nhỏ (0–8), giá trị cao là hợp lệ. |
| 43 dòng `Hours_Studied` ngoài [4, 36] | **Genuine Extreme Value** | **GIỮ NGUYÊN** | Nằm trong miền 1–44 giờ, phân bố đối xứng (skew ≈ 0). |
| Ô trống ở 3 cột phân loại | **Missing Data** | **IMPUTE bằng mode** | Tỷ lệ thấp (~1%), giữ được dòng. |

---

## 4. Kiểm Tra Tính Nhất Quán & Hợp Lệ (Consistency & Validity)

Dataset chỉ có **một bảng, không có khóa chính hay khóa ngoại**, nên không có bước kiểm tra toàn vẹn quan hệ như dự án nhiều bảng. Thay vào đó đã kiểm tra:

1. **Kiểu dữ liệu:** Sau làm sạch, `dataset_with_risk.csv` có 0 ô thiếu, 0 dòng trùng; gồm 19 cột số nguyên, 4 cột boolean (dummy), 1 cột thực (`Study_Sleep_Ratio`), 2 cột chữ (`Risk_Level`, `Attendance_Category`).
2. **Miền giá trị hợp lệ:** `Attendance` ≤ 100, `Hours_Studied` ≥ 0, `Sleep_Hours` ≥ 0 (không vi phạm); `Exam_Score` ≤ 100 (sau khi loại 1 dòng).
3. **Tính nhất quán nhãn:** Khoảng điểm của 3 lớp `Risk_Level` không chồng lấn (55–64, 65–68, 69–100).
4. **Đặc điểm cần lưu ý khi diễn giải:** Các biến số có phân bố rất đối xứng (độ lệch ≈ 0) và tương quan với `Exam_Score` chỉ nổi bật ở `Attendance` (r = 0.58) và `Hours_Studied` (r = 0.45); `Sleep_Hours` gần như không tương quan (r = -0.02). Nên thận trọng khi kết luận về nhân quả.

---

## 5. Kết Luận & Bàn Giao
Dữ liệu gốc 6,607 dòng được làm sạch còn 6,606 dòng, **bảo toàn 99.98%** số bản ghi (chỉ loại 1 dòng điểm thi vượt thang). Toàn bộ 235 ô thiếu được điền bằng mode, không có dòng trùng lặp, các ngoại lai thật được giữ nguyên, và dữ liệu đã được mã hóa sẵn sàng cho bước EDA và Modeling. Nhãn `Risk_Level` phân bố High 21.98% / Medium 43.99% / Low 34.03%.

**Bàn giao cho nhóm EDA & Modeling:** `clean_dataset.csv`, `dataset_with_risk.csv` và tài liệu `data_documentation.md`.

**Khuyến nghị:** loại `Exam_Score` khỏi tập features khi huấn luyện; dùng phân tầng khi chia Train/Test; ưu tiên Recall của lớp High trong đánh giá mô hình.
