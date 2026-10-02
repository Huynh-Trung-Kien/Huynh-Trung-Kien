# Báo Cáo Phân Tích Chất Lượng Dữ Liệu (Data Quality Report)

**Đề tài:** Ứng dụng Khoa học dữ liệu đa yếu tố trong phân tích, dự đoán và cảnh báo sớm nguy cơ học tập kém của sinh viên
**Vai trò:** Data Engineer / Data Analyst
**Tài liệu liên quan:** `docs/data_documentation.md` (mô tả dữ liệu và bàn giao), `notebooks/03_data_cleaning.ipynb` (mã xử lý)

---

## 1. Tổng quan Đánh giá Chất lượng Dữ liệu

Dataset Student Performance Factors gồm **1 bảng** (`StudentPerformanceFactors.csv`) với **6,607 dòng, 20 cột**. Sau khi kiểm tra và làm sạch, dữ liệu còn **6,606 dòng** và đạt mức **Tốt, sẵn sàng cho phân tích và mô hình hóa**, với các giới hạn nêu ở mục 5.

### Bảng chấm điểm chất lượng theo từng tiêu chí

| Tiêu chí | Kết quả kiểm tra | Đánh giá |
|---|---|---|
| **Tính đầy đủ (Completeness)** | 17/20 cột không thiếu. 3 cột phân loại thiếu tổng 235 ô (~0.18% trong 132,140 ô) | Tốt |
| **Tính duy nhất (Uniqueness)** | 0 dòng trùng lặp hoàn toàn (không có ID nên không kiểm tra được trùng theo khóa) | Tốt, có lưu ý |
| **Tính hợp lệ (Validity)** | Các biến số nằm trong miền hợp lý; chỉ 1 dòng `Exam_Score` = 101 vượt thang | Tốt |
| **Tính nhất quán (Consistency)** | Cột phân loại không có khoảng trắng thừa hay lệch hoa/thường; mỗi cột có 2–3 giá trị chuẩn | Tốt |
| **Tính đại diện (Representativeness)** | Chưa xác định được nguồn thu thập `[TODO: ghi rõ dữ liệu thật hay tổng hợp]` | Cần lưu ý (mục 5) |

### Các điểm sáng của Dataset:
1. **Tính duy nhất:** 0 dòng trùng lặp trong 6,607 dòng.
2. **Tính hợp lệ:** `Attendance` 60–100%, `Sleep_Hours` 4–10, `Hours_Studied` 1–44, `Previous_Scores` 50–100. Không có giá trị âm. Chỉ có 1 dòng `Exam_Score` = 101 (vượt thang 100).
3. **Tính đầy đủ:** chỉ 3 cột phân loại có missing, tổng 235 ô.
4. **Tính nhất quán:** các cột phân loại không có khoảng trắng thừa hay khác biệt hoa/thường.

---

## 2. Chi tiết Các Vấn đề Chất lượng Dữ liệu & Phương án Xử lý

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
- **Mức độ ảnh hưởng:** **Trung bình.** Tỷ lệ thiếu thấp (~1% mỗi cột) nhưng các mô hình như Logistic Regression, SVM không chấp nhận giá trị trống.
- **Cách xử lý:** Điền (impute) bằng **mode** của từng cột, không xóa record.
- **Tại sao chọn cách xử lý đó?**
  - Đây là biến phân loại nên mode là lựa chọn tự nhiên, không tạo ra giá trị "lạ" như mean.
  - Xóa 229 dòng sẽ mất 3.47% dữ liệu, trong khi các cột còn lại của những dòng này vẫn đầy đủ và có giá trị.
  - Tỷ lệ thiếu thấp nên việc điền mode ít làm lệch phân phối.
- **Hạn chế và kiểm tra bổ sung:**
  - Mode làm tăng nhẹ tần suất của giá trị phổ biến. Nếu cần chặt chẽ hơn, có thể so sánh kết quả mô hình với cách tạo nhóm "Unknown" như một phép kiểm tra độ nhạy.
  - Khi huấn luyện mô hình, mode nên được tính trên tập train rồi áp dụng cho tập test để tránh rò rỉ dữ liệu. `[TODO: xác nhận bước hiện tại điền mode trước hay sau khi chia train/test; sự khác biệt rất nhỏ ở tỷ lệ thiếu ~1% nhưng cần ghi đúng]`

---

### Vấn đề 2: Điểm thi vượt thang điểm (`Exam_Score` > 100)
- **Vấn đề:** Có bản ghi `Exam_Score = 101`, vượt thang điểm tối đa 100.
- **Feature bị ảnh hưởng:** `Exam_Score` (cột dùng để tạo nhãn `Risk_Level`).
- **Số record bị ảnh hưởng:** **1 bản ghi** (0.015%). Bản ghi này có `Attendance` = 98, `Previous_Scores` = 93, `Hours_Studied` = 27.
- **Mức độ ảnh hưởng:** **Cao về nguyên tắc** (giá trị bất khả thi), **rất thấp về số lượng**.
- **Cách xử lý:** Loại bỏ bản ghi này (6,607 → 6,606 dòng).
- **Tại sao chọn cách xử lý đó?** Điểm > 100 là lỗi nhập liệu và không thể biết điểm đúng là bao nhiêu. Vì chỉ 1 dòng nên loại bỏ không ảnh hưởng đến mẫu, và tránh phải đoán giá trị cho biến mục tiêu.
- **Phương án thay thế đã cân nhắc:** giới hạn điểm về 100 (clip). Bản ghi này thuộc nhóm sinh viên có chuyên cần và điểm cũ cao, nên điểm thật nhiều khả năng gần 100, và nhãn của dòng này sẽ là Low dù chọn phương án nào. Kết quả mô hình gần như không đổi giữa hai cách.

---

### Vấn đề 3: Dòng trùng lặp và hạn chế do thiếu mã định danh
- **Vấn đề:** Cần kiểm tra bản ghi trùng lặp, nhưng dataset không có cột ID sinh viên.
- **Feature bị ảnh hưởng:** Toàn bộ dòng.
- **Số record bị ảnh hưởng:** **0 dòng** trùng lặp hoàn toàn trong 6,607 dòng.
- **Mức độ ảnh hưởng:** **Không có** (về số liệu).
- **Cách xử lý:** Không cần xử lý.
- **Lưu ý:** Do không có ID, kiểm tra chỉ dựa trên việc so khớp toàn bộ 20 cột. Hai sinh viên khác nhau có giá trị giống hệt nhau về mọi cột sẽ bị coi là trùng, và ngược lại không thể phát hiện trường hợp một sinh viên xuất hiện hai lần với giá trị hơi khác nhau. Trong dữ liệu này không xảy ra trường hợp nào bị phát hiện.

---

### Vấn đề 4: Ngoại lai theo IQR (Outliers)
- **Vấn đề:** Một số cột có giá trị nằm ngoài khoảng [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
- **Feature bị ảnh hưởng và số record (tính trên dữ liệu gốc):**

| Feature | Khoảng hợp lệ theo IQR | Số ngoại lai | Độ lệch (skew) |
|---|---|---|---|
| `Exam_Score` | 59 – 75 | 104 | 1.64 |
| `Tutoring_Sessions` | -0.5 – 3.5 | 430 | 0.82 |
| `Hours_Studied` | 4 – 36 | 43 | 0.01 |
| `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Physical_Activity` | — | 0 | ≈ 0 |

- **Mức độ ảnh hưởng:** **Thấp.** Đây đều là giá trị nằm trong miền hợp lệ về nghiệp vụ.
- **Cách xử lý:** **Giữ nguyên 100%**, chỉ ghi nhận (dùng Boxplot để quan sát).
- **Tại sao chọn cách xử lý đó?**
  - `Exam_Score` lệch phải (skew 1.64): các điểm cao (> 75) là sinh viên học tốt thật sự, không phải lỗi.
  - `Tutoring_Sessions` là biến đếm nhỏ (0–8), 430 dòng "ngoại lai" thực chất là những sinh viên học thêm nhiều.
  - Nếu loại 104 dòng ngoại lai của `Exam_Score` sẽ làm mất các điểm cực trị và thu hẹp nhân tạo phân phối điểm, ảnh hưởng đến việc xác định ngưỡng `Risk_Level`. Đặc biệt các điểm thấp (< 59) chính là nhóm nguy cơ cao cần dự đoán.
- **Lưu ý về số liệu:** 104 ngoại lai của `Exam_Score` được tính trên dữ liệu gốc và **bao gồm cả dòng điểm 101 đã loại ở Vấn đề 2**. Trong dữ liệu sạch còn khoảng **103** dòng ngoại lai. `[TODO: chạy lại IQR trên clean_dataset.csv để xác nhận và đồng bộ con số này với data_documentation.md]`

---

### Vấn đề 5: Dữ liệu dạng chữ cần mã hóa (Categorical Encoding)
- **Vấn đề:** 13 cột dạng chữ không thể đưa trực tiếp vào mô hình.
- **Feature bị ảnh hưởng:** Toàn bộ 13 cột phân loại (xem bảng thuộc tính trong `data_documentation.md`).
- **Số record bị ảnh hưởng:** Toàn bộ 6,606 dòng.
- **Mức độ ảnh hưởng:** **Bắt buộc xử lý** (không xử lý thì không huấn luyện được).
- **Cách xử lý:**
  - **Ordinal Encoding** cho biến có thứ bậc: Low/Medium/High → 0/1/2; `Distance_from_Home` Near/Moderate/Far → 0/1/2.
  - **Label nhị phân** cho biến 2 giá trị (Yes/No, Public/Private, Male/Female).
  - **One-Hot Encoding (`drop_first=True`)** cho `Peer_Influence`, `Parental_Education_Level`.
- **Tại sao chọn cách xử lý đó?**
  - Ordinal Encoding giữ được thông tin thứ bậc và không làm tăng số cột.
  - One-Hot cho hai biến còn lại để mô hình không giả định khoảng cách đều giữa các mức. `drop_first=True` tránh đa cộng tuyến. Kết quả: 20 cột gốc → 22 cột.
- **Lưu ý:** `Parental_Education_Level` (High School < College < Postgraduate) và `Peer_Influence` (Negative < Neutral < Positive) về bản chất có thể coi là thứ bậc. Đây là lựa chọn của nhóm, không phải bắt buộc. `[TODO: nếu đổi sang ordinal thì cập nhật số cột ở đây và ở data_documentation.md]`

---

### Vấn đề 6: Phân bố nhãn `Risk_Level` (Class Distribution)
- **Vấn đề:** Nhãn mục tiêu được tạo từ `Exam_Score` theo quartile (`< 65` High, `< 69` Medium, còn lại Low), phân bố các lớp không đều hoàn toàn.
- **Feature bị ảnh hưởng:** `Risk_Level`.
- **Số record bị ảnh hưởng:**

| Lớp | Số dòng | Tỷ lệ | Khoảng `Exam_Score` |
|---|---|---|---|
| High | 1,452 | 21.98% | 55 – 64 |
| Medium | 2,906 | 43.99% | 65 – 68 |
| Low | 2,248 | 34.03% | 69 – 100 |

- **Mức độ ảnh hưởng:** **Thấp đến trung bình.** Lớp nhỏ nhất (High) vẫn chiếm ~22%, chưa phải mất cân bằng nghiêm trọng, nhưng đây lại là lớp quan trọng nhất cần phát hiện.
- **Cách xử lý:** Giữ nguyên ở bước làm sạch. Ở bước Modeling dùng chia Train/Test có phân tầng (`stratify`) và theo dõi **Recall của lớp High**; áp dụng `class_weight` hoặc SMOTE nếu cần (SMOTE chỉ áp dụng trên tập train, không áp dụng trên tập test).
- **Tại sao chọn cách xử lý đó?** Bước cleaning không nên tự thay đổi phân bố nhãn. Quyết định cân bằng lớp thuộc về bước Modeling.
- **Lưu ý về bản chất nhãn:** Nhãn chia theo quartile nên mang tính **tương đối**. "High risk" nghĩa là thuộc nhóm điểm thấp nhất trong mẫu (55–64, trong khi điểm trung bình là 67.24), không phải rớt môn theo thang điểm tuyệt đối. Cần nêu rõ điều này khi diễn giải kết quả.
- **Lưu ý quan trọng:** Vì nhãn được tính trực tiếp từ `Exam_Score`, **không được đưa `Exam_Score` vào tập features** khi dự đoán `Risk_Level` (data leakage).

---

## 3. Phân biệt: Data Error vs Genuine Extreme Value

Nguyên tắc: **không tự động xóa dữ liệu chỉ vì nó là ngoại lai.**

| Hiện tượng | Bản chất | Quyết định xử lý | Cơ sở |
|---|---|---|---|
| `Exam_Score = 101` | **Data Error** | **LOẠI BỎ** (1 dòng) | Vượt thang điểm 100, không thể xác định giá trị đúng. |
| Dòng `Exam_Score` ngoài [59, 75] (104 dòng trên dữ liệu gốc, ~103 sau khi loại dòng 101) | **Genuine Extreme Value** | **GIỮ NGUYÊN** | Điểm cao/thấp thật của sinh viên; là chính đối tượng cần phân tích và dự đoán. |
| 430 dòng `Tutoring_Sessions` > 3 | **Genuine Extreme Value** | **GIỮ NGUYÊN** | Biến đếm nhỏ (0–8), giá trị cao là hợp lệ. |
| 43 dòng `Hours_Studied` ngoài [4, 36] | **Genuine Extreme Value** | **GIỮ NGUYÊN** | Nằm trong miền 1–44 giờ, phân bố đối xứng (skew ≈ 0). |
| Ô trống ở 3 cột phân loại | **Missing Data** | **IMPUTE bằng mode** | Tỷ lệ thấp (~1%), giữ được dòng. |

---

## 4. Kiểm tra Tính nhất quán & Hợp lệ (Consistency & Validity)

Dataset chỉ có **một bảng, không có khóa chính hay khóa ngoại**, nên không có bước kiểm tra toàn vẹn quan hệ như dự án nhiều bảng. Thay vào đó đã kiểm tra:

1. **Kiểu dữ liệu:** Sau làm sạch, `dataset_with_risk.csv` có 0 ô thiếu, 0 dòng trùng; gồm 19 cột số nguyên, 4 cột boolean (dummy), 1 cột thực (`Study_Sleep_Ratio`), 2 cột chữ (`Risk_Level`, `Attendance_Category`).
2. **Miền giá trị hợp lệ:** `Attendance` ≤ 100, `Hours_Studied` ≥ 0, `Sleep_Hours` ≥ 0 (không vi phạm); `Exam_Score` ≤ 100 (sau khi loại 1 dòng).
3. **Tính nhất quán nhãn:** Khoảng điểm của 3 lớp `Risk_Level` không chồng lấn (55–64, 65–68, 69–100).
4. **Tính nhất quán giá trị chữ:** Không có khoảng trắng thừa, không lệch hoa/thường; mỗi cột phân loại chỉ có 2–3 giá trị chuẩn.
5. **Đặc điểm cần lưu ý khi diễn giải:** Các biến số có phân bố khá đối xứng (độ lệch ≈ 0). Tương quan với `Exam_Score` chỉ nổi bật ở `Attendance` (r = 0.58) và `Hours_Studied` (r = 0.45); `Sleep_Hours` gần như không tương quan (r = -0.02). Tương quan không chứng minh nhân quả, nên thận trọng khi kết luận.

---

## 5. Giới hạn của Dữ liệu và Rủi ro cần lưu ý

1. **Nguồn dữ liệu:** `[TODO: kiểm tra mô tả trên Kaggle xem dataset là dữ liệu thật hay tổng hợp]`. Kết quả chỉ nên diễn giải trong phạm vi dataset, không suy rộng ra sinh viên nói chung.
2. **Thiếu biến so với đề cương:** không có `Social Media`, `Gaming`, `Stress`, `Part-time Job`.
3. **Dữ liệu cắt ngang, không có ID:** không theo dõi được sinh viên theo thời gian. "Cảnh báo sớm" nghĩa là dự đoán nguy cơ từ các yếu tố có sẵn trước kỳ thi.
4. **Nhãn tương đối:** xem Vấn đề 6.
5. **Rò rỉ dữ liệu:** không dùng `Exam_Score` làm feature; `Cluster` cần được tạo mà không dùng `Exam_Score`; fit imputer/scaler trên tập train. `[TODO: xác nhận trong notebook]`

---

## 6. Kết luận & Bàn giao

Dữ liệu gốc 6,607 dòng được làm sạch còn 6,606 dòng, **bảo toàn 99.98%** số bản ghi (chỉ loại 1 dòng điểm thi vượt thang). Toàn bộ 235 ô thiếu được điền bằng mode, không có dòng trùng lặp, các ngoại lai thật được giữ nguyên, và dữ liệu đã được mã hóa sẵn sàng cho bước EDA và Modeling. Nhãn `Risk_Level` phân bố High 21.98% / Medium 43.99% / Low 34.03%.

**Bàn giao cho nhóm EDA & Modeling:** `clean_dataset.csv`, `dataset_with_risk.csv`, `docs/data_documentation.md` và báo cáo này.

**Khuyến nghị:**
- Loại `Exam_Score` khỏi tập features khi huấn luyện.
- Chia Train/Test có phân tầng; fit imputer và scaler chỉ trên tập train.
- Ưu tiên Recall và F1 của lớp High trong đánh giá mô hình, không chỉ accuracy.
- Ghi rõ giới hạn ở mục 5 trong chương kết luận của báo cáo đồ án.
