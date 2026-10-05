# Notebooks Presentation Layer & Domain Architecture

Thư mục `notebooks/` đóng vai trò là tầng trình diễn (Presentation Layer) cho đồ án Dự báo Lượng mưa TP.HCM, được cấu trúc theo từng **Domain chuyên biệt** với hợp đồng dữ liệu chuẩn hóa, tính độc lập tuyệt đối giữa các notebook và tuân thủ các quy tắc khoa học bất di bất dịch (**Zero-Tolerance Scientific Rules**).

---

## 1. Cấu trúc Thư mục Domain-Driven Chuẩn hóa

$$\boxed{\mathbf{00\text{ Data Quality}}} \longrightarrow \boxed{\mathbf{01\text{ EDA}}} \longrightarrow \boxed{\mathbf{02\text{ Feature Engineering}}} \longrightarrow \boxed{\mathbf{03\text{ Modeling}}}$$

```text
notebooks/
├── data_quality/                   # 00 — Tuyến phòng thủ Data Integrity
│   ├── 00.0_data_quality_contract.ipynb
│   ├── 00.1_ingestion_schema_audit.ipynb
│   ├── 00.2_missing_data_diagnosis.ipynb
│   ├── 00.3_temporal_integrity_split.ipynb
│   └── 00.4_data_quality_synthesis.ipynb
│
├── eda/                            # 01 — Khám phá Dữ liệu & Sinh Bằng chứng Khoa học
│   ├── 01.0_eda_contract.ipynb
│   ├── 01.1_threshold_occurrence.ipynb
│   ├── 01.2_distribution_intermittency.ipynb
│   ├── 01.3_correlation_multicollinearity.ipynb
│   ├── 01.4_temporal_structure.ipynb
│   ├── 01.5_stationarity_autocorrelation.ipynb
│   ├── 01.6_extreme_events.ipynb
│   └── 01.7_eda_synthesis_report.ipynb
│
├── feature_engineering/            # 02 — Hiện thực hóa Quyết định Đầu vào Mô hình
│   └── 02_feature_pipeline.ipynb
│
└── modeling/                       # 03 — Huấn luyện, Thử nghiệm & Đánh giá Đa chiều
    ├── 03.0_modeling_contract.ipynb
    ├── 03.1_statistical.ipynb
    ├── 03.2_ml_direct.ipynb
    ├── 03.3_ml_hurdle.ipynb
    ├── 03.4_ml_tweedie.ipynb
    ├── 03.5_neural.ipynb
    ├── 03.6_unified_benchmark.ipynb
    └── 03.7_crmae_visualization_gatecheck.ipynb
```

---

## 2. Phân định Rõ ràng Ranh giới 3 Tầng Trách nhiệm

$$\boxed{\text{Data Quality} \neq \text{EDA} \neq \text{Feature Engineering}}$$

| Tầng (Layer) | Trách nhiệm Khoa học Cốt lõi | Thao tác Tuyệt đối Bị Cấm tại Tầng này |
|---|---|---|
| **00 Data Quality** | **Fix data integrity problems**: Kiểm tra schema, dữ liệu khuyết, trùng lặp, tính liên tục thời gian, giá trị bất hợp lý vật lý và thiết lập canonical split. | **Tuyệt đối KHÔNG tính tương quan với target, KHÔNG tính VIF, KHÔNG chạy regplot, KHÔNG thực hiện feature selection.** |
| **01 EDA** | **Understand data & produce evidence**: Khám phá phân phối, tính gián đoạn (Syntetos-Boylan), tương quan Spearman, sàng lọc VIF, chu kỳ FFT/MSTL, tính dừng ADF/KPSS, hiện tượng cực đoan. | **Tuyệt đối KHÔNG biến đổi hoặc loại bỏ cột vào pipeline mô hình; chỉ xuất khuyến nghị khoa học.** |
| **02 Feature Engineering** | **Implement modeling-input decisions**: Hiện thực hóa các quyết định đầu vào cho mô hình: trễ `shift(1)`, rolling window, cyclical features, fit-on-train scalers, phân tách Construction vs Selection vs Scaling. | **Tuyệt đối KHÔNG tính toán thống kê trên toàn bộ dataset hoặc rò rỉ tập Test vào `fit()`.** |
| **03 Modeling** | **Train, Evaluate & Benchmark**: Huấn luyện các họ mô hình, tuân thủ nghiêm ngặt Horizon Purge ($H=7$), ranh giới fit trên Dev Train, đánh giá trên Validation và báo cáo trên Test. | **Tuyệt đối KHÔNG dùng Test set để chọn Feature Level, HPO hay threshold.** |

---

## 3. Chi tiết Từng Domain & Hợp đồng Khoa học (Contracts)

### Domain `data_quality/` (00.0 $\to$ 00.4)
Tập trung 100% vào tính toàn vẹn dữ liệu (Data Integrity Gate):
1. `00.0_data_quality_contract.ipynb`: Công bố hợp đồng chất lượng dữ liệu, tiêu chuẩn WMO, quy tắc xử lý dữ liệu và thiết lập phạm vi nghiệm thu. Xuất `data_quality_contract.json`.
2. `00.1_ingestion_schema_audit.ipynb`: Kiểm tra sự tồn tại của file raw, số dòng/cột, ánh xạ tên cột Việt hóa, kiểu dữ liệu, phạm vi vật lý hợp lệ (Physical Range Checks). Xuất `ingestion_audit_report.json`.
3. `00.2_missing_data_diagnosis.ipynb`: **Chẩn đoán Missing Pattern**:
   - Phát hiện 5 biến bức xạ/UV bị khuyết 366 ngày liên tục trong toàn bộ năm 2000.
   - Xác định nguyên nhân gốc rễ: **Structural / System-driven Missingness** (Sensor / product availability trước 2001 trong hệ thống NASA POWER).
   - Đánh giá khả năng phục hồi: Việc nội suy/tái tạo 366 ngày không có cơ sở khoa học tin cậy và có nguy cơ tạo dữ liệu giả làm sai lệch cấu trúc vật lý.
   - **Quyết định Missing Handling**:
     $$\boxed{\text{Loại bỏ toàn bộ dữ liệu năm 2000 (366 ngày)}} \quad \text{và} \quad \boxed{\text{Giữ lại toàn bộ 5 features bị missing trong năm 2000}}$$
   - Chuyển giao toàn bộ 36 features sạch từ 2001 sang EDA để đánh giá tính đa cộng tuyến thực tế. Xuất `missing_diagnosis_report.json`.
4. `00.3_temporal_integrity_split.ipynb`: **Tái Kiểm toán Toàn diện (Full Re-audit)** trên tập dữ liệu sau khi loại năm 2000:
   - Xác nhận: **9.251 dòng $\times$ 36 cột**, **0 missing**, **0 duplicates**, **0 date gaps**.
   - Thiết lập **Canonical Time Series Split** bất biến:
     - Cutoff: `2020-05-01`
     - **Train**: `2001-01-01` $\to$ `2020-04-30` (**7.060 ngày**, 36 cột)
     - **Test**: `2020-05-01` $\to$ `2026-04-30` (**2.191 ngày**, 36 cột) — cách ly tuyệt đối.
   - Xuất `canonical_split_report.json`.
5. `00.4_data_quality_synthesis.ipynb`: Đóng gói dữ liệu sạch vào `data/processed/clean_weather_data.csv` (9.251 $\times$ 36) và xuất `data_quality_report.json`.

---

### Domain `eda/` (01.0 $\to$ 01.7)
Tổ chức theo nguyên tắc $\boxed{1\ \text{notebook} = 1\ \text{analytical domain}}$, chạy độc lập, 100% trên `train_df` (7.060 ngày $\times$ 36 cột):
1. `01.0_eda_contract.ipynb`: Thiết lập ranh giới Train/Test, chính sách Zero Data Leakage, định nghĩa target & ngưỡng WMO $\tau = 0.1\text{ mm}$, loại bỏ dependency cycle. Xuất `eda_contract.json`.
2. `01.1_threshold_occurrence.ipynb`: So sánh thực nghiệm các ngưỡng $\tau \in \{0.0, 0.1, 0.6, 1.0\text{ mm}\}$, tần suất sự kiện mưa, chốt ngưỡng vận hành $\tau = 0.1\text{ mm}$. Xuất `threshold_report.json`.
3. `01.2_distribution_intermittency.ipynb`: Khảo sát hình thái lệch phải, đuôi nặng, phân phối log1p, ma trận Syntetos-Boylan ($ADI = 1.65 > 1.32, CV^2 = 1.99 > 0.49 \to$ chuỗi Lumpy, biện minh cho mô hình Two-Stage Hurdle và Tweedie). Xuất `distribution_report.json`.
4. `01.3_correlation_multicollinearity.ipynb`:
   - Phân tích tương quan hạng Spearman và sàng lọc VIF ($VIF < 10$) trên 34 biến dự báo.
   - **Tái đánh giá chuyên biệt 5 biến bức xạ**: Cả 5 biến đều có $VIF \ge 10$ ($27.2$ đến $634.2$) và tương quan chéo rất cao ($r > 0.84 - 0.99$) với các biến bức xạ khác.
   - Khuyến nghị khoa học: Sàng lọc loại bỏ cho mô hình tuyến tính, giữ lại cho mô hình cây GBDT. Xuất `correlation_report.json`.
5. `01.4_temporal_structure.ipynb`: Cấu trúc thời gian đa tầng bằng bộ công cụ chính $\boxed{\text{FFT + MSTL}}$ (loại bỏ Wavelet để tránh dư thừa tính toán). Phân biệt rành mạch candidate periods $[7, 30, 122, 365]$, FFT detected periods và MSTL decomposition periods $[7, 30, 365]$. Xuất `temporal_structure_report.json`.
6. `01.5_stationarity_autocorrelation.ipynb`: Kiểm định nghiệm đơn vị ADF & KPSS (`regression='c'`), chẩn đoán phần dư MSTL, ACF/PACF. Cấu hình $(p, d, q)(P, D, Q)_s = (3, 0, 3)(1, 1, 1)_7$ được định danh rõ là $\boxed{\text{EDA-suggested configuration}}$. Xuất `stationarity_report.json`.
7. `01.6_extreme_events.ipynb`: Ngưỡng phân vị cực đoan $P_{95} = 17.19\text{ mm}$, $P_{99} = 34.16\text{ mm}$, $P_{99.9} = 61.59\text{ mm}$, chu kỳ lặp lại tính theo ngày lịch thực tế `date_series.diff().dt.days` (Rule 7). Xuất `extreme_events_report.json`.
8. `01.7_eda_synthesis_report.ipynb`: **Không chạy lại thuật toán**. Tổng hợp 6 domain artifacts thành `data/processed/eda_report.json` và thực hiện round-trip validation. Đảm bảo luồng phụ thuộc đơn hướng: $\text{DQ} \to \text{EDA} \to \text{FE} \to \text{Modeling}$.

---

### Domain `feature_engineering/` (02_feature_pipeline.ipynb)
- **File**: `02_feature_pipeline.ipynb`
- **Vai trò**: Triển khai `FeatureBuilder` làm nguồn chân lý duy nhất (Single Source of Truth):
  - Tách bạch 3 tầng trực giao: $\mathcal{T}_{\text{Scaling}} \circ \mathcal{S}_{\text{Selection}} \circ \mathcal{C}_{\text{Construction}}$.
  - Sinh 4 cấp độ cấu trúc đặc trưng ($F_0 \to F_3$):
    - $F_0$: 34 raw predictors
    - $F_1$: $F_0$ + 16 lags
    - $F_2$: $F_1$ + 12 rolling (`shift(1)` bắt buộc)
    - $F_3$: $F_2$ + 5 seasonal features (`Month_sin/cos`, `DayOfYear_sin/cos`, `Is_Wet_Season`)
  - Áp dụng `shift(1)` trước mọi rolling window trên biến target để ngăn ngừa lookahead leakage.
  - Xác thực tính nhất quán **Train–Serve Parity**: `build_single_step()` cho dự báo đệ quy khớp tuyệt đối với batch `transform()` ($\max |\Delta| < 10^{-14}$).
  - Khóa Fit Boundary: Chỉ fit imputation và transformation trên `Dev Train` ($6.329$ ngày).

---

### Domain `modeling/` (03.0 $\to$ 03.7)
Tập trung 8 notebook chuyên biệt với phạm vi rành mạch:
1. `03.0_modeling_contract.ipynb`: **Không huấn luyện**. Thiết lập protocol thực nghiệm chung, quy chuẩn Horizon Purge ($H=7$, loại bỏ 7 origins biên mỗi split), chính sách Future Covariates, Fit Boundary trên Dev Train và xuất `modeling_contract.json`.
2. `03.1_statistical.ipynb`: Huấn luyện & đánh giá nhóm Statistical & Intermittent (`Naive`, `SeasonalNaive`, `ARIMA`, `SARIMA_EDA`, `AutoARIMA`, `AutoETS`, `Croston`, `TSB`, `ADIDA`, `IMAPA`). Xuất `statistical_results.csv` & `.npz`.
3. `03.2_ml_direct.ipynb`: Huấn luyện ML Direct Multi-Horizon (`LinearRegression`, `RandomForest`, `XGBoost_Direct`, `LightGBM_Direct`) với mẫu huấn luyện tuân thủ nghiêm ngặt Horizon Purge $t + H \le T_{\text{split\_end}}$. Xuất `ml_direct_results.csv` & `.npz`.
4. `03.3_ml_hurdle.ipynb`: Huấn luyện ML Hurdle Two-Stage (`Logistic+Linear`, `RandomForest_Hurdle`, `LightGBM_Hurdle`, `XGBoost_Hurdle`). Ngưỡng $\tau_c$ được tinh chỉnh trên tập **Validation** và **đóng băng (freeze)** trước khi inference trên Test. Xuất `ml_hurdle_results.csv` & `.npz`.
5. `03.4_ml_tweedie.ipynb`: Huấn luyện mô hình Compound Poisson-Gamma đơn kỳ (`LightGBM_Tweedie`, `XGBoost_Tweedie`, $p=1.5$). Xuất `ml_tweedie_results.csv` & `.npz`.
6. `03.5_neural.ipynb`: Huấn luyện mô hình Deep Learning (`LSTM`, `TCN`, `NHITS`, `PatchTST`, `TFT`) với protocol đo lường VRAM/RAM và CUDA synchronization. Xuất `neural_results.csv` & `.npz`.
7. `03.6_unified_benchmark.ipynb`: **Tích hợp Benchmark Đa chiều**:
   - Section A: Model Group Benchmark (Bộ đặc trưng chuẩn $F_3$).
   - Section B: **Feature Complexity Benchmark** ($F_0 \to F_3$ ablation không filter vs optimized pipeline).
   - Section C: Model Capacity Benchmark (Small vs Base vs Large).
   - Section D: Pareto Frontier 2D Projections (Error vs Features, Error vs Time, Error vs Memory).
8. `03.7_crmae_visualization_gatecheck.ipynb`: Trực quan hóa tiến trình `CR-MAE@H`, `CR-Bias@7`, so sánh thực tế vs dự báo và chạy **Automated Gate Check** bằng code assertion kiểm tra toàn bộ tiêu chuẩn khoa học.

---

## 4. Các Nguyên tắc P0 Bắt buộc (Zero-Tolerance Invariants)
- **P0.1 — Horizon Purge ($H=7$)**: Loại bỏ đúng $7$ origins cuối cùng của mỗi split (`04-24` $\to$ `04-30`) để ngăn nhãn tương lai $\mathbf{y}_{t+1:t+7}$ vươn sang ranh giới kế tiếp.
  - Dev Train: $6.322$ valid origins ($6.329 - 7$)
  - Validation: $724$ valid origins ($731 - 7$)
  - Final Test: $2.184$ valid origins ($2.191 - 7$)
- **P0.2 — Zero Test Contamination**: Mọi bước học (Scaler, Imputer, VIF, Feature Importance, HPO, Threshold tuning) CHỈ ĐƯỢC THỰC HIỆN trên Dev Train / Validation. Final Test bị cách ly hoàn toàn cho đến bước báo cáo cuối cùng.
- **P0.3 — Hợp đồng Khả dụng Tương lai (Future Covariates)**: Nghiêm cấm sử dụng biến khí tượng tương lai thực tế trong dự báo $t+1 \dots t+7$. Chỉ cho phép dùng biến lịch thiên văn biết trước (`DayOfYear_sin/cos`, `Month_sin/cos`, `Is_Wet_Season`) và các trễ/dự báo đệ quy.
- **P0.4 — Tách biệt Ablation (Benchmark A) vs Pipeline Tối ưu (Benchmark B)**: Phân định rạch ròi giữa việc đo lường giá trị gia tăng của cấu trúc đặc trưng và việc tối ưu hóa pipeline của từng họ mô hình.
- **Metric Contract**: Triển khai thống nhất `CR-MAE@H` và `CR-Bias@H` qua module chuẩn `src.evaluation.metrics` (hỗ trợ backward compatibility qua `src.training.metrics`).
- **Automated Gate Checks**: Toàn bộ kiểm định nghiệm thu được thực thi bằng logic code có điều kiện (`assert all_pass == True`), không hard-code trạng thái.
