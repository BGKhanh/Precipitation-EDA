# Notebooks Presentation Layer

Thư mục `notebooks/` đóng vai trò là tầng trình diễn (Presentation Layer) cho đồ án Dự báo Lượng mưa HCMC.

## 1. Quy tắc Nghiêm ngặt (Rule #5 — `.agents/rules/project-structure.md`)
- **Không viết logic inline**: Mọi logic nạp dữ liệu, làm sạch, kỹ thuật đặc trưng, huấn luyện mô hình và dự báo đệ quy đều được triển khai trong `src/` và được import vào notebook.
- **Tính độc lập & Tái lập**: Mỗi notebook có thể chạy độc lập từ đầu đến cuối trên một kernel mới (`Restart & Run All`).
- **Cơ chế truyền dữ liệu**: Tham số phân tích EDA được xuất ra `data/processed/eda_report.json` ở cuối notebook `02_eda.ipynb` và được nạp tự động ở đầu notebook `03_modeling_forecasting.ipynb`.

## 2. Thứ tự Đọc Khuyến nghị

| STT | Notebook | Giai đoạn tương ứng | Nội dung chính |
|---|---|---|---|
| 1 | `00_data_quality.ipynb` | Step 1 (Workflow) | Đánh giá chất lượng, xử lý 5 cột missing theo Option A, kiểm tra tính liên tục thời gian và thiết lập Canonical Train/Test split. |
| 2 | `01_feature_pipeline.ipynb` | Step 2 (Workflow) | Trình diễn `FeatureBuilder`, đảm bảo Fit/Transform tách bạch, kiểm tra `build_single_step` khớp với `transform` (Train-Serve consistency). |
| 3 | `02_eda.ipynb` | Step 3 (Workflow) | Gọi `EDAPipeline.run()`, phân tích phân phối (Syntetos-Boylan ADI & CV²), mùa vụ, tính dừng, xác thực ngưỡng mưa và xuất `eda_report.json`. |
| 4 | `03_modeling_forecasting.ipynb` | Step 4 & 5 (Workflow) | Nạp `eda_report.json`, huấn luyện mô hình hai giai đoạn (Two-Stage), thực hiện dự báo đa bước (`RecursiveForecaster`) và phân tích suy giảm theo horizon ($h=1..7$). |
| 5 | `99_summary.ipynb` | Tổng kết Capstone | Báo cáo tóm tắt dành cho Reviewer/Interviewer có thời gian giới hạn; bảng tổng hợp đối sánh benchmark. |
