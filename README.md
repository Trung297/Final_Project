# KKBOX Churn Prediction System

Hệ thống dự đoán nguy cơ người dùng ngừng hoạt động trên nền tảng nghe nhạc KKBOX bằng Machine Learning, kết hợp MySQL Data Warehouse, XGBoost, SHAP, FastAPI và Streamlit.

> **Định nghĩa Churn trong nghiên cứu:** người dùng không có hoạt động nghe nhạc được ghi nhận trong tháng dự đoán; không nhất thiết là hủy thuê bao.

## 1. Thành phần hệ thống

- **Data Pipeline:** xử lý dữ liệu KKBOX và nạp vào MySQL Data Warehouse.
- **Machine Learning:** so sánh Logistic Regression, Decision Tree, Random Forest và XGBoost; sử dụng XGBoost cho inference.
- **SHAP:** giải thích các đặc trưng đóng góp vào kết quả dự đoán.
- **FastAPI Backend:** dự đoán theo dữ liệu đầu vào hoặc mã người dùng, truy vấn kết quả batch, cung cấp giải thích SHAP.
- **Streamlit Frontend:** ba trang **Overview**, **Users** và **User Analysis**.
- **Docker Compose:** chạy **2 container Backend + Frontend**; MySQL chạy trực tiếp trên Windows, không nằm trong Docker Compose.

## 2. Yêu cầu trước khi chạy

- Git, Docker Desktop (Linux containers / WSL 2 trên Windows).
- MySQL trên Windows, mặc định cổng `3306`.
- Database `kkbox_dwh` đã được tạo và **nạp dữ liệu**, với các bảng `dim_members`, `dim_date`, `fact_user_daily_activity`, `fact_subscription_transaction`, `churn_predictions`.
- Các model artifact trong `artifacts/`: `xgboost_model.pkl`, `xgboost_info.pkl`, `inference_preprocessor.pkl`, `best_shap_explainer.pkl`.

**Lưu ý:** Repository không cung cấp bản sao MySQL đã nạp dữ liệu. Docker Compose không tự khởi tạo các bảng Data Warehouse. Bảng `churn_predictions` **phải tồn tại trước khi Backend khởi động**, ngay cả khi chưa có bản ghi dự đoán. Có thể tham khảo notebook tại `Notebook/` và mã ETL tại `utils/` để xây dựng dữ liệu.

Nguồn dữ liệu: [WSDM – KKBox's Churn Prediction Challenge (Kaggle)](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge).

## 3. Chạy bằng Docker (khuyến nghị)

Các lệnh sau thực hiện trong **PowerShell** trên Windows.

### Bước 1 — Clone repository

```powershell
git clone https://github.com/Trung297/Final_Project.git
cd Final_Project
```

### Bước 2 — Cấu hình MySQL

```powershell
Copy-Item .env.example .env
```

Mở `.env` và điền thông tin thực tế:

```dotenv
DB_HOST=localhost
DB_PORT=3306
DB_USER=kkbox_app
DB_PASSWORD=YOUR_MYSQL_PASSWORD
DB_NAME=kkbox_dwh
PYTHON_VERSION=3.12
```

- Đổi `DB_USER`, `DB_PASSWORD` theo tài khoản MySQL được cấp quyền.
- `DB_HOST=localhost` dùng cho Python chạy trực tiếp trên Windows. Docker Compose tự ghi đè `DB_HOST=host.docker.internal` bên trong Backend container.
- **Không commit `.env`** hoặc chia sẻ mật khẩu thật.

Nếu cần tạo tài khoản ứng dụng, chạy bằng quyền quản trị MySQL sau khi đã tạo database và các bảng:

```sql
CREATE USER 'kkbox_app'@'%' IDENTIFIED BY 'CHANGE_TO_A_STRONG_PASSWORD';
GRANT SELECT ON kkbox_dwh.* TO 'kkbox_app'@'%';
-- Chỉ cần các quyền sau nếu muốn thực thi batch prediction:
GRANT INSERT, UPDATE ON kkbox_dwh.churn_predictions TO 'kkbox_app'@'%';
```

Tài khoản và phạm vi host cho phép truy cập nên được giới hạn theo môi trường thực tế. Không mở cổng MySQL ra Internet.

### Bước 3 — Khởi động dịch vụ

Đảm bảo Docker Desktop và MySQL trên Windows đều đang chạy:

```powershell
docker compose up --build -d
docker compose ps
```

| Dịch vụ | Đường dẫn |
| --- | --- |
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Swagger API Docs | http://localhost:8000/docs |

### Bước 4 — Kiểm tra API và giao diện

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/predictions/summary
```

Mở http://localhost:8501 để thử **Overview**, **Users** và **User Analysis**. Chọn một `msno` có hoạt động trong tháng 02/2017 để xem dự đoán tháng 03/2017 và SHAP.

**Chú ý:** Trong Docker, Frontend gọi Backend qua `http://backend:8000` (tên service nội bộ). Không đặt địa chỉ này thành `http://localhost:8000` trong container.

## 4. Dự đoán hàng loạt (nếu chưa có kết quả)

Dashboard đọc dữ liệu dự đoán đã lưu trong bảng `churn_predictions`. Nếu bảng đã tồn tại nhưng chưa có dữ liệu cho model hiện tại, chạy:

Thử một batch 500 người dùng:

```powershell
docker compose exec backend python -m backend.batch_prediction --start 2017-02-01 --end 2017-02-28 --batch-size 500 --limit-batches 1
```

Chạy đầy đủ:

```powershell
docker compose exec backend python -m backend.batch_prediction --start 2017-02-01 --end 2017-02-28 --batch-size 500
```

Quá trình yêu cầu đủ bảng DWH và quyền `SELECT`, `INSERT`, `UPDATE`. Sau khi hoàn tất, tải lại Dashboard.

## 5. Các lệnh Docker hữu ích

```powershell
# Xem trạng thái
docker compose ps

# Xem log
docker compose logs -f backend
docker compose logs -f frontend

# Build lại sau khi sửa code
docker compose up --build -d

# Dừng hệ thống
docker compose down
```

## 6. Chạy trực tiếp không qua Docker (tùy chọn)

Sử dụng Python tương thích với model artifact (Docker mặc định Python 3.12), và đã chuẩn bị MySQL + `.env` như hướng dẫn ở trên.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-backend.txt
python -m pip install -r requirements-frontend.txt
```

Terminal 1 — Backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — Frontend:

```powershell
$env:API_BASE_URL = "http://127.0.0.1:8000"
python -m streamlit run frontend/app.py
```

**Lưu ý tương thích:** `requirements-backend.txt` hiện chưa khóa toàn bộ phiên bản. Các artifact pickle/joblib cần phiên bản Python và thư viện ML tương thích với môi trường lúc xuất model.

## 7. Lỗi thường gặp

| Lỗi | Giải pháp |
| --- | --- |
| Backend không khởi động | Kiểm tra MySQL, file `.env`, bảng `churn_predictions`; xem `docker compose logs backend`. |
| `Access denied` hoặc không kết nối được MySQL | Kiểm tra username/password, grants, firewall, `host.docker.internal`. |
| Dashboard không có dữ liệu | Chạy batch prediction cho tháng 02/2017, kiểm tra bản ghi trong `churn_predictions` tương ứng model hiện tại. |
| Lỗi XGBoost/SHAP/joblib | Kiểm tra các file trong `artifacts/` và phiên bản thư viện khi train model. |
| Port 8000/8501 đã sử dụng | Dừng Uvicorn/Streamlit local hoặc chỉnh port mapping trong `docker-compose.yml`. |
| Frontend không gọi được Backend | Kiểm tra service `backend` healthy và `API_BASE_URL=http://backend:8000`. |

## 8. Cấu trúc thư mục

```text
Final_Project/
├── Notebook/                  # EDA, Preprocessing, Model Training, SHAP
├── utils/                     # ETL, kết nối dữ liệu, inference
├── backend/                   # FastAPI, batch prediction, API truy vấn, SHAP
├── frontend/app.py            # Streamlit Dashboard
├── artifacts/                 # Model và preprocessing artifacts
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements-backend.txt
├── requirements-frontend.txt
├── .env.example
└── README.md
```

## 9. Bảo mật và phạm vi triển khai

Đây là hệ thống **chạy local**, chưa triển khai public. Các cổng 8000 và 8501 hiện được Docker Compose giới hạn trên `127.0.0.1`. Nếu triển khai Internet trong tương lai, cần bổ sung xác thực, phân quyền, HTTPS, quản lý secret và cơ sở dữ liệu phù hợp.

Chỉ tải các file model `.pkl` từ nguồn đáng tin cậy.

Tài liệu Docker chi tiết đã có sẵn trong [README-DOCKER.md](README-DOCKER.md).
