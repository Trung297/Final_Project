
# KKBOX Churn Prediction System

Hệ thống dự đoán người dùng rời bỏ dịch vụ nghe nhạc KKBOX bằng Machine Learning, kết hợp MySQL, XGBoost, SHAP, FastAPI, Streamlit và Docker.

> **Định nghĩa trong đề tài:** `Churn = 1` khi người dùng **không phát sinh hoạt động nghe nhạc trong Prediction Window** (tháng tiếp theo). Đây là *behavioral churn*, **không đồng nghĩa** với việc người dùng hủy gói thuê bao.
> **Phạm vi triển khai:** Docker Compose hiện chỉ chạy **Backend FastAPI và Frontend Streamlit**. **MySQL được cài và chạy trên máy Windows**, không phải một service Docker trong repository này.

## 1. Thứ tự chạy các Notebook

Trước khi chạy thì giải nén file rar trong thư mục kkbox_dataset

**Bước 1: `Notebook/EDA.ipynb`**

- Đọc, làm sạch dữ liệu KKBOX và thực hiện ETL vào MySQL Data Warehouse.
- Phân tích dữ liệu, xây dựng hai snapshot và gán nhãn Churn.
- **File đầu ra:** `artifacts/master_eda.pkl`.

*Lưu ý: Cần tạo các bảng MySQL trước. Nếu database đã có dữ liệu, bỏ qua các ô ETL nạp dữ liệu để tránh trùng lặp.*

**Bước 2: `Notebook/Preprocessing.ipynb`**

- Đọc `master_eda.pkl`.
- Feature Engineering, chia Train/Validation/Test, chuẩn hóa và áp dụng SMOTE.
- **File đầu ra:**
  - `artifacts/processed_data.pkl`
  - `artifacts/inference_preprocessor.pkl`

**Bước 3: `Notebook/model.ipynb`**

- Đọc `processed_data.pkl`.
- Huấn luyện và đánh giá Logistic Regression, Decision Tree, Random Forest và XGBoost.
- **File đầu ra:** Các file model và thông tin mô hình trong `artifacts/`:
  - `logistic_model.pkl`, `logistic_info.pkl`
  - `decision_tree_model.pkl`, `decision_tree_info.pkl`
  - `random_forest_model.pkl`, `random_forest_info.pkl`
  - `xgboost_model.pkl`, `xgboost_info.pkl`

**Bước 4: `Notebook/SHAP.ipynb`**

- Đọc `processed_data.pkl` và các mô hình đã huấn luyện.
- Phân tích SHAP và lưu bộ giải thích của XGBoost.
- **File đầu ra:** `artifacts/best_shap_explainer.pkl`.

**Bước 5: Batch Prediction**

Chạy từ thư mục gốc project:

`python -m backend.batch_prediction`

- Dự đoán Churn cho người dùng trong tháng quan sát.
- **Đầu ra:** Kết quả được lưu vào bảng `churn_predictions` trong MySQL.

*Bảng `churn_predictions` cần được tạo trước khi chạy.*

## 2. Chạy hệ thống bằng Docker

**Bước 1: Chuẩn bị môi trường**

- Cài đặt Docker Desktop và MySQL.
- Đảm bảo database `kkbox_dwh` đã có dữ liệu và bảng `churn_predictions`.
- Tạo file `.env` từ `.env.example`, sau đó cấu hình tài khoản và mật khẩu MySQL.
- Đảm bảo thư mục `artifacts/` có đủ các file model, preprocessor và SHAP.

**Bước 2: Khởi động Docker**

Mở Terminal tại thư mục gốc project và chạy:

`docker compose up --build -d`

**Bước 3: Truy cập hệ thống**

- **Frontend (Streamlit):** http://localhost:8501
- **Backend (FastAPI):** http://localhost:8000
- **Swagger API:** http://localhost:8000/docs

**Bước 4: Dừng hệ thống**

`docker compose down`

## 3. Lưu ý

- Docker Compose chỉ chạy **hai container: Backend và Frontend**. MySQL chạy trên máy Windows.
- Nếu đã có các file model trong `artifacts/` và dữ liệu MySQL đầy đủ, **không cần chạy lại toàn bộ notebook**.
- Hệ thống hiện được triển khai trên môi trường local, chưa public lên Internet.

## 4. Cấu trúc thư mục

```text
Final_Project/
├── Notebook/
│   ├── EDA.ipynb                    # ETL, EDA, tạo master_eda.pkl
│   ├── Preprocessing.ipynb          # Feature Engineering, Split, Scale, SMOTE
│   ├── model.ipynb                  # Train/đánh giá/lưu 4 thuật toán
│   ├── SHAP.ipynb                   # SHAP, lưu best_shap_explainer.pkl
│   └── kkbox-dataset/              # Chứa file rar, giải nén trước khi chạy các notebook
├── utils/
│   ├── ETL_Pipeline.py              # Hàm extract/transform/load MySQL
│   ├── cleaning.py
│   ├── base_model.py                # Hàm huấn luyện và đánh giá
│   ├── inference.py                 # Chạy model đã huấn luyện
│   ├── config.py                    # Đọc DB_* từ biến môi trường
│   └── db.py
├── artifacts/                       # Pickle model, preprocessor, SHAP, dữ liệu xử lý
├── backend/
│   ├── main.py                      # FastAPI
│   ├── batch_prediction.py          # Sinh và lưu churn_predictions
│   ├── repository.py
│   ├── predictions_api.py
│   └── shap_api.py
├── frontend/
│   └── app.py                       # Streamlit Dashboard
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements-backend.txt
├── requirements-frontend.txt
├── .env.example
└── README.md
```

---

**Lưu ý bảo mật mã nguồn:** Một số hàm ETL trong `utils/ETL_Pipeline.py` hiện có giá trị mật khẩu *fallback* viết trực tiếp trong source. Trước khi chia sẻ hoặc triển khai, nên bỏ các giá trị mặc định nhạy cảm này, chỉ lấy mật khẩu từ `.env`; nếu từng công bố mật khẩu sử dụng thực tế, hãy đổi mật khẩu và rà soát lịch sử commit. Không đưa mật khẩu vào README hoặc GitHub.

**Ghi chú về tái lập thực nghiệm:** README phản ánh các file và lệnh **đang có trong repository**. Phần cấu hình schema MySQL và quy trình nạp DWH từ database rỗng chưa được đóng gói thành một script tự động hoàn chỉnh; phải chuẩn bị schema và kiểm tra các ô ETL trong `EDA.ipynb` trước khi chạy lại toàn bộ. Khi chỉ cần chạy hệ thống đã huấn luyện, có thể dùng các model artifact có sẵn và bỏ qua các bước train, nhưng **vẫn cần database MySQL tương thích**.
