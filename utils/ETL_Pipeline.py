import pandas as pd
from datetime import datetime
import logging
import os
import pymysql
import gc

from sqlalchemy import create_engine
from utils.config import *
from utils.db import *
from utils.cleaning import *
from utils.stats import *

# logging.basicConfig(
#     level = logging.INFO,
#     format = "%(asctime)s - %(levelname)s - %(message)s",
#     datefmt = "%H:%M:%S",
#     handlers = [
#         logging.FileHandler(f"log/ETL_UserLogs_Pipeline_{datetime.now():%Y-%m-%d}.log", 
#                             encoding="utf-8"),
#         logging.StreamHandler()
#     ]
# )

# log = logging.getLogger(__name__)

def extract(file_paths, chunksize=2_000_000):
    # log.info("EXTRACT start (Streaming Data Mode)")
    for file in file_paths:
        # log.info(f"Extracting file: {file}")
        for i, chunk in enumerate(pd.read_csv(file, chunksize=chunksize)):
            yield file, i, chunk
    # log.info("EXTRACT completely done")


def transform(df_user_logs, df_transactions, df_members):

    df_user_logs['date'] = pd.to_datetime(df_user_logs['date'], format='%Y%m%d')
    df_user_logs = df_user_logs.dropna(subset=['msno'])
    df_user_logs = df_user_logs.drop_duplicates(subset=['msno', 'date'])
    invalid_idx = df_user_logs.index[(df_user_logs['total_secs'] < 0) | (df_user_logs['total_secs'] > 86400)]
    if len(invalid_idx) > 0:
        df_user_logs.drop(index=invalid_idx, inplace=True)
    del invalid_idx
    df_user_logs['dtime_data_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'].astype(str), format='%Y%m%d')
    df_transactions['membership_expire_date'] = pd.to_datetime(df_transactions['membership_expire_date'], format='%Y%m%d')
    df_transactions = df_transactions.dropna(subset=['msno'])
    df_transactions = df_transactions.drop_duplicates()
    df_transactions = df_transactions.sort_values(
        by=['msno', 'transaction_date', 'membership_expire_date', 'actual_amount_paid']
    )
    df_transactions = df_transactions.drop_duplicates(
        subset=['msno', 'transaction_date'], 
        keep='last'
    )
    df_transactions = df_transactions[(df_transactions['actual_amount_paid'] >= 0) & (df_transactions['payment_plan_days'] >= 0)]
    df_transactions['dtime_data_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    df_members['registration_init_time'] = pd.to_datetime(df_members['registration_init_time'], format='%Y%m%d')
    df_members = df_members.dropna(subset=['msno'])
    if hasattr(df_members['gender'], 'cat'):
        if 'Unknown' not in df_members['gender'].cat.categories:
            df_members['gender'] = df_members['gender'].cat.add_categories('Unknown')
        df_members['gender'] = df_members['gender'].fillna('Unknown')
    else:
        df_members['gender'] = df_members['gender'].astype(object).fillna('Unknown')
    df_members['bd'] = df_members['bd'].apply(lambda x: x if 0 < x <= 100 else -1)
    df_members['dtime_data_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df_user_logs, df_transactions, df_members

def load_data_warehouse(df_user_logs, df_transactions, df_members):
    print(f"--- KẾT NỐI VÀO DATABASE: ---")
    
    connection = pymysql.connect(
        host=DB_CONFIG.get('host', 'localhost'),
        user=DB_CONFIG.get('user', 'root'),
        password=DB_CONFIG.get('password', '2972005'),
        database=DB_CONFIG.get('database', 'kkbox_dwh'),
        port=DB_CONFIG.get('port', 3306),
        local_infile=True,
        connect_timeout=60,
        read_timeout=36000,
        write_timeout=36000,
        autocommit=False
    )
    
    # 2. Tạo thư mục tạm để chứa các tệp CSV chuyển đổi
    temp_dir = "temp_load_staging"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_members = os.path.abspath(os.path.join(temp_dir, "clean_members.csv")).replace("\\", "/")
    file_trans = os.path.abspath(os.path.join(temp_dir, "clean_trans.csv")).replace("\\", "/")
    file_logs = os.path.abspath(os.path.join(temp_dir, "clean_logs.csv")).replace("\\", "/")
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        cursor.execute("SET UNIQUE_CHECKS = 0;")
        cursor.execute("SET SQL_LOG_BIN = 0;")
        
        print("1. Đang xuất CSV và nạp dim_members...")
        df_members.to_csv(file_members, index=False, na_rep='\\N')
        sql_members = f"""
            LOAD DATA LOCAL INFILE '{file_members}'
            INTO TABLE dim_members
            FIELDS TERMINATED BY ',' 
            OPTIONALLY ENCLOSED BY '"'
            LINES TERMINATED BY '\n'
            IGNORE 1 LINES;
        """
        cursor.execute(sql_members)
        connection.commit()
        print(f" -> Đã nạp thành công dim_members ({len(df_members):,} dòng).")

        # -------------------------------------------------------------
        # BƯỚC 2: NẠP fact_subscription_transaction (~22.9 triệu dòng)
        # -------------------------------------------------------------
        print("2. Đang xuất CSV và nạp fact_subscription_transaction...")
        df_transactions.to_csv(file_trans, index=False, na_rep='\\N')
        sql_trans = f"""
            LOAD DATA LOCAL INFILE '{file_trans}'
            INTO TABLE fact_subscription_transaction
            FIELDS TERMINATED BY ',' 
            OPTIONALLY ENCLOSED BY '"'
            LINES TERMINATED BY '\n'
            IGNORE 1 LINES;
        """
        cursor.execute(sql_trans)
        connection.commit()
        print(f" -> Đã nạp thành công fact_subscription_transaction ({len(df_transactions):,} dòng).")

        # -------------------------------------------------------------
        # BƯỚC 3: NẠP fact_user_daily_activity (~20M - 38M dòng)
        # -------------------------------------------------------------
        print("3. Đang xuất CSV và nạp fact_user_daily_activity...")
        df_user_logs.to_csv(file_logs, index=False, na_rep='\\N')
        sql_logs = f"""
            LOAD DATA LOCAL INFILE '{file_logs}'
            INTO TABLE fact_user_daily_activity
            FIELDS TERMINATED BY ',' 
            OPTIONALLY ENCLOSED BY '"'
            LINES TERMINATED BY '\n'
            IGNORE 1 LINES;
        """
        cursor.execute(sql_logs)
        connection.commit()
        print(f" -> Đã nạp thành công fact_user_daily_activity ({len(df_user_logs):,} dòng).")

    except Exception as e:
        connection.rollback()
        print(f"[LỖI XẢY RA]: {e}")
        raise e

    finally:
        # Bật lại toàn bộ các ràng buộc toàn vẹn của CSDL
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        cursor.execute("SET UNIQUE_CHECKS = 1;")
        cursor.execute("SET SQL_LOG_BIN = 1;")
        cursor.close()
        connection.close()

        # Dọn dẹp toàn bộ tệp tạm để giải phóng bộ nhớ đĩa
        for f in [file_members, file_trans, file_logs]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            
        print("--- HOÀN TẤT BULK LOAD TOÀN BỘ DATA WAREHOUSE THÀNH CÔNG ---")

def populate_dim_date():
    engine = make_engine(DB_CONFIG)
    
    date_range = pd.date_range(start='2015-01-01', end='2017-04-30', freq='D')
    df_date = pd.DataFrame({'date': date_range})
    
    df_date['day'] = df_date['date'].dt.day
    df_date['month'] = df_date['date'].dt.month
    df_date['year'] = df_date['date'].dt.year
    df_date['is_weekend'] = df_date['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    df_date.to_sql('dim_date', con=engine, if_exists='append', index=False)
    print(f"Đã nạp thành công {len(df_date)} ngày vào dim_date trên MySQL.")


def load_user_logs_final(df_user_logs):
    db_name = DB_CONFIG.get('database', 'kkbox_dwh')
    print(f"--- BẮT ĐẦU NẠP fact_user_daily_activity VÀO {db_name} ---")
    
    conn = pymysql.connect(
        host=DB_CONFIG.get('host', 'localhost'),
        user=DB_CONFIG.get('user', 'root'),
        password=DB_CONFIG.get('password', '2972005'),
        database=db_name,
        port=DB_CONFIG.get('port', 3306),
        local_infile=True,
        connect_timeout=120,
        read_timeout=36000,
        write_timeout=36000,
        autocommit=False
    )
    
    temp_csv = "temp_clean_logs.csv"
    cursor = conn.cursor()
    
    try:
        # Tạm tắt các chốt chặn kiểm tra để đạt tốc độ nạp khối tối đa
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        cursor.execute("SET UNIQUE_CHECKS = 0;")
        cursor.execute("SET SQL_LOG_BIN = 0;")
        
        print("1. Đang xuất CSV tạm từ DataFrame...")
        df_user_logs.to_csv(temp_csv, index=False, na_rep='\\N')
        abs_path = os.path.abspath(temp_csv).replace("\\", "/")
        
        print(f"2. Đang nạp khối {len(df_user_logs):,} dòng vào MySQL...")
        sql = f"""
            LOAD DATA LOCAL INFILE '{abs_path}'
            INTO TABLE fact_user_daily_activity
            FIELDS TERMINATED BY ',' 
            OPTIONALLY ENCLOSED BY '"'
            LINES TERMINATED BY '\n'
            IGNORE 1 LINES;
        """
        cursor.execute(sql)
        conn.commit()
        print("-> Đã nạp thành công 100% dữ liệu vào fact_user_daily_activity!")
        
    except Exception as e:
        conn.rollback()
        print(f"[LỖI XẢY RA]: {e}")
        raise e
        
    finally:
        # Bật lại các ràng buộc toàn vẹn
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        cursor.execute("SET UNIQUE_CHECKS = 1;")
        cursor.execute("SET SQL_LOG_BIN = 1;")
        cursor.close()
        conn.close()
        
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        print("--- HOÀN TẤT NẠP DATA WAREHOUSE ---")

def process_only_user_logs(df_user_logs):
    print(f"--- 1. BẮT ĐẦU TRANSFORM fact_user_daily_activity ({len(df_user_logs):,} dòng) ---")
    
    # 1.1. Parse ngày trực tiếp từ int32 mà không qua ép chuỗi (.astype(str)) để tiết kiệm RAM
    df_user_logs['date'] = pd.to_datetime(df_user_logs['date'].astype('int32'), format='%Y%m%d')
    
    # 1.2. Loại bỏ bản ghi thiếu msno và trùng lặp khóa chính
    df_user_logs.dropna(subset=['msno'], inplace=True)
    df_user_logs.drop_duplicates(subset=['msno', 'date'], inplace=True)
    
    # 1.3. Lọc ngoại lai bằng index (KHÔNG dùng Boolean Mask để tránh ArrowMemoryError)
    invalid_idx = df_user_logs.index[
        (df_user_logs['total_secs'] < 0) | (df_user_logs['total_secs'] > 86400)
    ]
    if len(invalid_idx) > 0:
        df_user_logs.drop(index=invalid_idx, inplace=True)
    del invalid_idx
    
    # 1.4. Bổ sung dấu thời gian ETL
    df_user_logs['dtime_data_created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gc.collect()
    print(f" -> Transform hoàn tất: {len(df_user_logs):,} dòng sạch sẵn sàng nạp.")