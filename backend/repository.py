from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterator
import logging

import pandas as pd
import warnings
from sqlalchemy import Engine, bindparam, text

logger = logging.getLogger(__name__)


RAW_COLUMNS = (
    "msno",
    "active_days",
    "total_secs",
    "num_100",
    "num_25",
    "num_unq",
    "secs_first_half",
    "secs_second_half",
    "city",
    "age",
    "gender",
    "registered_via",
    "tx_count",
    "total_paid",
    "auto_renew_ratio",
    "cancel_count",
)

NUMERIC_COLUMNS = tuple(c for c in RAW_COLUMNS if c not in ("msno", "gender"))


def _parse_date(value: date | str, name: str) -> date:
    if isinstance(value, datetime):
        raise ValueError(f"{name} phải là ngày YYYY-MM-DD, không phải datetime")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} phải có định dạng YYYY-MM-DD") from exc
    raise TypeError(f"{name} phải là date hoặc chuỗi YYYY-MM-DD")


class ChurnRepository:

    def __init__(self, engine: Engine, *, transaction_mode: str = "eda_legacy"):
        if engine is None:
            raise ValueError("Cần cung cấp SQLAlchemy engine")
        if transaction_mode not in {"eda_legacy", "source_swapped", "standard"}:
            raise ValueError("transaction_mode phải là eda_legacy/source_swapped/standard")
        self.engine = engine
        self.transaction_mode = transaction_mode

    @staticmethod
    def _window_params(
        observation_start: date | str,
        observation_end: date | str,
    ) -> dict:
        start = _parse_date(observation_start, "observation_start")
        end = _parse_date(observation_end, "observation_end")
        if start >= end:
            raise ValueError("Observation Window phải có ít nhất 2 ngày")

        days = (end - start).days + 1
        half2_start = start + timedelta(days=days // 2)
        return {
            "obs_start": start,
            "end_exclusive": end + timedelta(days=1),
            "half2_start": half2_start,
        }

    def _query(self, filter_user: bool, *, filter_batch: bool = False) -> str:
        if filter_user and filter_batch:
            raise ValueError("Chỉ sử dụng một chế độ lọc người dùng")
        if filter_user:
            activity_filter = "AND ul.msno = :msno"
            transaction_filter = "AND t.msno = :msno"
        elif filter_batch:
            activity_filter = "AND ul.msno IN :msnos"
            transaction_filter = "AND t.msno IN :msnos"
        else:
            activity_filter = ""
            transaction_filter = ""

        if self.transaction_mode == "source_swapped":
            paid_expr = "SUM(t.is_auto_renew)"
            auto_expr = "AVG(t.actual_amount_paid)"
        else:
            paid_expr = "SUM(t.actual_amount_paid)"
            auto_expr = "AVG(t.is_auto_renew)"

        return f"""
            SELECT
                a.msno,
                a.active_days,
                a.total_secs,
                a.num_100,
                a.num_25,
                a.num_unq,
                a.secs_first_half,
                a.secs_second_half,
                COALESCE(m.city, -1) AS city,
                COALESCE(m.bd, -1) AS age,
                COALESCE(m.gender, 'Unknown') AS gender,
                COALESCE(m.registered_via, -1) AS registered_via,
                COALESCE(tr.tx_count, 0) AS tx_count,
                COALESCE(tr.total_paid, 0) AS total_paid,
                COALESCE(tr.auto_renew_ratio, 0) AS auto_renew_ratio,
                COALESCE(tr.cancel_count, 0) AS cancel_count
            FROM (
                SELECT
                    ul.msno,
                    COUNT(DISTINCT ul.date) AS active_days,
                    COALESCE(SUM(ul.total_secs), 0) AS total_secs,
                    COALESCE(SUM(ul.num_100), 0) AS num_100,
                    COALESCE(SUM(ul.num_25), 0) AS num_25,
                    COALESCE(SUM(ul.num_unq), 0) AS num_unq,
                    COALESCE(SUM(CASE
                        WHEN ul.date < :half2_start
                        THEN ul.total_secs ELSE 0 END), 0) AS secs_first_half,
                    COALESCE(SUM(CASE
                        WHEN ul.date >= :half2_start
                        THEN ul.total_secs ELSE 0 END), 0) AS secs_second_half
                FROM fact_user_daily_activity AS ul
                WHERE ul.date >= :obs_start
                  AND ul.date < :end_exclusive
                  {activity_filter}
                GROUP BY ul.msno
            ) AS a
            LEFT JOIN dim_members AS m ON m.msno = a.msno
            LEFT JOIN (
                SELECT
                    t.msno,
                    COUNT(*) AS tx_count,
                    {paid_expr} AS total_paid,
                    {auto_expr} AS auto_renew_ratio,
                    SUM(t.is_cancel) AS cancel_count
                FROM fact_subscription_transaction AS t
                WHERE t.transaction_date < :end_exclusive
                  {transaction_filter}
                GROUP BY t.msno
            ) AS tr ON tr.msno = a.msno
            ORDER BY a.msno
        """

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame(columns=RAW_COLUMNS)
        missing = set(RAW_COLUMNS).difference(data.columns)
        if missing:
            raise ValueError(f"Thiếu cột trong truy vấn MySQL: {sorted(missing)}")

        result = data.loc[:, RAW_COLUMNS].copy()
        for col in NUMERIC_COLUMNS:
            result[col] = pd.to_numeric(result[col], errors="raise")

        result["gender"] = result["gender"].fillna("Unknown")
        result[["city", "age", "registered_via"]] = result[
            ["city", "age", "registered_via"]
        ].fillna(-1)
        result[["tx_count", "total_paid", "auto_renew_ratio", "cancel_count"]] = result[
            ["tx_count", "total_paid", "auto_renew_ratio", "cancel_count"]
        ].fillna(0)

        if self.transaction_mode == "eda_legacy":
            # IMPORTANT: this mirrors notebook cell 35 AFTER aggregation.
            # For two source-swapped tx with flag=1, this creates ratio=2.
            paid = result["total_paid"].copy()
            result["total_paid"] = result["auto_renew_ratio"]
            result["auto_renew_ratio"] = paid

        valid_ratio = result["auto_renew_ratio"].between(-1e-8, 1 + 1e-8)
        if not bool(valid_ratio.all()):
            message = (
            )
            if self.transaction_mode == "eda_legacy":
                warnings.warn(message, UserWarning, stacklevel=2)
            else:
                raise ValueError(message)
        if bool((result["total_paid"] < 0).any()):
            raise ValueError("total_paid có giá trị âm; cần kiểm tra dữ liệu nguồn")
        return result.reset_index(drop=True)

    def get_user_features(
        self,
        msno: str,
        observation_start: date | str = "2017-02-01",
        observation_end: date | str = "2017-02-28",
    ) -> pd.DataFrame:
        if not isinstance(msno, str) or not msno.strip():
            raise ValueError("msno phải là chuỗi không rỗng")
        params = self._window_params(observation_start, observation_end)
        params["msno"] = msno
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(self._query(filter_user=True)), conn, params=params)
        return self._prepare(df)

    def get_active_users_features(
        self,
        observation_start: date | str = "2017-02-01",
        observation_end: date | str = "2017-02-28",
    ) -> pd.DataFrame:
        """Raw features for ALL users active in this observation window."""
        params = self._window_params(observation_start, observation_end)
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(self._query(filter_user=False)), conn, params=params)
        return self._prepare(df)

    def iter_active_users_features(
        self,
        observation_start: date | str = "2017-02-01",
        observation_end: date | str = "2017-02-28",
        batch_size: int = 1000,
    ) -> Iterator[pd.DataFrame]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size phải là số nguyên dương")

        params = self._window_params(observation_start, observation_end)
        # With the existing index (msno, date), MySQL can scan msno order
        # while checking the observation window, then stop at LIMIT.
        page_sql_base = """
            SELECT DISTINCT ul.msno
            FROM fact_user_daily_activity AS ul
            WHERE ul.date >= :obs_start
              AND ul.date < :end_exclusive
        """
        batch_sql = text(self._query(filter_user=False, filter_batch=True)).bindparams(
            bindparam("msnos", expanding=True)
        )

        after_msno: str | None = None
        batch_number = 0
        with self.engine.connect() as conn:
            while True:
                # Keep the keyset predicate sargable. Do not use OFFSET
                # (later pages would become slower as OFFSET grows).
                after_filter = "AND ul.msno > :after_msno" if after_msno is not None else ""
                page_sql = text(
                    page_sql_base
                    + after_filter
                    + " ORDER BY ul.msno LIMIT :page_size"
                )
                page_params = {**params, "page_size": batch_size}
                if after_msno is not None:
                    page_params["after_msno"] = after_msno

                logger.info("Đang chọn danh sách msno cho batch %d...", batch_number + 1)
                msnos = list(conn.execute(page_sql, page_params).scalars())
                if not msnos:
                    return

                logger.info("Batch %d: tổng hợp đặc trưng cho %d người dùng...", batch_number + 1, len(msnos))
                data = pd.read_sql_query(
                    batch_sql,
                    conn,
                    params={**params, "msnos": msnos},
                )
                prepared = self._prepare(data)
                if len(prepared) != len(msnos):
                    raise RuntimeError(
                        "Số người dùng trong batch không khớp số msno được chọn; "
                        "hãy kiểm tra dữ liệu hoặc collation của cột msno"
                    )
                yield prepared

                batch_number += 1
                after_msno = msnos[-1]
                if len(msnos) < batch_size:
                    return

    def count_active_users(
        self,
        observation_start: date | str = "2017-02-01",
        observation_end: date | str = "2017-02-28",
    ) -> int:
        params = self._window_params(observation_start, observation_end)
        sql = text("""
            SELECT COUNT(DISTINCT msno)
            FROM fact_user_daily_activity
            WHERE date >= :obs_start AND date < :end_exclusive
        """)
        with self.engine.connect() as conn:
            return int(conn.execute(sql, params).scalar_one())
