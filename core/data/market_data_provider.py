from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import akshare as ak
except Exception:  # pragma: no cover - optional dependency
    ak = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

try:
    from Ashare import get_price as ashare_get_price
except Exception:  # pragma: no cover - optional dependency
    ashare_get_price = None


@dataclass(frozen=True)
class MarketDataQuery:
    symbol: str
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    period_days: int = 365
    adjust: str = "qfq"
    market: str = "CN"


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    created_at_utc: str
    query: Dict[str, object]
    provider: str
    rows: int
    sha256: str
    file: str


class MarketDataProvider:
    """
    Unified OHLCV provider with source fallback, cache, and snapshot reproducibility.
    """

    INTERVAL_MAP_AK = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
    }
    VALID_INTERVALS = set(INTERVAL_MAP_AK.keys()) | {"1d"}
    STANDARD_COLUMNS = [
        "datetime",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "symbol",
        "interval",
        "provider",
        "adjust",
        "market",
    ]

    def __init__(
        self,
        cache_dir: str = "data/cache/market",
        snapshot_dir: str = "data/snapshots",
        provider_priority: Optional[Sequence[str]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.provider_priority = tuple(provider_priority or ("akshare", "yfinance", "ashare"))
        self._cn_trade_calendar: Optional[pd.Series] = None

    def get_ohlcv(
        self,
        query: MarketDataQuery,
        use_cache: bool = True,
        freeze_snapshot: bool = False,
        snapshot_name: Optional[str] = None,
    ) -> pd.DataFrame:
        self._validate_query(query)

        cache_key = self._build_cache_key(query)
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None and not cached.empty:
                return cached

        frame, provider = self._fetch_with_fallback(query)
        normalized = self._normalize(frame, query, provider)

        if query.market.upper() == "CN":
            normalized = self._align_cn_calendar(normalized, query.interval)

        if query.start:
            start_ts = pd.to_datetime(query.start, errors="coerce")
            normalized = normalized[pd.to_datetime(normalized["datetime"], errors="coerce") >= start_ts]
        if query.end:
            end_ts = pd.to_datetime(query.end, errors="coerce")
            normalized = normalized[pd.to_datetime(normalized["datetime"], errors="coerce") <= end_ts]
        if not query.start and not query.end and query.period_days > 0 and query.interval == "1d":
            normalized = normalized.tail(query.period_days)

        normalized = normalized.reset_index(drop=True)
        if use_cache and not normalized.empty:
            self._write_cache(cache_key, normalized)

        if freeze_snapshot and not normalized.empty:
            self.create_snapshot(
                frame=normalized,
                query=query,
                provider=provider,
                snapshot_name=snapshot_name,
            )
        return normalized

    def create_snapshot(
        self,
        frame: pd.DataFrame,
        query: MarketDataQuery,
        provider: str,
        snapshot_name: Optional[str] = None,
    ) -> str:
        payload = frame.to_csv(index=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = (snapshot_name or query.symbol).replace(" ", "_")
        snapshot_id = f"{stamp}_{slug}_{digest[:10]}"
        root = self.snapshot_dir / snapshot_id
        root.mkdir(parents=True, exist_ok=True)

        data_file = root / "data.csv"
        manifest_file = root / "manifest.json"
        data_file.write_text(payload, encoding="utf-8")

        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            query=asdict(query),
            provider=provider,
            rows=int(len(frame)),
            sha256=digest,
            file=str(data_file.as_posix()),
        )
        manifest_file.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> pd.DataFrame:
        data_file = self.snapshot_dir / snapshot_id / "data.csv"
        if not data_file.exists():
            return pd.DataFrame()
        return pd.read_csv(data_file)

    def _validate_query(self, query: MarketDataQuery) -> None:
        if not query.symbol:
            raise ValueError("symbol is required")
        if query.interval not in self.VALID_INTERVALS:
            raise ValueError(f"unsupported interval: {query.interval}")
        if query.adjust not in ("", "qfq", "hfq"):
            raise ValueError("adjust must be one of '', 'qfq', 'hfq'")

    def _build_cache_key(self, query: MarketDataQuery) -> str:
        payload = json.dumps(
            {
                "query": asdict(query),
                "priority": list(self.provider_priority),
                "version": 1,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.csv"

    def _read_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _write_cache(self, cache_key: str, frame: pd.DataFrame) -> None:
        path = self._cache_path(cache_key)
        frame.to_csv(path, index=False)

    def _fetch_with_fallback(self, query: MarketDataQuery) -> Tuple[pd.DataFrame, str]:
        errors: List[str] = []
        for provider in self.provider_priority:
            try:
                if provider == "synthetic":
                    return self._fetch_synthetic(query, errors), provider
                if provider == "akshare":
                    return self._fetch_from_akshare(query), provider
                if provider == "yfinance":
                    return self._fetch_from_yfinance(query), provider
                if provider == "ashare":
                    return self._fetch_from_ashare(query), provider
            except Exception as exc:  # pragma: no cover - network and optional package path
                errors.append(f"{provider}: {exc}")
        if str(os.environ.get("CIVITAS_DISABLE_SYNTHETIC_MARKET_FALLBACK", "")).lower() not in {"1", "true", "yes"}:
            return self._fetch_synthetic(query, errors), "synthetic"
        raise RuntimeError("; ".join(errors) if errors else "no data provider available")

    def _fetch_synthetic(self, query: MarketDataQuery, errors: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Build deterministic OHLCV fallback data for offline CI/demo replay."""

        seed_text = json.dumps(asdict(query), sort_keys=True, ensure_ascii=True)
        seed_int = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        base = 3000.0 if self._looks_index_symbol(query.symbol) else 100.0
        drift = ((seed_int % 97) - 48) / 100000.0
        amp = 0.006 + (seed_int % 13) / 10000.0

        def _price(idx: int) -> float:
            seasonal = math.sin((idx + (seed_int % 31)) / 5.0) * amp
            trend = drift * idx
            return max(1.0, base * (1.0 + trend + seasonal))

        end_ts = pd.to_datetime(query.end, errors="coerce") if query.end else pd.Timestamp("2024-12-31")
        if pd.isna(end_ts):
            end_ts = pd.Timestamp("2024-12-31")
        start_ts = pd.to_datetime(query.start, errors="coerce") if query.start else None
        if start_ts is not None and pd.isna(start_ts):
            start_ts = None

        if query.interval == "1d":
            if start_ts is not None:
                dates = pd.bdate_range(start=start_ts, end=end_ts)
            else:
                dates = pd.bdate_range(end=end_ts, periods=max(10, int(query.period_days or 60)))
            rows = []
            prev = _price(0)
            for idx, day in enumerate(dates):
                close = _price(idx)
                open_ = prev
                high = max(open_, close) * (1.0 + 0.0025 + (idx % 5) * 0.0002)
                low = min(open_, close) * (1.0 - 0.0025 - (idx % 3) * 0.0002)
                volume = int(800_000_000 + (seed_int % 1000) * 10000 + (idx % 17) * 9_000_000)
                rows.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "amount": close * volume,
                    }
                )
                prev = close
            return pd.DataFrame(rows)

        minute_step = int(self.INTERVAL_MAP_AK.get(query.interval, "1"))
        if start_ts is not None:
            days = pd.bdate_range(start=start_ts.normalize(), end=end_ts.normalize())
        else:
            days = pd.bdate_range(end=end_ts.normalize(), periods=max(1, min(int(query.period_days or 5), 10)))
        intraday_times: List[pd.Timestamp] = []
        for day in days:
            morning = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=11, minutes=30), freq=f"{minute_step}min")
            afternoon = pd.date_range(day + pd.Timedelta(hours=13), day + pd.Timedelta(hours=15), freq=f"{minute_step}min")
            intraday_times.extend(list(morning) + list(afternoon))
        rows = []
        prev = _price(0)
        for idx, ts in enumerate(intraday_times):
            close = _price(idx) * (1.0 + math.sin(idx / 11.0) * 0.001)
            open_ = prev
            high = max(open_, close) * 1.0008
            low = min(open_, close) * 0.9992
            volume = int(5_000_000 + (seed_int % 100) * 1000 + (idx % 31) * 80_000)
            rows.append(
                {
                    "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": ts.strftime("%Y-%m-%d"),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": close * volume,
                    "fallback_errors": " | ".join(errors or []),
                }
            )
            prev = close
        return pd.DataFrame(rows)

    def _fetch_from_akshare(self, query: MarketDataQuery) -> pd.DataFrame:
        if ak is None:
            raise ImportError("akshare is not installed")

        if query.interval == "1d":
            if self._looks_index_symbol(query.symbol):
                raw = ak.stock_zh_index_daily(symbol=query.symbol)
            else:
                raw = ak.stock_zh_a_hist(
                    symbol=self._strip_exchange_prefix(query.symbol),
                    period="daily",
                    start_date=self._to_compact_date(query.start),
                    end_date=self._to_compact_date(query.end),
                    adjust=query.adjust,
                )
        else:
            raw = ak.stock_zh_a_hist_min_em(
                symbol=self._strip_exchange_prefix(query.symbol),
                start_date=self._to_ak_datetime(query.start),
                end_date=self._to_ak_datetime(query.end),
                period=self.INTERVAL_MAP_AK[query.interval],
                adjust=query.adjust,
            )

        if raw is None or raw.empty:
            raise ValueError("empty response from akshare")
        return raw

    def _fetch_from_yfinance(self, query: MarketDataQuery) -> pd.DataFrame:
        if yf is None:
            raise ImportError("yfinance is not installed")

        ticker = self._to_yf_ticker(query.symbol, query.market)
        start = query.start
        if not start and query.period_days > 0:
            start = (datetime.utcnow() - timedelta(days=max(query.period_days * 2, 30))).strftime("%Y-%m-%d")
        end = query.end

        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=query.interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise ValueError("empty response from yfinance")

        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        return raw

    def _fetch_from_ashare(self, query: MarketDataQuery) -> pd.DataFrame:
        if ashare_get_price is None:
            raise ImportError("Ashare is not installed")

        kwargs: Dict[str, object] = {}
        signature = inspect.signature(ashare_get_price)
        if "frequency" in signature.parameters:
            kwargs["frequency"] = query.interval
        elif "freq" in signature.parameters:
            kwargs["freq"] = query.interval

        if query.end and "end_date" in signature.parameters:
            kwargs["end_date"] = query.end
        if query.start and "start_date" in signature.parameters:
            kwargs["start_date"] = query.start
        if "count" in signature.parameters and not query.start:
            kwargs["count"] = max(query.period_days, 1)

        raw = ashare_get_price(query.symbol, **kwargs)
        if raw is None:
            raise ValueError("empty response from Ashare")
        if isinstance(raw, pd.Series):
            raw = raw.to_frame().T
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("Ashare returned unsupported type")
        if raw.empty:
            raise ValueError("empty response from Ashare")
        return raw

    def _normalize(self, frame: pd.DataFrame, query: MarketDataQuery, provider: str) -> pd.DataFrame:
        df = frame.copy()
        if df.empty:
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)

        mapping = {
            "日期": "date",
            "时间": "datetime",
            "datetime": "datetime",
            "Datetime": "datetime",
            "Date": "date",
            "date": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume",
        }
        df.rename(columns=mapping, inplace=True)

        if "datetime" not in df.columns and "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "datetime"})
            else:
                raise ValueError("missing datetime/date column")
        if "datetime" not in df.columns and "date" in df.columns:
            df["datetime"] = df["date"]
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")

        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in df.columns:
                if col == "amount":
                    df[col] = pd.NA
                elif col == "volume":
                    df[col] = 0.0
                else:
                    df[col] = pd.NA
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["datetime", "open", "high", "low", "close"])

        if df["amount"].isna().all():
            df["amount"] = df["close"] * df["volume"].fillna(0.0)

        df["symbol"] = query.symbol
        df["interval"] = query.interval
        df["provider"] = provider
        df["adjust"] = query.adjust
        df["market"] = query.market.upper()

        df = df.sort_values("datetime").reset_index(drop=True)
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df[self.STANDARD_COLUMNS]

    def _align_cn_calendar(self, frame: pd.DataFrame, interval: str) -> pd.DataFrame:
        if frame.empty:
            return frame

        df = frame.copy()
        if interval == "1d":
            dates = pd.to_datetime(df["date"], errors="coerce")
            calendar = set(self._get_cn_trading_days(dates.min(), dates.max()))
            if calendar:
                df = df[df["date"].isin(calendar)]
        else:
            ts = pd.to_datetime(df["datetime"], errors="coerce")
            minutes = ts.dt.hour * 60 + ts.dt.minute
            weekday_ok = ts.dt.weekday <= 4
            morning = (minutes >= 9 * 60 + 30) & (minutes <= 11 * 60 + 30)
            afternoon = (minutes >= 13 * 60) & (minutes <= 15 * 60)
            df = df[weekday_ok & (morning | afternoon)]
        return df.reset_index(drop=True)

    def _get_cn_trading_days(self, start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
        if pd.isna(start) or pd.isna(end):
            return []

        if self._cn_trade_calendar is None:
            if ak is not None:
                try:
                    raw = ak.tool_trade_date_hist_sina()
                    col = "trade_date" if "trade_date" in raw.columns else raw.columns[0]
                    self._cn_trade_calendar = pd.to_datetime(raw[col], errors="coerce").dropna()
                except Exception:
                    self._cn_trade_calendar = pd.Series(dtype="datetime64[ns]")
            else:
                self._cn_trade_calendar = pd.Series(dtype="datetime64[ns]")

        if self._cn_trade_calendar.empty:
            bdays = pd.bdate_range(start=start, end=end)
            return [d.strftime("%Y-%m-%d") for d in bdays]

        mask = (self._cn_trade_calendar >= start.normalize()) & (self._cn_trade_calendar <= end.normalize())
        return [d.strftime("%Y-%m-%d") for d in self._cn_trade_calendar.loc[mask].tolist()]

    @staticmethod
    def _strip_exchange_prefix(symbol: str) -> str:
        if symbol.startswith(("sh", "sz")) and len(symbol) >= 8:
            return symbol[2:]
        return symbol

    @staticmethod
    def _looks_index_symbol(symbol: str) -> bool:
        return symbol.startswith(("sh", "sz")) and len(symbol) == 8 and symbol[2] in {"0", "3"}

    @staticmethod
    def _to_compact_date(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return pd.to_datetime(value, errors="coerce").strftime("%Y%m%d")

    @staticmethod
    def _to_ak_datetime(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return pd.to_datetime(value, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _to_yf_ticker(symbol: str, market: str) -> str:
        if market.upper() != "CN":
            return symbol
        if symbol.startswith("sh"):
            return f"{symbol[2:]}.SS"
        if symbol.startswith("sz"):
            return f"{symbol[2:]}.SZ"
        if symbol.isdigit() and len(symbol) == 6:
            return f"{symbol}.SS" if symbol.startswith("6") else f"{symbol}.SZ"
        return symbol
