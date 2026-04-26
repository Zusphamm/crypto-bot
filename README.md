# Multi-Timeframe Price Analyzer

Công cụ phân tích xu hướng giá đa khung thời gian cho crypto, **triển khai trực tiếp từ file `advanced_price_direction_algorithms.md`**.

Gồm **2 tool**:

1. **`binance_fetcher.py`** — Kéo dữ liệu OHLCV từ Binance API (khung M1 → H12, tối đa 10.000 nến, lookback tối đa 5 năm).
2. **Multi-timeframe analyzer** — Phân tích xu hướng từng khung + xác suất empirical dựa trên backtest ~2000+ điểm lịch sử.

---

## 📦 Cài đặt

```bash
pip install pandas numpy requests scipy hmmlearn pykalman
```

Python ≥ 3.9.

---

## 🚀 Cách dùng nhanh nhất

### 1. Chạy full pipeline (tất cả 11 khung, 10k nến)

```bash
python main.py BTCUSDT 10000 spot
```

### 2. Quick demo (5 khung, 3k nến — ~30 giây)

```bash
python quick_demo.py
python quick_demo.py ETHUSDT 5000
```

### 3. Dùng như library

```python
from binance_fetcher import fetch_all_timeframes
from multi_timeframe import multi_timeframe_analysis, print_report

# Kéo data
data = fetch_all_timeframes("BTCUSDT", max_candles=10000, max_years=5)

# Phân tích
result = multi_timeframe_analysis(data, run_backtest=True)

# In báo cáo
print_report(result)

# Hoặc lấy xác suất raw
probs = result["overall"]["weighted_probabilities"]
print(f"P(UP) = {probs['up']*100:.1f}%")
print(f"P(DOWN) = {probs['down']*100:.1f}%")
```

---

## 📂 Kiến trúc module

| File | Vai trò | File md (section) |
|------|---------|-------------------|
| `binance_fetcher.py` | Kéo OHLCV từ Binance, paginate 1000 nến/request → tối đa 10k nến | — |
| `indicators.py` | MA/RSI/MACD/BBands/ATR + Ichimoku + Market Structure (HH/HL/LH/LL + BOS) | §1.2 |
| `advanced_algorithms.py` | HMM regime, Kalman trend, Hurst exponent | §7, §8, §9 |
| `single_timeframe.py` | Ensemble 9 model → xác suất {up, down, neutral} cho 1 khung | §12 Layer 3 |
| `backtest.py` | Vectorized backtest trên toàn lịch sử → empirical hit rate | §14 Walk-Forward |
| `multi_timeframe.py` | Orchestrator đa khung + Top-Down confluence | §1.1, §12.2 |
| `main.py` | CLI entry point | — |
| `quick_demo.py` | Chạy nhanh với 5 khung | — |
| `test_pipeline.py` | End-to-end test với synthetic data | — |

---

## 🧠 Cách hoạt động

### Tool 1 — Binance fetcher

Binance giới hạn 1000 nến / request. Tool auto-paginate theo `startTime` để lấy đến **10.000 nến**, đồng thời giới hạn lookback tối đa **5 năm** (lấy `max(earliest, candles_limit)` → không vượt cả 2 ràng buộc).

```python
df = fetch_klines(
    symbol="BTCUSDT",
    interval="1h",
    max_candles=10_000,    # cap 10k
    max_years=5,           # cap 5 năm
    market="spot",         # hoặc "futures"
)
# → DataFrame với index UTC, cột: open, high, low, close, volume, ...
```

**Khung hỗ trợ:** `1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h`.

### Tool 2 — Phân tích xu hướng đa khung

**Bước 1 — Single-timeframe analyzer (`analyze_timeframe`)**

Với mỗi khung, chạy ensemble 9 model:

| Model | Vote | Weight | Nguồn (file md) |
|-------|------|--------|-----------------|
| MA Alignment | ±1/0 | 2.0 (strong) / 1.2 (normal) | §1.2 |
| Market Structure (HH/HL/LH/LL + BOS) | ±1/0 | 1.5 + bonus BOS 0.8 | §1.2 |
| Ichimoku cloud | ±1/0 | 1.5 / 0.5 | §1.2 |
| RSI(14) | ±1/0 | 0.5-1.0 | classical |
| MACD histogram | ±1 | 1.5 (cross) / 0.8 | classical |
| Bollinger position | ±1/0 | 0.3-0.5 | classical |
| **HMM 3-state (Bull/Sideway/Bear)** | ±1/0 | 2.0 × posterior | **§7** |
| **Kalman velocity** | ±1/0 | 0.5 + 50× strength | **§9** |
| **Hurst exponent (R/S)** | ±1/0 | 0.1-1.5 | **§8** |

Xác suất cuối = weighted sum các vote → `{up, down, neutral}`.

**Bước 2 — Empirical backtest (`compute_empirical_hit_rate`)**

Đây là phần bạn yêu cầu: **"đưa mô hình vào càng nhiều điểm mua bán trong quá khứ"**.

Với mỗi bar trong lịch sử:
1. Tính ensemble score từ 5 vectorized indicators (MA, MACD, RSI, BB, momentum)
2. Gán nhãn dự đoán: UP / DOWN / NEUTRAL
3. Đo forward return sau N nến → hướng thực tế
4. So sánh → hit/miss

**Output:**
```python
{
    "n_samples": 2545,              # số điểm đã backtest
    "hit_rate_up": 0.608,           # khi model nói UP, giá thực sự lên 60.8% lần
    "hit_rate_down": 0.573,
    "hit_rate_overall": 0.591,
    "avg_forward_return_on_up": 0.011,
    "n_up_signals": 1043,
    "n_down_signals": 759,
    ...
}
```

**Default forward_bars theo khung** (để so sánh fair):

| Khung | Forward bars | Tương đương |
|-------|-------------|-------------|
| 1m | 15 | 15 phút |
| 5m | 12 | 1 giờ |
| 15m | 8 | 2 giờ |
| 1h | 12 | 12 giờ |
| 4h | 6 | 1 ngày |
| 12h | 4 | 2 ngày |

**Bước 3 — Bayesian calibration (`bayesian_adjusted_probability`)**

Lấy ensemble vote hiện tại + hit_rate lịch sử → xác suất calibrated:

```
P(giá thật sự lên | model nói UP) = hit_rate_up
```

Nếu ensemble nói UP nhưng hit_rate_up chỉ 48% → model **không có edge** trên khung đó.

**Bước 4 — Multi-timeframe confluence (`multi_timeframe_analysis`)**

Áp dụng triết lý **Top-Down** (file md §1.1):

- **Khung lớn** (4h, 6h, 8h, 12h) = **context** (xu hướng)
- **Khung trung** (1h, 2h) = **setup** (xác nhận)
- **Khung nhỏ** (1m-30m) = **trigger** (entry timing)

Trọng số khung lớn > khung nhỏ:

| Khung | Weight |
|-------|--------|
| 1m | 0.5 |
| 15m | 0.9 |
| 1h | 1.3 |
| 4h | 1.8 |
| 12h | 2.3 |

**High-quality setup** khi **cả 3 nhóm cùng hướng** (file md §12.2: "trade only khi ≥ 2/3 layer agree").

---

## 📊 Ví dụ output

```
[  4h] strong_up | up=75.8% dn=10.7% nu=13.5% | hit=56.5% base=48.2% edge= +8.3% ✓ (n=2545)
```

**Cách đọc:**

| Phần | Ý nghĩa |
|------|---------|
| `strong_up` | Ensemble 9 model hiện tại nói xu hướng tăng mạnh |
| `up=75.8%` | Xác suất UP từ weighted vote hiện tại |
| `hit=56.5%` | Trong 2545 lần model nói UP trong quá khứ, giá thực sự lên 56.5% lần |
| `base=48.2%` | Baseline — tỉ lệ UP tự nhiên trên khung này (không cần model) |
| `edge= +8.3%` | Model tốt hơn baseline +8.3% → **có skill thực sự** |
| `✓` | Có edge (≥ +2%) — có thể trade |
| `✗` | Không có edge — bỏ qua, model chỉ đoán bằng hoặc tệ hơn random |

**Quy tắc vàng:** chỉ tin tín hiệu khi có `✓`. `hit=60%` nghe cao nhưng nếu `base=65%` thì `edge=-5%` → model còn thua buy-and-hold.

Cuối báo cáo:
```
================================================================
                   MULTI-TIMEFRAME TREND REPORT
================================================================

  📈 Overall direction : UP
  📊 Overall prob     : UP=54.5%  DOWN=35.8%  NEUTRAL=9.7%

  Higher TF (4h-12h) : up  (agree=75%)
  Middle TF (1h-2h)  : up  (agree=100%)
  Lower  TF (1m-30m) : up  (agree=60%)

  High quality setup : ✅
  Khung có edge > baseline: 8/11
  💬 🟢 HIGH QUALITY UP SETUP — cả 3 nhóm khung đồng thuận.
```

**Nếu `Khung có edge > baseline: 0/11`** → tool sẽ override thành:
```
⛔ NO-EDGE — không khung nào có hit_rate > baseline. KHÔNG TRADE.
```

---

## ⚠️ Lưu ý quan trọng

### Binance IP restriction

Một số cloud provider / quốc gia bị Binance chặn (403 Forbidden). Nếu gặp lỗi:

```
RuntimeError: Không thể kết nối Binance sau 3 lần thử.
Nguyên nhân có thể: (1) IP bị Binance chặn...
```

**Giải pháp:**
- Chạy trên máy local (Việt Nam OK)
- Dùng VPN
- Hoặc thay endpoint bằng exchange khác (cần sửa `binance_fetcher.py`)

### Đây KHÔNG phải tư vấn đầu tư

File md gốc ghi rõ: *"Chỉ dùng cho mục đích nghiên cứu và giáo dục. Không phải tư vấn đầu tư tài chính."* Tool này cũng vậy.

Hit rate 55-60% **KHÔNG đảm bảo** bạn sẽ có lãi — còn phải tính:
- Risk management (position size, stop-loss)
- Trading fees + slippage
- Walk-forward validation (file md §14.2) trước khi dùng tiền thật
- Paper trading phase (file md §13.7 Phase 6)

### Những gì file md nói SOTA nhưng chưa implement ở đây

File md đề cập các model state-of-the-art 2024-2026 **cần GPU + training data lớn**:
- Mamba / MambaTS (§2)
- Temporal Fusion Transformer (§5)
- Hierarchical Multi-Agent DRL (§6)
- Diffusion models (§10)
- LLM-augmented (§11)

Những model này cần setup riêng (PyTorch + GPU + pipeline training). Tool hiện tại triển khai **tầng classical + HMM/Kalman/Hurst** — đã đủ mạnh cho đa số use case và **chạy real-time trên CPU**.

Nếu bạn muốn mở rộng thêm các model deep learning, hãy xem §13 "Implementation Roadmap" trong file md.

---

## 🧪 Test không cần Binance

```bash
python test_pipeline.py
```

Sinh synthetic data giống BTC với regime shifts → chạy full pipeline. Dùng để verify logic khi không truy cập được Binance.

---

## 🔧 CLI reference

```
python main.py [SYMBOL] [CANDLES] [MARKET] [options]

Positional:
  SYMBOL       Trading pair (default: BTCUSDT)
  CANDLES      Max candles per TF (default: 10000, cap: 10000)
  MARKET       spot | futures (default: spot)

Options:
  --years N              Max lookback years (default: 5)
  --intervals 1h 4h 12h  Custom timeframes
  --no-backtest          Skip empirical backtest (chạy nhanh hơn)
  --save output.json     Lưu kết quả chi tiết ra JSON
```

**Ví dụ:**
```bash
# Chỉ phân tích khung lớn (swing trading)
python main.py BTCUSDT 10000 spot --intervals 1h 4h 12h --save btc_report.json

# Futures, 5000 nến, nhanh (không backtest)
python main.py ETHUSDT 5000 futures --no-backtest
```
