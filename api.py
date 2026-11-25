"""
Portfolio Compounding Simulator - API (FastAPI)

This is the "brain" of the app. It does everything the Streamlit app did behind the scenes
(reads etfs.csv, downloads prices from Yahoo Finance, runs the month-by-month math) and returns
plain JSON. The Vercel front-end (index.html) just sends it the user's inputs and draws the answer.

Run locally:   uvicorn api:app --reload
Run on Render: uvicorn api:app --host 0.0.0.0 --port $PORT

Needs etfs.csv in the same folder, with columns:  Ticker, ETF Name, Expense Ratio  (e.g. "0.0400%").
"""
import math
import os
import threading
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------------------------------------------
CSV_PATH = Path(__file__).parent / "etfs.csv"
FIRST_MONTH = pd.Timestamp(2006, 1, 1)      # earliest month the app supports
MAX_FUNDS = 10                              # most funds one portfolio can hold
PRICE_TTL_SECONDS = 24 * 60 * 60            # reuse downloaded prices for one day

# Which websites may call this API. "*" = anyone (fine for testing). For production set the
# ALLOWED_ORIGINS environment variable on Render to your Vercel address, e.g.
#   https://my-portfolio-app.vercel.app
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# --------------------------------------------------------------------------------------
# 2. LOAD THE ETF LIST (etfs.csv)
# --------------------------------------------------------------------------------------
def load_catalog():
    """Read etfs.csv into {ticker: {"name", "fee", "label"}}. fee is a decimal (0.0004) or None."""
    df = pd.read_csv(CSV_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = {"Ticker", "ETF Name", "Expense Ratio"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"etfs.csv is missing column(s): {', '.join(sorted(missing_cols))}")

    # Drop blank/footer rows, then keep only rows that look like real tickers.
    df = df.dropna(subset=["Ticker", "ETF Name"]).copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df[df["Ticker"].str.fullmatch(r"[A-Z0-9.\-]+") & (df["Ticker"] != "TICKER")]
    df = df.drop_duplicates(subset="Ticker")

    # "0.0400%" -> 0.0004. Unreadable values become None instead of crashing.
    fee_pct = pd.to_numeric(
        df["Expense Ratio"].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce"
    )

    catalog = {}
    for ticker, name, fee in zip(df["Ticker"], df["ETF Name"].astype(str).str.strip(), fee_pct):
        catalog[ticker] = {
            "name": name,
            "fee": None if pd.isna(fee) else float(fee) / 100.0,
            "label": f"{ticker} — {name}",
        }
    return catalog

CATALOG = load_catalog()   # loaded once when the server starts

# --------------------------------------------------------------------------------------
# 3. MONTH HELPERS
# --------------------------------------------------------------------------------------
def last_full_month():
    """The current month is incomplete (its price is still moving), so the last usable month is the one before."""
    this_month = pd.Timestamp.today().normalize().replace(day=1)
    return this_month - pd.DateOffset(months=1)

def month_labels():
    """('01/2006', '02/2006', ...) up to the last complete month, plus a label -> Timestamp lookup."""
    months = pd.date_range(FIRST_MONTH, last_full_month(), freq="MS")
    return [m.strftime("%m/%Y") for m in months], {m.strftime("%m/%Y"): m for m in months}

# --------------------------------------------------------------------------------------
# 4. PRICE DOWNLOAD WITH A SIMPLE ONE-DAY CACHE
# --------------------------------------------------------------------------------------
_price_cache = {}            # ticker -> (time fetched, pandas Series of month-end prices)
_price_lock = threading.Lock()

def fetch_closes(tickers):
    """Monthly dividend-adjusted prices (total returns) for each ticker, indexed by first day of month."""
    now = time.time()
    with _price_lock:
        stale = [t for t in tickers if t not in _price_cache or now - _price_cache[t][0] > PRICE_TTL_SECONDS]
        if stale:
            raw = yf.download(stale, start="2005-12-01", interval="1mo", auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                raise RuntimeError("Yahoo Finance returned no data.")

            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"].copy()
            else:
                closes = raw[["Close"]].rename(columns={"Close": stale[0]})

            # Put every bar on the first day of its month and drop duplicate bars for one month.
            closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).to_period("M").to_timestamp()
            closes = closes[~closes.index.duplicated(keep="last")].sort_index()

            for t in stale:
                if t in closes.columns and not closes[t].dropna().empty:
                    _price_cache[t] = (now, closes[t])

        return pd.DataFrame({t: _price_cache[t][1] for t in tickers if t in _price_cache})

# --------------------------------------------------------------------------------------
# 5. REQUEST SHAPE (what the front-end sends)
# --------------------------------------------------------------------------------------
class FundIn(BaseModel):
    ticker: str
    pct: float = Field(ge=0, le=100)           # target allocation, 0-100

class SimulationRequest(BaseModel):
    initial: float = Field(0, ge=0, le=10_000_000)   # starting lump sum ($)
    monthly: float = Field(0, ge=0, le=100_000)      # monthly contribution ($)
    start: str | None = None                         # "MM/YYYY"; None/earlier-than-possible = earliest possible
    end: str | None = None                           # "MM/YYYY"; None = last complete month
    rebalance: bool = True                           # reset to target % every December
    funds: list[FundIn] = Field(min_length=1, max_length=MAX_FUNDS)

def bad_request(message):
    return HTTPException(status_code=400, detail=message)

def num(x, digits=2):
    """JSON can't carry NaN/infinity, so turn those into null."""
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), digits)

# --------------------------------------------------------------------------------------
# 6. THE APP
# --------------------------------------------------------------------------------------
app = FastAPI(title="Portfolio Compounding Simulator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Tiny endpoint the front-end can ping to wake the server up."""
    return {"status": "ok"}

@app.get("/funds")
def list_funds():
    """The searchable fund list and the month choices for the From/To boxes."""
    labels, _ = month_labels()
    funds = [
        {"ticker": t, "name": info["name"], "label": info["label"], "expense_ratio": info["fee"]}
        for t, info in sorted(CATALOG.items())
    ]
    return {"funds": funds, "months": labels, "max_funds": MAX_FUNDS}

@app.post("/simulate")
def simulate(req: SimulationRequest):
    # ---------------- 6a. validate the inputs ----------------
    tickers_in = [f.ticker.strip().upper() for f in req.funds]

    if req.initial == 0 and req.monthly == 0:
        raise bad_request("Please enter at least $1 in either the initial investment or the monthly contribution.")

    seen = set()
    for t in tickers_in:
        if t not in CATALOG:
            raise bad_request(f"'{t}' is not in the supported fund list.")
        if t in seen:
            raise bad_request(f"Duplicate fund: {t} is listed more than once. Please pick a different ETF.")
        seen.add(t)

    total_pct = round(sum(f.pct for f in req.funds), 2)
    if abs(total_pct - 100.0) > 0.005:
        diff = round(abs(100.0 - total_pct), 2)
        if total_pct < 100:
            raise bad_request(f"Your allocation adds up to {total_pct}%. Please add {diff}% more to reach 100%.")
        raise bad_request(f"Your allocation adds up to {total_pct}%. Please reduce it by {diff}% to equal 100%.")

    # Only funds with more than 0% are used, kept in the order the user entered them.
    active_portfolio = {t: f.pct for t, f in zip(tickers_in, req.funds) if f.pct > 0}
    active_tickers = list(active_portfolio)

    labels, label_to_ts = month_labels()
    last_month = last_full_month()

    # ---------------- 6b. download prices and turn them into monthly returns ----------------
    try:
        prices = fetch_closes(sorted(active_tickers))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=("Couldn't download price history from Yahoo Finance. This can be a temporary outage or "
                    f"rate limiting. Please try again in a minute. ({e})"),
        )

    no_data = [t for t in active_tickers if t not in prices.columns]
    if no_data:
        raise HTTPException(
            status_code=404,
            detail=f"Yahoo Finance returned no price history for: {', '.join(no_data)}. Please pick a different fund.",
        )

    # Line prices up on every month from Dec 2005 to the last complete month. This also drops the
    # partial current month, and a missing month stays empty (NaN) instead of being silently skipped.
    price_months = pd.date_range("2005-12-01", last_month, freq="MS")
    prices = prices[active_tickers].reindex(price_months)
    returns_df = prices.pct_change(fill_method=None).loc[FIRST_MONTH:]

    # ---------------- 6c. work out the date range ----------------
    # Each fund's first month with a usable return. The portfolio can only start once ALL funds have
    # data, i.e. at the latest of these dates.
    first_valid = {t: returns_df[t].first_valid_index() for t in active_tickers}
    unusable = [t for t, d in first_valid.items() if d is None]
    if unusable:
        raise bad_request(f"{', '.join(unusable)} has no usable monthly returns. Please pick a different fund.")

    common_start = max(first_valid.values())

    start_requested = None
    if req.start:
        if req.start not in label_to_ts:
            raise bad_request(f"Invalid start month '{req.start}'. Use MM/YYYY between {labels[0]} and {labels[-1]}.")
        start_requested = label_to_ts[req.start]
    if req.end and req.end not in label_to_ts:
        raise bad_request(f"Invalid end month '{req.end}'. Use MM/YYYY between {labels[0]} and {labels[-1]}.")

    # Start at the earliest possible month unless the user chose a LATER one.
    start_ts = common_start if (start_requested is None or start_requested <= common_start) else start_requested
    end_ts = label_to_ts[req.end] if req.end else last_month

    if end_ts < common_start:
        raise bad_request(
            f"The end month ({end_ts:%m/%Y}) is before the earliest date all your funds have data "
            f"({common_start:%m/%Y}). Please choose a later end month."
        )
    if start_ts > end_ts:
        raise bad_request(f"Invalid date range: the start ({start_ts:%m/%Y}) cannot be after the end ({end_ts:%m/%Y}).")

    sliced_returns = returns_df.loc[start_ts:end_ts]
    gaps = [t for t in active_tickers if sliced_returns[t].isna().any()]
    if gaps:
        parts = []
        for t in gaps:
            first, last = returns_df[t].first_valid_index(), returns_df[t].last_valid_index()
            parts.append(f"{t} has data only from {first:%m/%Y} to {last:%m/%Y}")
        raise bad_request("Not enough price history for the selected date range: " + "; ".join(parts) + ".")

    # Tell the user why the portfolio doesn't start in 01/2006.
    notices = []
    if common_start > FIRST_MONTH and start_ts == common_start:
        late = sorted(((d, t) for t, d in first_valid.items() if d > FIRST_MONTH), reverse=True)
        binding_date, binding_ticker = late[0]
        note = (f"This portfolio starts in {binding_date:%m/%Y} instead of 01/2006 because {binding_ticker} "
                "only has monthly return data from that month.")
        others = [f"{t} ({d:%m/%Y})" for d, t in late[1:]]
        if others:
            note += f" Other funds that also start after 01/2006: {', '.join(others)}."
        note += " Every fund is measured over the same period so the comparison stays fair."
        notices.append(note)

    # ---------------- 6d. blended expense ratio ----------------
    fees = {t: (CATALOG[t]["fee"] or 0.0) for t in active_tickers}   # unreadable fee counts as 0%
    missing_fee = [t for t in active_tickers if CATALOG[t]["fee"] is None]
    weighted_fee = sum((pct / 100.0) * fees[t] for t, pct in active_portfolio.items())
    if missing_fee:
        notices.append(f"No readable expense ratio in etfs.csv for {', '.join(missing_fee)}; treated as 0%.")

    # ---------------- 6e. MONTH-BY-MONTH COMPOUNDING ENGINE ----------------
    # Two parallel sets of balances, one per fund:
    #   NET   = the real after-fee history (Yahoo prices already have each fund's fees taken out).
    #   GROSS = a hypothetical zero-fee version (net return + that month's share of the expense ratio).
    # The gap between GROSS and NET is the "Est. Fee Drag".
    #
    # Every month, for every fund i:   balance_i = (balance_i + deposit_i) * (1 + return_i)
    # i.e. the deposit goes in at the START of the month, then the month's return is applied.
    target_weights = {t: pct / 100.0 for t, pct in active_portfolio.items()}
    net_balances = {t: req.initial * w for t, w in target_weights.items()}
    gross_balances = dict(net_balances)
    monthly_inputs = {t: req.monthly * w for t, w in target_weights.items()}

    total_invested = req.initial
    rebalance_count = 0
    monthly_portfolio_returns = []

    series = {
        "dates": [], "total_invested": [], "net_total": [], "gross_total": [],
        "fee_impact": [], "monthly_return": [], "rebalanced": [],
        "funds": {t: [] for t in active_tickers},
    }

    for date_ts, row in sliced_returns.iterrows():
        prev_net_total = sum(net_balances.values())
        total_invested += req.monthly

        for t in active_tickers:
            net_r = float(row[t])                      # the fund's actual return this month
            gross_r = net_r + fees[t] / 12.0           # add one month of the expense ratio back
            net_balances[t] = (net_balances[t] + monthly_inputs[t]) * (1 + net_r)
            gross_balances[t] = (gross_balances[t] + monthly_inputs[t]) * (1 + gross_r)

        net_total = sum(net_balances.values())
        gross_total = sum(gross_balances.values())

        # Portfolio return for the month = ending value / (starting value + this month's deposit) - 1.
        # Dividing by the deposit too keeps new contributions from looking like gains.
        start_base = prev_net_total + req.monthly
        port_ret = net_total / start_base - 1 if start_base > 0 else float("nan")
        monthly_portfolio_returns.append(port_ret)

        # ANNUAL REBALANCING (end of December): reset every fund to its target share of the CURRENT total:
        #     new_balance_i = total_portfolio_value x target_weight_i
        # Net and gross are rebalanced separately, each to its own total. The total never changes -
        # only the split between funds. With one fund there is nothing to rebalance.
        did_rebalance = False
        if req.rebalance and date_ts.month == 12 and len(active_tickers) > 1:
            for t in active_tickers:
                net_balances[t] = net_total * target_weights[t]
                gross_balances[t] = gross_total * target_weights[t]
            did_rebalance = True
            rebalance_count += 1

        series["dates"].append(date_ts.strftime("%Y-%m-01"))
        series["total_invested"].append(num(total_invested))
        series["net_total"].append(num(net_total))
        series["gross_total"].append(num(gross_total))
        series["fee_impact"].append(num(max(0.0, gross_total - net_total)))
        series["monthly_return"].append(num(port_ret, 6))
        series["rebalanced"].append(did_rebalance)
        for t in active_tickers:
            series["funds"][t].append(num(net_balances[t]))

    # ---------------- 6f. summary numbers ----------------
    final_value = sum(net_balances.values())
    total_gain = final_value - total_invested

    # Annualized volatility = standard deviation of the monthly returns x sqrt(12). Needs 2+ months.
    returns_clean = pd.Series(monthly_portfolio_returns).dropna()
    annualized_vol = returns_clean.std() * math.sqrt(12) if len(returns_clean) >= 2 else None

    # "Growth of $1": what $1 would have become using the same monthly portfolio returns. Each month's
    # return already adjusts for deposits, so this curve is not distorted by when you contributed.
    # (position 0 is the starting point; position k is the end of the k-th month)
    n_months = len(returns_clean)
    growth_curve = pd.concat([pd.Series([1.0]), (1 + returns_clean).cumprod()], ignore_index=True)

    # Annualized growth rate = (final growth of $1) ^ (12 / months) - 1, shown only for 12+ months
    # because stretching a few months into a yearly rate is misleading.
    annual_growth = growth_curve.iloc[-1] ** (12 / n_months) - 1 if n_months >= 12 else None

    # Biggest drop (maximum drawdown) = largest fall from a previous high to a later low, measured
    # on month-end values: drawdown = growth_curve / highest value so far - 1.
    drawdowns = growth_curve / growth_curve.cummax() - 1
    trough_pos = int(drawdowns.idxmin())
    max_drawdown = float(drawdowns.iloc[trough_pos])                       # 0 or negative
    peak_pos = int(growth_curve.iloc[: trough_pos + 1][::-1].idxmax())     # the high just before the low

    def drawdown_label(pos):
        """'start' for the starting point, otherwise the month as MM/YYYY."""
        if pos == 0:
            return "start"
        d = series["dates"][pos - 1]          # "YYYY-MM-01"
        return f"{d[5:7]}/{d[:4]}"

    has_drop = max_drawdown <= -0.0005 and n_months == len(series["dates"])

    return {
        "period": {
            "start": start_ts.strftime("%m/%Y"),
            "end": end_ts.strftime("%m/%Y"),
            "months": len(series["dates"]),
        },
        "notices": notices,
        "kpis": {
            "total_invested": num(total_invested),
            "ending_value": num(final_value),
            "total_gain": num(total_gain),
            "gain_pct": num(total_gain / total_invested * 100, 1),   # total_invested is always > 0 here
            "fee_drag": num(series["fee_impact"][-1]),
            "avg_fee_pct": num(weighted_fee * 100, 4),
            "annualized_vol_pct": num(annualized_vol * 100, 2) if annualized_vol is not None else None,
            "vol_months": int(len(returns_clean)),
            "annualized_growth_pct": num(annual_growth * 100, 2) if annual_growth is not None else None,
            "max_drawdown_pct": num(max_drawdown * 100, 2),                # 0 or negative
            "drawdown_peak": drawdown_label(peak_pos) if has_drop else None,
            "drawdown_trough": drawdown_label(trough_pos) if has_drop else None,
            "rebalance_on": bool(req.rebalance),
            "rebalance_count": rebalance_count,
        },
        "series": series,
        "ending_mix": {t: num(net_balances[t]) for t in active_tickers},
        "tickers": active_tickers,
    }
