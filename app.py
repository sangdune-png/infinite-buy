import math
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

st.set_page_config(
    page_title="라오어 무한매수법 v2.2 AI 백테스터",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 1. 자연어 파싱용 데이터 스키마 (Pydantic)
# ==========================================
class BacktestParams(BaseModel):
    ticker: str = Field(default="TQQQ", description="미국 주식/ETF 티커 (예: TQQQ, SOXL, UPRO)")
    start_date: str = Field(default="2020-01-01", description="시작일 YYYY-MM-DD")
    end_date: str = Field(default="2024-12-31", description="종료일 YYYY-MM-DD")
    initial_cash: float = Field(default=10000.0, description="초기 원금(달러). 원화 언급 시 1달러=1350원 환산")
    split_count: int = Field(default=40, description="분할 횟수 (v2.2 기본 40)")
    use_quarter_cut: bool = Field(default=True, description="원금 소진 시 쿼터 손절 모드 적용 여부")
    apply_tax: bool = Field(default=True, description="연말 양도소득세 22% 정산 반영 여부")
    fee_rate_pct: float = Field(default=0.07, description="편도 수수료+슬리피지(%)")
    parsed_note: str = Field(default="", description="해석한 조건을 한국어 한 줄로 요약")


def parse_user_prompt(api_key: str, user_text: str) -> BacktestParams:
    client = genai.Client(api_key=api_key)
    today_str = datetime.now().strftime("%Y-%m-%d")

    system_instruction = f"""
    너는 라오어 무한매수법 v2.2 백테스트 파라미터 추출기다.
    오늘 날짜는 {today_str}이다.
    사용자의 자연어 요청에서 백테스트 실행에 필요한 파라미터를 정확히 추출해 JSON으로 반환하라.
    - 언급되지 않은 항목은 기본값을 사용하라.
    - '작년', '최근 3년', '코로나 때' 같은 상대적 기간은 오늘 날짜({today_str}) 기준으로 YYYY-MM-DD로 변환하라.
    - 사용자가 원화(예: 1천만원, 5000만원)로 말하면 1달러=1350원 기준으로 initial_cash(달러)로 환산하라.
    - '존버', '쿼터 손절 빼고'라고 하면 use_quarter_cut을 false로 설정하라.
    - '세전', '양도세 제외'라고 하면 apply_tax를 false로 설정하라.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=BacktestParams,
            temperature=0.0,
        ),
    )
    return BacktestParams.model_validate_json(response.text)


# ==========================================
# 2. 주가 데이터 로드 (캐시 적용)
# ==========================================
@st.cache_data(ttl=3600)
def load_ohlc(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    end_dt_plus = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker.upper(), start=start_date, end=end_dt_plus, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df


# ==========================================
# 3. 무한매수법 v2.2 백테스트 엔진
# ==========================================
def run_infinite_buy_v22(df: pd.DataFrame, params: BacktestParams):
    initial_cash = float(params.initial_cash)
    split_count = max(1, int(params.split_count))
    fee_rate = float(params.fee_rate_pct) / 100.0
    tax_rate = 0.22
    tax_deduction_usd = 1850.0  # 연 250만 원 공제 기준

    years = df.index.year.to_numpy()
    is_year_end_arr = np.append(years[:-1] != years[1:], True)

    cash = initial_cash
    shares = 0
    avg_price = 0.0
    invested = 0.0
    unit_amt = initial_cash / split_count

    mode = "NORMAL"
    q_unit = 0.0
    q_count = 0

    yearly_pnl = 0.0
    total_tax_paid = 0.0
    cycle_wins = 0
    quarter_triggers = 0

    equity_curve = []
    trade_logs = []

    def execute_buy(date, qty, fill_p, reason):
        nonlocal cash, shares, avg_price, invested
        if qty <= 0:
            return 0
        max_affordable = int(cash // (fill_p * (1 + fee_rate)))
        actual_qty = min(qty, max_affordable)
        if actual_qty <= 0:
            return 0

        cost = actual_qty * fill_p * (1 + fee_rate)
        cash -= cost
        invested += actual_qty * fill_p
        shares += actual_qty
        avg_price = invested / shares

        trade_logs.append({
            "날짜": date.strftime("%Y-%m-%d"),
            "구분": "매수",
            "사유": reason,
            "체결가($)": round(fill_p, 2),
            "수량": actual_qty,
            "보유주수": shares,
            "평단가($)": round(avg_price, 2),
            "잔여현금($)": round(cash, 2)
        })
        return actual_qty

    def execute_sell(date, qty, fill_p, reason):
        nonlocal cash, shares, avg_price, invested, yearly_pnl
        actual_qty = min(qty, shares)
        if actual_qty <= 0:
            return 0

        proceeds = actual_qty * fill_p * (1 - fee_rate)
        buy_cost_with_fee = actual_qty * avg_price * (1 + fee_rate)
        pnl = proceeds - buy_cost_with_fee

        cash += proceeds
        yearly_pnl += pnl
        shares -= actual_qty
        invested -= actual_qty * avg_price

        if shares == 0:
            avg_price = 0.0
            invested = 0.0

        trade_logs.append({
            "날짜": date.strftime("%Y-%m-%d"),
            "구분": "매도",
            "사유": reason,
            "체결가($)": round(fill_p, 2),
            "수량": actual_qty,
            "보유주수": shares,
            "평단가($)": round(avg_price, 2),
            "잔여현금($)": round(cash, 2)
        })
        return actual_qty

    for i, (date, row) in enumerate(df.iterrows()):
        open_p = float(row["Open"])
        high_p = float(row["High"])
        close_p = float(row["Close"])
        is_year_end = bool(is_year_end_arr[i])

        if shares == 0:
            mode = "NORMAL"
            q_count = 0
            unit_amt = cash / split_count
            first_qty = int(unit_amt // (close_p * (1 + fee_rate)))
            execute_buy(date, first_qty, close_p, "사이클 시작(T=0)")
        else:
            raw_t = round(invested / unit_amt, 6)
            T = math.ceil(raw_t * 100) / 100.0
            half_split = split_count / 2.0

            need_quarter_moc = False
            if params.use_quarter_cut:
                if mode == "NORMAL":
                    if T > (split_count - 1.0) or (T >= half_split and cash < unit_amt * 0.5):
                        need_quarter_moc = True
                elif mode == "QUARTER":
                    if q_count >= 10 or cash < q_unit * 0.5:
                        need_quarter_moc = True

            if need_quarter_moc:
                quarter_triggers += 1
                moc_qty = max(1, shares // 4) if shares >= 2 else shares
                execute_sell(date, moc_qty, close_p, "쿼터 손절(1/4 MOC)")
                if shares == 0:
                    mode = "NORMAL"
                    q_count = 0
                else:
                    mode = "QUARTER"
                    q_unit = cash / 10.0
                    q_count = 0
            else:
                if mode == "NORMAL":
                    star_pct = 10.0 - (T / (split_count / 20.0))
                    star_p = round(avg_price * (1 + star_pct / 100.0), 2)
                else:
                    star_p = round(avg_price * 0.90, 2)

                loc_buy_p = round(star_p - 0.01, 2)
                limit_sell_p = round(avg_price * 1.10, 2)

                loc_sell_qty = shares // 4
                limit_sell_qty = shares - loc_sell_qty
                sold_any_today = False

                if limit_sell_qty > 0 and high_p >= limit_sell_p:
                    fill_p = open_p if open_p >= limit_sell_p else limit_sell_p
                    if execute_sell(date, limit_sell_qty, fill_p, f"{mode} 지정가(+10%)") > 0:
                        sold_any_today = True

                if loc_sell_qty > 0 and close_p >= star_p:
                    if execute_sell(date, loc_sell_qty, close_p, f"{mode} LOC매도(별%)") > 0:
                        sold_any_today = True

                if shares == 0:
                    cycle_wins += 1
                    mode = "NORMAL"
                    q_count = 0
                else:
                    if mode == "QUARTER" and sold_any_today:
                        mode = "NORMAL"
                        q_count = 0
                    else:
                        if mode == "NORMAL":
                            if T < half_split:
                                half_unit = unit_amt * 0.5
                                avg_order_p = round(avg_price, 2)
                                if close_p <= avg_order_p:
                                    qty1 = int(half_unit // (avg_order_p * (1 + fee_rate)))
                                    execute_buy(date, qty1, close_p, "전반전 평단 LOC")
                                if close_p <= loc_buy_p:
                                    qty2 = int(half_unit // (loc_buy_p * (1 + fee_rate)))
                                    execute_buy(date, qty2, close_p, "전반전 별% LOC")
                            else:
                                if close_p <= loc_buy_p:
                                    qty = int(unit_amt // (loc_buy_p * (1 + fee_rate)))
                                    execute_buy(date, qty, close_p, "후반전 별% LOC")
                        else:
                            if close_p <= loc_buy_p:
                                qty = int(q_unit // (loc_buy_p * (1 + fee_rate)))
                                if execute_buy(date, qty, close_p, "쿼터모드 -10% LOC") > 0:
                                    q_count += 1

        if params.apply_tax and is_year_end:
            if yearly_pnl > tax_deduction_usd:
                tax = (yearly_pnl - tax_deduction_usd) * tax_rate
                if cash < tax and shares > 0:
                    shortfall = tax - cash
                    sell_for_tax = min(shares, math.ceil(shortfall / (close_p * (1 - fee_rate))))
                    execute_sell(date, sell_for_tax, close_p, "양도세 납부용 매도")
                cash -= tax
                total_tax_paid += tax
                trade_logs.append({
                    "날짜": date.strftime("%Y-%m-%d"),
                    "구분": "세금",
                    "사유": "연말 양도세(22%)",
                    "체결가($)": 0.0,
                    "수량": 0,
                    "보유주수": shares,
                    "평단가($)": round(avg_price, 2),
                    "잔여현금($)": round(cash, 2)
                })
            yearly_pnl = 0.0

        equity_curve.append({
            "Date": date,
            "Equity": cash + shares * close_p,
            "Close": close_p
        })

    res = pd.DataFrame(equity_curve).set_index("Date")
    bh_shares = initial_cash / (df["Close"].iloc[0] * (1 + fee_rate))
    res["BH_Equity"] = bh_shares * res["Close"]

    years_elapsed = max((res.index[-1] - res.index[0]).days / 365.25, 0.01)

    def calc_stats(series):
        final_val = float(series.iloc[-1])
        total_ret = (final_val / initial_cash - 1) * 100
        cagr = ((final_val / initial_cash) ** (1 / years_elapsed) - 1) * 100
        dd = (series / series.cummax() - 1) * 100
        mdd = float(dd.min())
        return final_val, total_ret, cagr, mdd, dd

    v_final, v_ret, v_cagr, v_mdd, v_dd = calc_stats(res["Equity"])
    b_final, b_ret, b_cagr, b_mdd, b_dd = calc_stats(res["BH_Equity"])
    res["v22_DD"] = v_dd
    res["BH_DD"] = b_dd

    summary = {
        "actual_start": res.index[0].strftime("%Y-%m-%d"),
        "actual_end": res.index[-1].strftime("%Y-%m-%d"),
        "v_final": v_final, "v_ret": v_ret, "v_cagr": v_cagr, "v_mdd": v_mdd,
        "b_final": b_final, "b_ret": b_ret, "b_cagr": b_cagr, "b_mdd": b_mdd,
        "cycle_wins": cycle_wins,
        "quarter_triggers": quarter_triggers,
        "total_tax_paid": total_tax_paid
    }
    return res, pd.DataFrame(trade_logs), summary


# ==========================================
# 4. Plotly 차트 생성 (모바일 최적화)
# ==========================================
def build_chart(res: pd.DataFrame, ticker: str, summary: dict):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=[0.65, 0.35],
        subplot_titles=(f"{ticker} 자산 추이 (로그 스케일)", "낙폭 (Drawdown %)")
    )

    fig.add_trace(
        go.Scatter(
            x=res.index, y=res["Equity"],
            name=f"무한매수법 (CAGR {summary['v_cagr']:.1f}%)",
            line=dict(color="#1f77b4", width=2)
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=res.index, y=res["BH_Equity"],
            name=f"단순보유 (CAGR {summary['b_cagr']:.1f}%)",
            line=dict(color="#999999", width=1.5, dash="dot")
        ), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=res.index, y=res["v22_DD"],
            name="무한매수 DD", fill="tozeroy",
            line=dict(color="#1f77b4", width=1)
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=res.index, y=res["BH_DD"],
            name="단순보유 DD",
            line=dict(color="#d62728", width=1, dash="dot")
        ), row=2, col=1
    )

    fig.update_yaxes(type="log", title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        hovermode="x unified"
    )
    return fig


# ==========================================
# 5. Streamlit 채팅 UI
# ==========================================
st.title("📈 무한매수법 v2.2 AI 백테스터")
st.caption("말로 지시하면 Gemini가 조건을 해석하고 파이썬 엔진이 정확히 계산합니다.")

with st.sidebar:
    st.subheader("설정")
    secret_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = secret_key or st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**입력 예시**\n"
        "- TQQQ 2011년부터 2024년까지 1만달러 돌려줘\n"
        "- SOXL 2022년 폭락장 쿼터손절 빼고 세전으로 확인\n"
        "- TQQQ 최근 5년 30분할 3천만원 결과 보여줘"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_result(msg):
    params = msg["params"]
    s = msg["summary"]
    st.info(
        f"**적용 조건:** {params['ticker']} | {s['actual_start']}부터 {s['actual_end']}까지 | "
        f"원금 ${params['initial_cash']:,.0f} ({params['split_count']}분할) | "
        f"쿼터손절 {'ON' if params['use_quarter_cut'] else 'OFF'} | "
        f"양도세 {'ON' if params['apply_tax'] else 'OFF'}"
    )
    c1, c2 = st.columns(2)
    c1.metric("최종 평가액", f"${s['v_final']:,.0f}", f"누적 {s['v_ret']:+.1f}%")
    c2.metric("연복리 (CAGR)", f"{s['v_cagr']:.2f}%", f"단순보유 {s['b_cagr']:.2f}%")

    c3, c4 = st.columns(2)
    c3.metric("최대낙폭 (MDD)", f"{s['v_mdd']:.2f}%", f"단순보유 {s['b_mdd']:.2f}%")
    c4.metric("익절 / 쿼터손절", f"{s['cycle_wins']}회 / {s['quarter_triggers']}회", f"세금 ${s['total_tax_paid']:,.0f}")

    st.plotly_chart(build_chart(msg["res"], params["ticker"], s), use_container_width=True)

    with st.expander("최근 매매 로그 및 CSV 다운로드"):
        st.dataframe(msg["logs"].tail(30), use_container_width=True, hide_index=True)
        csv_data = msg["logs"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="전체 매매 로그 CSV 다운로드",
            data=csv_data,
            file_name=f"{params['ticker']}_v22_log.csv",
            mime="text/csv",
            key=msg["dl_key"]
        )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg)

if prompt := st.chat_input("예: TQQQ 2022년부터 2024년까지 쿼터손절 켜고 돌려줘"):
    if not api_key:
        st.error("사이드바 또는 Streamlit Secrets에 GEMINI_API_KEY를 먼저 등록해 주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("조건 해석 및 백테스트 계산 중..."):
            try:
                params = parse_user_prompt(api_key, prompt)
                df = load_ohlc(params.ticker, params.start_date, params.end_date)
                if df.empty or len(df) < 2:
                    st.error(f"'{params.ticker}'의 해당 기간 데이터를 불러오지 못했습니다. 티커나 날짜를 확인해 주세요.")
                else:
                    res, logs, summary = run_infinite_buy_v22(df, params)
                    result_payload = {
                        "role": "assistant",
                        "params": params.model_dump(),
                        "summary": summary,
                        "res": res,
                        "logs": logs,
                        "dl_key": f"dl_{len(st.session_state.messages)}"
                    }
                    render_result(result_payload)
                    st.session_state.messages.append(result_payload)
            except Exception as e:
                st.error(f"실행 중 오류가 발생했습니다: {e}")
