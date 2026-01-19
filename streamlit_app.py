import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. 기본 설정 및 유틸리티
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NEWS&NOW 플랫폼 트래픽 AI 대시보드", page_icon="📊", layout="wide")

@st.cache_data(ttl=300)
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 전처리: 컬럼명 정리 + 콤마 제거 및 숫자 변환"""
    df_clean = df.copy()

    # 컬럼명 앞뒤 공백 제거(키에러 방지)
    df_clean.columns = df_clean.columns.astype(str).str.strip()

    # 숫자 컬럼 처리
    for col in df_clean.columns:
        if col not in ["주차", "날짜", "Date"]:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
    return df_clean

def fmt_delta(curr, prev) -> str:
    """전주 대비 변화율 표시"""
    try:
        if prev is None:
            return "N/A"
        prev_val = float(prev)
        curr_val = float(curr)
        if prev_val == 0:
            return "N/A"
        pct = (curr_val - prev_val) / prev_val * 100
        return f"{pct:+.1f}%"
    except Exception:
        return "N/A"

# -----------------------------------------------------------------------------
# 2. 사이드바 (설정)
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1) 사이드바 헤더
    st.markdown("## ⚙️ 설정")
    st.caption("대시보드 구동을 위해 아래 입력이 필요합니다")

    # 2) 입력(필수) 카드: CSV URL
    with st.container(border=True):
        st.markdown("### 1) CSV URL (필수)")
        st.caption("지정된 플랫폼 트래픽 데이터 문서(CSV)를 입력합니다")

        csv_url = st.text_input(
            label="CSV URL",
            value="",
            placeholder="https://docs.google.com/spreadsheets/d/.../export?format=csv&gid=0",
            help="Google Sheets의 CSV export 링크를 입력하세요"
        )

        # 입력이 비어있으면 즉시 강조
        if not csv_url:
            st.warning("CSV URL을 입력해야 데이터가 표시됩니다", icon="⚠️")

    # 3) 입력(선택) 카드: Gemini API Key
    with st.container(border=True):
        st.markdown("### 2) Gemini API Key (선택)")
        st.caption("AI 심층분석 기능을 사용하려면 필요합니다")

        # 기본은 접어서 깔끔하게, 필요할 때만 펼치게
        with st.expander("API Key 입력하기", expanded=False):
            api_key = st.text_input(
                label="Gemini API Key",
                type="password",
                value="",
                placeholder="AI Studio에서 발급받은 키",
                help="키가 없으면 AI 심층분석만 비활성화되며, 대시보드 데이터는 정상 표시됩니다"
            )
    # expander 밖에서도 api_key가 정의되도록 보정
    if "api_key" not in locals():
        api_key = ""

    # 4) 테스트/운영 메모 (읽기 영역)
    st.markdown("### 🧪 테스트 메모")
    st.info(
        "외부 유입 방어를 위해 데이터(CSV URL)와 API Key는 수동 입력 방식으로 운영합니다",
        icon="✅"
    )

# -----------------------------------------------------------------------------
# 3. 메인 로직
# -----------------------------------------------------------------------------
st.title("📊 NEWS&NOW 플랫폼 트래픽 AI 대시보드")

if not csv_url:
    st.warning("👈 왼쪽 사이드바에 CSV URL을 입력해주세요.")
    st.stop()

if not csv_url:
    st.warning(
        "1️⃣ 👈 좌측 사이드바에서 CSV URL을 입력해주세요.\n"
        "2️⃣ 입력 즉시 트래픽 대시보드가 자동으로 로딩됩니다",
        icon="🚀"
    )
    st.stop()

try:
    # 데이터 로드 및 전처리
    df_raw = load_data(csv_url)
    df = preprocess_data(df_raw)

    if len(df) < 2:
        st.error("데이터가 너무 적습니다. (최소 2주치 필요)")
        st.stop()

    # 컬럼명 상수(요청 반영)
    TOTAL_MEM = "총회원수"
    CONV_MEM  = "누적전환회원"
    NEW_MEM   = "신규회원"
    CHURN_MEM = "탈퇴회원"

    # -----------------------------------------------------------------------------
    # [드릴다운] 주차 선택 (선택 주차에 따라 latest/prev 재정의)
    # -----------------------------------------------------------------------------
    st.divider()
    st.subheader("🗓️ 기준 주차")

    weeks = df["주차"].astype(str).tolist()[::-1]  # 최신 주차가 위로
    selected_week = st.selectbox("주차", options=weeks, index=0)

    st.caption("※ 선택한 주차 기준으로 모든 지표와 AI 분석이 업데이트됩니다")

    # 선택 주차 index 찾기
    idx = df[df["주차"].astype(str) == str(selected_week)].index[0]

    latest = df.loc[idx]
    prev = df.loc[idx - 1] if idx > 0 else None

    # 앱 다운로드 합계(선택 주차 기준)
    curr_app = latest.get("방송_AOS 다운로드", 0) + latest.get("방송_iOS 다운로드", 0)
    prev_app = (prev.get("방송_AOS 다운로드", 0) + prev.get("방송_iOS 다운로드", 0)) if prev is not None else None

    # -----------------------------------------------------------------------------
    # [섹션 1] 주간 핵심 지표 (KPI)
    # -----------------------------------------------------------------------------
    st.markdown("### 🚀 주간 핵심 지표")

    # 1행: 트래픽/다운로드
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("📺 방송 PV", f"{latest.get('방송_PV', 0):,.0f}", fmt_delta(latest.get("방송_PV", 0), prev.get("방송_PV", 0) if prev is not None else None))
    with k2:
        st.metric("📰 뉴스 PV", f"{latest.get('뉴스_PV', 0):,.0f}", fmt_delta(latest.get("뉴스_PV", 0), prev.get("뉴스_PV", 0) if prev is not None else None))
    with k3:
        st.metric("👥 방송 UV", f"{latest.get('방송_사용자', 0):,.0f}", fmt_delta(latest.get("방송_사용자", 0), prev.get("방송_사용자", 0) if prev is not None else None))
    with k4:
        st.metric("📱 앱 다운로드", f"{curr_app:,.0f}", fmt_delta(curr_app, prev_app))

    # 2행: 회원 지표
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("👤 총회원수", f"{latest.get(TOTAL_MEM, 0):,.0f}", fmt_delta(latest.get(TOTAL_MEM, 0), prev.get(TOTAL_MEM, 0) if prev is not None else None))
    with m2:
        st.metric("✅ 누적전환회원", f"{latest.get(CONV_MEM, 0):,.0f}", fmt_delta(latest.get(CONV_MEM, 0), prev.get(CONV_MEM, 0) if prev is not None else None))
    with m3:
        st.metric("➕ 신규회원", f"{latest.get(NEW_MEM, 0):,.0f}", fmt_delta(latest.get(NEW_MEM, 0), prev.get(NEW_MEM, 0) if prev is not None else None))
    with m4:
        st.metric("➖ 탈퇴회원", f"{latest.get(CHURN_MEM, 0):,.0f}", fmt_delta(latest.get(CHURN_MEM, 0), prev.get(CHURN_MEM, 0) if prev is not None else None))

    st.divider()

    # -----------------------------------------------------------------------------
    # [섹션 2] 차트 분석 (선택 주차 기준선 표시)
    # -----------------------------------------------------------------------------
    st.subheader("📈 채널별 트래픽 추이 분석")

    tab1, tab2, tab3 = st.tabs(["PV 추이 (통합)", "앱 다운로드 추이", "회원 지표 추이"])

    with tab1:
        fig_pv = px.line(
            df,
            x="주차",
            y=["방송_PV", "뉴스_PV"],
            markers=True,
            title="방송 vs 뉴스 PV 변화 추이"
        )
        fig_pv.update_layout(
            hovermode="x unified",
            xaxis_title=None,
            yaxis_title="페이지뷰 (PV)",
            legend_title="채널",
            template="plotly_white"
        )
        # 선택 주차 기준선
        fig_pv.add_vline(
            x=selected_week,
            line_width=2,
            line_dash="dash",
            line_color="red"
        )
        st.plotly_chart(fig_pv, use_container_width=True)

    with tab2:
        fig_app = px.bar(
            df,
            x="주차",
            y=["방송_AOS 다운로드", "방송_iOS 다운로드"],
            title="OS별 앱 다운로드 추이",
            barmode="group"
        )
        fig_app.update_layout(
            hovermode="x unified",
            xaxis_title=None,
            template="plotly_white"
        )
        fig_app.add_vline(
            x=selected_week,
            line_width=2,
            line_dash="dash",
            line_color="red"
        )
        st.plotly_chart(fig_app, use_container_width=True)

    with tab3:
        mem_cols = [c for c in [TOTAL_MEM, CONV_MEM, NEW_MEM, CHURN_MEM] if c in df.columns]
        if not mem_cols:
            st.warning("회원 지표 컬럼을 찾지 못했습니다. (총회원수/누적전환회원/신규회원/탈퇴회원 헤더 확인 필요)")
        else:
            fig_mem = px.line(
                df,
                x="주차",
                y=mem_cols,
                markers=True,
                title="회원 지표 추이 (총/전환/신규/탈퇴)"
            )
            fig_mem.update_layout(
                hovermode="x unified",
                xaxis_title=None,
                yaxis_title="회원 수",
                legend_title="지표",
                template="plotly_white"
            )
            fig_mem.add_vline(
                x=selected_week,
                line_width=2,
                line_dash="dash",
                line_color="red"
            )
            st.plotly_chart(fig_mem, use_container_width=True)

    # -----------------------------------------------------------------------------
    # [섹션 3] 규칙 기반 자동 요약 (선택 주차 기준)
    # -----------------------------------------------------------------------------
    st.divider()
    st.subheader("⚡ 트래픽 급등/급락 감지 (Quick Check)")

    alerts = []

    def check_surge(label, curr, prev, threshold=0.1):
        try:
            if prev is None:
                return
            prev_val = float(prev)
            curr_val = float(curr)
            if prev_val == 0:
                return
            pct = (curr_val - prev_val) / prev_val
            if abs(pct) >= threshold:
                direction = "급등 📈" if pct > 0 else "급락 📉"
                alerts.append(
                    f"- **{label}**: 전주 대비 **{pct*100:.1f}%** {direction} ({prev_val:,.0f} → {curr_val:,.0f})"
                )
        except Exception:
            return

    # 트래픽/앱 다운로드
    check_surge("방송 PV", latest.get("방송_PV", 0), prev.get("방송_PV", None) if prev is not None else None, threshold=0.1)
    check_surge("뉴스 PV", latest.get("뉴스_PV", 0), prev.get("뉴스_PV", None) if prev is not None else None, threshold=0.1)
    check_surge("방송 앱 다운로드", curr_app, prev_app, threshold=0.15)

    # 회원 지표
    check_surge("신규회원", latest.get(NEW_MEM, 0), prev.get(NEW_MEM, None) if prev is not None else None, threshold=0.2)
    check_surge("탈퇴회원", latest.get(CHURN_MEM, 0), prev.get(CHURN_MEM, None) if prev is not None else None, threshold=0.2)
    check_surge("누적전환회원", latest.get(CONV_MEM, 0), prev.get(CONV_MEM, None) if prev is not None else None, threshold=0.05)

    if prev is None:
        st.info("선택한 주차가 첫 번째 주차라 전주 대비 계산이 불가합니다.")
    elif alerts:
        st.warning("⚠️ 주요 변동 사항이 감지되었습니다:")
        for alert in alerts:
            st.markdown(alert)
    else:
        st.success("✅ 특이 사항 없이 안정적인 추세를 보이고 있습니다.")

    # -----------------------------------------------------------------------------
    # [섹션 4] Gemini AI 심층 리포트 (풍부한 입력 + 보고서형 프롬프트)
    # -----------------------------------------------------------------------------
    st.divider()
    st.subheader("🤖 AI 심층 분석 리포트")

    if "ai_report" not in st.session_state:
        st.session_state["ai_report"] = None

    if st.session_state["ai_report"] is None:
        if st.button("✨ AI 분석 내용 확인하기", type="primary"):
            if not api_key:
                st.error("사이드바에 Gemini API 키를 먼저 입력해주세요!")
            else:
                with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel("gemini-2.5-flash")

                        # ---------------------------
                        # 1) 입력 데이터(근거) 확장
                        # ---------------------------
                        tail_n = 8
                        tail_df = df.tail(tail_n).copy()

                        def safe_int(x):
                            try:
                                return int(float(x))
                            except Exception:
                                return 0

                        def fmt_abs_delta(curr, prev):
                            if prev is None:
                                return "N/A"
                            try:
                                curr = float(curr)
                                prev = float(prev)
                                return f"{curr - prev:+,.0f}"
                            except Exception:
                                return "N/A"

                        metrics = {
                            "방송_PV": ("방송 PV", latest.get("방송_PV", 0), prev.get("방송_PV", 0) if prev is not None else None),
                            "뉴스_PV": ("뉴스 PV", latest.get("뉴스_PV", 0), prev.get("뉴스_PV", 0) if prev is not None else None),
                            "방송_사용자": ("방송 UV", latest.get("방송_사용자", 0), prev.get("방송_사용자", 0) if prev is not None else None),
                            "앱다운로드": ("앱 다운로드", curr_app, prev_app),
                            "총회원수": ("총회원수", latest.get(TOTAL_MEM, 0), prev.get(TOTAL_MEM, 0) if prev is not None else None),
                            "누적전환회원": ("누적전환회원", latest.get(CONV_MEM, 0), prev.get(CONV_MEM, 0) if prev is not None else None),
                            "신규회원": ("신규회원", latest.get(NEW_MEM, 0), prev.get(NEW_MEM, 0) if prev is not None else None),
                            "탈퇴회원": ("탈퇴회원", latest.get(CHURN_MEM, 0), prev.get(CHURN_MEM, 0) if prev is not None else None),
                        }

                        # 최근 8주 근거(간단 딕셔너리)
                        tail_rows = []
                        for _, r in tail_df.iterrows():
                            tail_rows.append({
                                "주차": str(r.get("주차", "")),
                                "방송_PV": safe_int(r.get("방송_PV", 0)),
                                "뉴스_PV": safe_int(r.get("뉴스_PV", 0)),
                                "방송_사용자": safe_int(r.get("방송_사용자", 0)),
                                "앱다운로드": safe_int(r.get("방송_AOS 다운로드", 0) + r.get("방송_iOS 다운로드", 0)),
                                "총회원수": safe_int(r.get(TOTAL_MEM, 0)),
                                "누적전환회원": safe_int(r.get(CONV_MEM, 0)),
                                "신규회원": safe_int(r.get(NEW_MEM, 0)),
                                "탈퇴회원": safe_int(r.get(CHURN_MEM, 0)),
                            })

                        data_summary = f"""
[기준 주차]: {latest.get('주차','')}

[이번주 KPI & 전주 대비]
{chr(10).join([
f"- {label}: {curr:,.0f} (전주대비 {fmt_delta(curr, p)} / {fmt_abs_delta(curr, p)})"
for _, (label, curr, p) in metrics.items()
])}

[규칙 기반 변화 감지(Quick Check)]
{chr(10).join(alerts) if alerts else "- 특이사항 없음"}

[최근 {tail_n}주 추이 데이터(근거)]
{tail_rows}
""".strip()

                        # ---------------------------
                        # 2) 보고서형 프롬프트
                        # ---------------------------
                        prompt = f"""
너는 JTBC의 '수석 데이터 분석가'이며, 임원 보고용 주간 리포트를 작성함
반드시 아래 규칙을 지켜라

[규칙]
- 근거는 제공된 입력 데이터(이번주/전주/최근 8주/Quick Check)에서만 사용
- 입력에 없는 사실은 단정 금지 → 반드시 '확실하지 않음' 또는 '(추측입니다)'로 표시
- 가능하면 숫자를 포함해 근거를 제시(전주대비 %, 절대증감, 최근 8주 추이 중 특징)
- 문장 끝 마침표 금지
- 한국어, 간결한 보고서체(~함/~임)
- 과장 금지, 실행 가능한 제언 중심

[입력 데이터]
{data_summary}

[출력 형식(반드시 준수)]
JTBC 주간 데이터 분석 리포트 ({latest.get('주차','')})
작성자: 안가르쳐주지롱

1. 📌 금주 3줄 요약
- (3줄, 각 줄에 근거 숫자 포함)

2. 🚨 주목해야 할 지표 (Top 2)
- 지표1: (이번주 값 / 전주 대비 % / 절대증감) + 해석 2줄
- 지표2: (이번주 값 / 전주 대비 % / 절대증감) + 해석 2줄

3. 💡 원인 추론 및 제언 (가설)
- 가설 1:  ...
  - 근거(입력 데이터 기반): ...
  - 확인해야 할 데이터/질문: ...
  - 제언(바로 할 액션): ...
- 가설 2:  ...
  - 근거(입력 데이터 기반): ...
  - 확인해야 할 데이터/질문: ...
  - 제언(바로 할 액션): ...
- 가설 3:  ...
  - 근거(입력 데이터 기반): ...
  - 확인해야 할 데이터/질문: ...
  - 제언(바로 할 액션): ...

4. ✅ 다음 액션 체크리스트
- (3~6개, 담당자가 바로 할 수 있는 형태로)
""".strip()

                        # ---------------------------
                        # 3) 생성
                        # ---------------------------
                        response = model.generate_content(prompt)
                        st.session_state["ai_report"] = response.text
                        st.rerun()

                    except Exception as e:
                        st.error(f"AI 분석 중 오류 발생: {e}")
    else:
        st.info("✅ 생성된 리포트 (캐시됨)")
        st.markdown(st.session_state["ai_report"])
        if st.button("🔄 리포트 다시 만들기"):
            st.session_state["ai_report"] = None
            st.rerun()

except Exception as e:
    st.error(f"시스템 오류가 발생했습니다: {e}")
    st.write("힌트: CSV URL이 정확한지, 혹은 컬럼명이 코드와 일치하는지 확인해보세요.")
