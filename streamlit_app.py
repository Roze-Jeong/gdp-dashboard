import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. 기본 설정 및 유틸리티
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NEWS&방송플랫폼 트래픽 AI 대시보드", layout="wide")


@st.cache_data(ttl=300)
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 전처리: 컬럼명 정리 + 숫자 변환 (키워드/기사 순위 텍스트는 유지)"""
    df_clean = df.copy()

    # BOM 제거 + 공백 제거
    df_clean.columns = df_clean.columns.astype(str)
    df_clean.columns = df_clean.columns.str.replace("\ufeff", "", regex=False).str.strip()

    def is_text_col(col: str) -> bool:
        col = str(col)
        if col in ["주차", "날짜", "Date"]:
            return True
        # 키워드/기사 순위 컬럼은 텍스트 유지
        if ("키워드" in col or "기사" in col) and "순위" in col:
            return True
        return False

    for col in df_clean.columns:
        if is_text_col(col):
            df_clean[col] = df_clean[col].astype(str).str.strip()
            continue

        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
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


def add_selected_week_line(fig, df_part, selected_week: str):
    try:
        if str(selected_week) in df_part["주차"].astype(str).tolist():
            fig.add_vline(
                x=selected_week,
                line_width=2,
                line_dash="dash",
                line_color="red"
            )
    except Exception:
        pass


def to_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------
# 2. 사이드바
# -----------------------------------------------------------------------------
with st.sidebar:
    # -----------------------------
    # 공통 스타일
    # -----------------------------
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }
    section[data-testid="stSidebar"] hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-section-title {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        opacity: 0.8;
        margin-bottom: 0.35rem;
    }
    .sidebar-note {
        font-size: 0.83rem;
        line-height: 1.5;
        opacity: 0.9;
        padding: 0.75rem 0.85rem;
        border-radius: 0.75rem;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

    # -----------------------------
    # 1) 메뉴
    # -----------------------------
    st.markdown("## 메뉴")
    st.caption("현재 보고 싶은 영역을 선택하세요")

    page_view = st.radio(
        "페이지 선택",
        options=["전체", "뉴스", "방송"],
        index=0,
        label_visibility="collapsed",
        key="page_view"
    )

    st.divider()

    # -----------------------------
    # 2) 데이터 연결
    # -----------------------------
    st.markdown("## 데이터 연결")
    st.caption("대시보드 구동에 필요한 입력값입니다")

    st.markdown('<div class="sidebar-section-title">CSV URL</div>', unsafe_allow_html=True)
    csv_url = st.text_input(
        label="CSV URL",
        value="",
        placeholder="https://docs.google.com/spreadsheets/d/.../export?format=csv&gid=0",
        label_visibility="collapsed",
        help="Google Sheets의 CSV export 링크를 입력하세요"
    )
    st.caption("지정된 플랫폼 트래픽 데이터 문서(CSV)를 입력합니다")

    if not csv_url:
        st.warning("CSV URL을 입력해야 데이터가 표시됩니다", icon="⚠️")

    with st.expander("AI 분석 확장 설정", expanded=False):
        st.markdown('<div class="sidebar-section-title">Gemini API Key</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            label="Gemini API Key",
            type="password",
            value="",
            placeholder="AI Studio에서 발급받은 키",
            label_visibility="collapsed",
            help="키가 없으면 AI 심층분석만 비활성화되며, 기본 데이터는 정상 표시됩니다"
        )
        st.caption("AI 심층분석 기능을 사용할 때만 필요합니다")

    if "api_key" not in locals():
        api_key = ""

    st.divider()

    # -----------------------------
    # 3) 운영 안내
    # -----------------------------
    st.markdown("## 운영 안내")
    st.markdown(
        """
        <div class="sidebar-note">
        외부 유입 방어를 위해 데이터(CSV URL)와 API Key는 수동 입력 방식으로 운영합니다
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 3. 메인 시작
# -----------------------------------------------------------------------------
st.title("NEWS&방송 플랫폼 트래픽 AI 대시보드")

if not csv_url:
    st.warning("좌측 사이드바에서 CSV URL(필수)을 입력하면 대시보드가 자동으로 로딩됩니다")
    st.stop()

try:
    # -------------------------------------------------------------------------
    # 데이터 로드
    # -------------------------------------------------------------------------
    df_raw = load_data(csv_url)
    df = preprocess_data(df_raw)

    if len(df) < 2:
        st.error("데이터가 너무 적습니다. (최소 2주치 필요)")
        st.stop()

    TOTAL_MEM = "총회원수"
    CONV_MEM = "누적전환회원"
    NEW_MEM = "신규회원"
    CHURN_MEM = "탈퇴회원"

    # 기준 주차
    st.divider()
    st.subheader("기준 주차")
    weeks = df["주차"].astype(str).tolist()[::-1]
    selected_week = st.selectbox("주차", options=weeks, index=0)
    st.caption("※ 선택한 주차를 기준으로 모든 지표와 AI 분석 결과가 업데이트됩니다")

    idx = df[df["주차"].astype(str) == str(selected_week)].index[0]
    latest = df.loc[idx]
    prev = df.loc[idx - 1] if idx > 0 else None

    # 전체 앱다운로드 합계(기존 전체 KPI용)
    curr_app = latest.get("방송_AOS 다운로드", 0) + latest.get("방송_iOS 다운로드", 0)
    prev_app = (prev.get("방송_AOS 다운로드", 0) + prev.get("방송_iOS 다운로드", 0)) if prev is not None else None

    NEWS_UV_COL_CANDIDATES = ["뉴스_사용자", "뉴스_UV", "뉴스UV", "뉴스_사용자수"]
    news_uv_col_global = next((c for c in NEWS_UV_COL_CANDIDATES if c in df.columns), None)

    # -------------------------------------------------------------------------
    # 상세 렌더 함수
    # -------------------------------------------------------------------------
    def render_news_detail(df2, latest_row, selected_week_value, news_uv_col):
        st.header("방송/뉴스 상세 지표")
        st.markdown("### 뉴스")
        st.caption("선택 주차 기준 뉴스 PV/UV/앱다운로드 · 키워드 · 유입을 확인합니다")

        st.markdown("##### 뉴스 PV 추이")
        fig_n_pv = px.line(df2, x="주차", y=["뉴스_PV"], markers=True, title=None)
        fig_n_pv.update_layout(
            hovermode="x unified",
            xaxis_title=None,
            yaxis_title="PV",
            template="plotly_white"
        )
        add_selected_week_line(fig_n_pv, df2, selected_week_value)
        st.plotly_chart(fig_n_pv, use_container_width=True, key="news_pv_line")

        st.markdown("##### 뉴스 UV 추이")
        if news_uv_col:
            fig_n_uv = px.line(df2, x="주차", y=[news_uv_col], markers=True, title=None)
            fig_n_uv.update_layout(
                hovermode="x unified",
                xaxis_title=None,
                yaxis_title="UV",
                template="plotly_white"
            )
            add_selected_week_line(fig_n_uv, df2, selected_week_value)
            st.plotly_chart(fig_n_uv, use_container_width=True, key="news_uv_line")
        else:
            st.info("뉴스 UV 컬럼을 찾지 못했습니다 (예: 뉴스_사용자)")

        st.markdown("##### 뉴스 앱 다운로드 추이")
        if "뉴스_AOS 다운로드" in df2.columns and "뉴스_iOS 다운로드" in df2.columns:
            fig_n_app = px.bar(
                df2,
                x="주차",
                y=["뉴스_AOS 다운로드", "뉴스_iOS 다운로드"],
                title="뉴스 앱 다운로드 추이",
                barmode="stack"
            )
            fig_n_app.update_layout(
                hovermode="x unified",
                xaxis_title=None,
                yaxis_title="다운로드",
                template="plotly_white",
                legend_title_text=None
            )
            add_selected_week_line(fig_n_app, df2, selected_week_value)
            st.plotly_chart(fig_n_app, use_container_width=True, key="news_app_stack")
        else:
            st.info("뉴스 앱다운로드 컬럼을 찾지 못했습니다")

        st.markdown("#### 주별 뉴스 키워드 TOP3")
        st.caption("선택 주차 기준 주요 키워드와 비중(%)")

        kw_cols = ["뉴스_키워드1순위", "뉴스_키워드2순위", "뉴스_키워드3순위"]
        kw_share_cols = ["뉴스_키워드1비중", "뉴스_키워드2비중", "뉴스_키워드3비중"]

        missing = [c for c in kw_cols + kw_share_cols if c not in df.columns]
        if missing:
            st.info(f"키워드 TOP3 컬럼을 찾지 못했습니다: {', '.join(missing)}")
        else:
            rows_kw = []
            for i in range(3):
                kw = str(latest_row.get(kw_cols[i], "")).strip()
                share = latest_row.get(kw_share_cols[i], 0)

                if not kw or kw.lower() == "nan":
                    continue

                try:
                    share_val = float(share)
                except Exception:
                    share_val = 0.0

                rows_kw.append({
                    "순위": f"{i+1}위",
                    "키워드": kw,
                    "비중(%)": share_val
                })

            if not rows_kw:
                st.caption("키워드 값이 비어 있습니다")
            else:
                top_df = pd.DataFrame(rows_kw)
                st.dataframe(top_df, use_container_width=True, hide_index=True)

                fig_kw = px.bar(
                    top_df,
                    x="순위",
                    y="비중(%)",
                    text="키워드",
                    title="키워드 비중(%)"
                )
                fig_kw.update_layout(
                    xaxis_title=None,
                    yaxis_title="비중(%)",
                    template="plotly_white"
                )
                fig_kw.update_traces(textposition="outside")
                st.plotly_chart(fig_kw, use_container_width=True, key="news_kw_bar")

        st.markdown("#### 뉴스 유입 소스 (사용자/세션)")
        st.caption("소스 순서: 전체 → 다이렉트 → 네이버 → 다음 → 구글 → 기타")

        ordered_sources = ["전체", "다이렉트", "네이버", "다음", "구글", "기타"]
        rows = []
        for s in ordered_sources:
            u = latest_row.get(f"뉴스_유입_{s}_사용자", 0)
            se = latest_row.get(f"뉴스_유입_{s}_세션", 0)

            try:
                u = float(u)
            except Exception:
                u = 0.0
            try:
                se = float(se)
            except Exception:
                se = 0.0

            rows.append({"유입소스": s, "사용자": u, "세션": se})

        acq_df = pd.DataFrame(rows)

        if acq_df["사용자"].sum() == 0 and acq_df["세션"].sum() == 0:
            st.info("뉴스 유입 컬럼(뉴스_유입_XXX_사용자/세션)을 찾지 못했거나 값이 모두 0입니다")
        else:
            c1, c2 = st.columns(2)

            with c1:
                fig_u = px.bar(
                    acq_df,
                    x="유입소스",
                    y="사용자",
                    title="사용자 기준",
                    category_orders={"유입소스": ordered_sources},
                    color="유입소스",
                    color_discrete_map={"전체": "red"}
                )
                fig_u.update_layout(
                    xaxis_title=None,
                    yaxis_title="사용자",
                    template="plotly_white",
                    showlegend=False
                )
                st.plotly_chart(fig_u, use_container_width=True, key="news_users_bar")

            with c2:
                fig_s = px.pie(
                    acq_df,
                    names="유입소스",
                    values="세션",
                    title="세션 기준",
                    category_orders={"유입소스": ordered_sources},
                    color="유입소스",
                    color_discrete_map={"전체": "red"}
                )
                fig_s.update_layout(template="plotly_white")
                st.plotly_chart(fig_s, use_container_width=True, key="news_sessions_pie")

    def render_broadcast_detail(df2, selected_week_value):
        st.header("방송/뉴스 상세 지표")
        st.markdown("### 방송")
        st.caption("선택 주차 기준 방송 PV/UV/앱다운로드 추이를 확인합니다")

        st.markdown("##### 방송 PV 추이")
        fig_b_pv = px.line(df2, x="주차", y=["방송_PV"], markers=True, title=None)
        fig_b_pv.update_layout(
            hovermode="x unified",
            xaxis_title=None,
            yaxis_title="PV",
            template="plotly_white"
        )
        add_selected_week_line(fig_b_pv, df2, selected_week_value)
        st.plotly_chart(fig_b_pv, use_container_width=True, key="broadcast_pv_line")

        st.markdown("##### 방송 UV 추이")
        fig_b_uv = px.line(df2, x="주차", y=["방송_사용자"], markers=True, title=None)
        fig_b_uv.update_layout(
            hovermode="x unified",
            xaxis_title=None,
            yaxis_title="UV",
            template="plotly_white"
        )
        add_selected_week_line(fig_b_uv, df2, selected_week_value)
        st.plotly_chart(fig_b_uv, use_container_width=True, key="broadcast_uv_line")

        st.markdown("##### 방송 앱 다운로드 추이")
        if "방송_AOS 다운로드" in df2.columns and "방송_iOS 다운로드" in df2.columns:
            fig_b_app = px.bar(
                df2,
                x="주차",
                y=["방송_AOS 다운로드", "방송_iOS 다운로드"],
                title="방송 앱 다운로드 추이",
                barmode="stack"
            )
            fig_b_app.update_layout(
                hovermode="x unified",
                xaxis_title=None,
                yaxis_title="다운로드",
                template="plotly_white",
                legend_title_text=None
            )
            add_selected_week_line(fig_b_app, df2, selected_week_value)
            st.plotly_chart(fig_b_app, use_container_width=True, key="broadcast_app_stack")
        else:
            st.info("방송 앱다운로드 컬럼을 찾지 못했습니다")

    # -------------------------------------------------------------------------
    # 페이지 라우팅
    # -------------------------------------------------------------------------
    weeks_map = {"최근 1년": 52, "최근 6개월": 26, "최근 3개월": 13}

    if page_view == "전체":
        st.divider()
        st.header("주간 핵심 지표")

        news_uv_val = latest.get(news_uv_col_global, 0) if news_uv_col_global else 0
        prev_news_uv_val = prev.get(news_uv_col_global, 0) if (prev is not None and news_uv_col_global) else None

        left, right = st.columns([7, 5], gap="large")

        with left:
            with st.container(border=True):
                st.subheader("뉴스 지표")
                n1, n2, n3 = st.columns(3)
                with n1:
                    st.metric(
                        "뉴스 PV",
                        f"{latest.get('뉴스_PV', 0):,.0f}",
                        fmt_delta(latest.get("뉴스_PV", 0), prev.get("뉴스_PV", 0) if prev is not None else None)
                    )
                with n2:
                    st.metric(
                        "뉴스 UV",
                        f"{news_uv_val:,.0f}",
                        fmt_delta(news_uv_val, prev_news_uv_val)
                    )
                with n3:
                    st.metric(
                        "앱 다운로드",
                        f"{curr_app:,.0f}",
                        fmt_delta(curr_app, prev_app)
                    )

            with st.container(border=True):
                st.subheader("방송 지표")
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.metric(
                        "방송 PV",
                        f"{latest.get('방송_PV', 0):,.0f}",
                        fmt_delta(latest.get("방송_PV", 0), prev.get("방송_PV", 0) if prev is not None else None)
                    )
                with b2:
                    st.metric(
                        "방송 UV",
                        f"{latest.get('방송_사용자', 0):,.0f}",
                        fmt_delta(latest.get("방송_사용자", 0), prev.get("방송_사용자", 0) if prev is not None else None)
                    )
                with b3:
                    b_app = latest.get("방송_AOS 다운로드", 0) + latest.get("방송_iOS 다운로드", 0)
                    prev_b_app = (prev.get("방송_AOS 다운로드", 0) + prev.get("방송_iOS 다운로드", 0)) if prev is not None else None
                    st.metric("방송 앱다운", f"{b_app:,.0f}", fmt_delta(b_app, prev_b_app))

        with right:
            with st.container(border=True):
                st.subheader("회원 지표")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("총회원수", f"{latest.get(TOTAL_MEM, 0):,.0f}",
                              fmt_delta(latest.get(TOTAL_MEM, 0), prev.get(TOTAL_MEM, 0) if prev is not None else None))
                with m2:
                    st.metric("누적전환회원", f"{latest.get(CONV_MEM, 0):,.0f}",
                              fmt_delta(latest.get(CONV_MEM, 0), prev.get(CONV_MEM, 0) if prev is not None else None))
                with m3:
                    st.metric("신규회원", f"{latest.get(NEW_MEM, 0):,.0f}",
                              fmt_delta(latest.get(NEW_MEM, 0), prev.get(NEW_MEM, 0) if prev is not None else None))
                with m4:
                    st.metric("탈퇴회원", f"{latest.get(CHURN_MEM, 0):,.0f}",
                              fmt_delta(latest.get(CHURN_MEM, 0), prev.get(CHURN_MEM, 0) if prev is not None else None))

        st.divider()
        st.header("채널별 트래픽 추이 분석")

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
            add_selected_week_line(fig_pv, df, selected_week)
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
            add_selected_week_line(fig_app, df, selected_week)
            st.plotly_chart(fig_app, use_container_width=True)

        with tab3:
            mem_cols = [c for c in [TOTAL_MEM, CONV_MEM, NEW_MEM, CHURN_MEM] if c in df.columns]
            if not mem_cols:
                st.warning("회원 지표 컬럼을 찾지 못했습니다. (총회원수/누적전환회원/신규회원/탈퇴회원)")
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
                add_selected_week_line(fig_mem, df, selected_week)
                st.plotly_chart(fig_mem, use_container_width=True)

        st.divider()
        st.header("트래픽 급등/급락 감지")

        alerts = []

        def check_surge(label, curr, prev_value, threshold=0.1):
            try:
                if prev_value is None:
                    return
                prev_val = float(prev_value)
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

        check_surge("방송 PV", latest.get("방송_PV", 0), prev.get("방송_PV", None) if prev is not None else None, threshold=0.1)
        check_surge("뉴스 PV", latest.get("뉴스_PV", 0), prev.get("뉴스_PV", None) if prev is not None else None, threshold=0.1)
        check_surge("방송 앱 다운로드", curr_app, prev_app, threshold=0.15)
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

        st.divider()
        st.header("AI 심층 분석")

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

                            tail_n = 8
                            tail_df = df.tail(tail_n).copy()

                            def safe_int(x):
                                try:
                                    return int(float(x))
                                except Exception:
                                    return 0

                            def fmt_abs_delta(curr, prev_value):
                                if prev_value is None:
                                    return "N/A"
                                try:
                                    curr = float(curr)
                                    prev_value = float(prev_value)
                                    return f"{curr - prev_value:+,.0f}"
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
작성자: Gemini

1. 📌 금주 3줄 요약
- (3줄, 각 줄에 근거 숫자 포함)

2. 🚨 주목해야 할 지표 (Top 2)
- 지표1: (이번주 값 / 전주 대비 % / 절대증감) + 해석 2줄
- 지표2: (이번주 값 / 전주 대비 % / 절대증감) + 해석 2줄

3. 💡 원인 추론 및 제언 (가설)
- 가설 1: ...
  - 근거(입력 데이터 기반): ...
  - 확인해야 할 데이터/질문: ...
  - 제언(바로 할 액션): ...
- 가설 2: ...
  - 근거(입력 데이터 기반): ...
  - 확인해야 할 데이터/질문: ...
  - 제언(바로 할 액션): ...
- 가설 3: ...
  - 근거(입력 데이터 기반): ...
  - 확인해야 할 데이터/질문: ...
  - 제언(바로 할 액션): ...

4. ✅ 다음 액션 체크리스트
- (3~6개, 담당자가 바로 할 수 있는 형태로)
""".strip()

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

    elif page_view == "뉴스":
        st.divider()
        st.markdown("##### 조회 기간")
        range_label = st.radio(
            "조회 기간",
            options=["최근 1년", "최근 6개월", "최근 3개월"],
            horizontal=True,
            index=0,
            key="range_label_news_only",
            label_visibility="collapsed"
        )
        df2 = df.tail(weeks_map[range_label]).copy()
        render_news_detail(df2, latest, selected_week, news_uv_col_global)

    elif page_view == "방송":
        st.divider()
        st.markdown("##### 조회 기간")
        range_label = st.radio(
            "조회 기간",
            options=["최근 1년", "최근 6개월", "최근 3개월"],
            horizontal=True,
            index=0,
            key="range_label_broadcast_only",
            label_visibility="collapsed"
        )
        df2 = df.tail(weeks_map[range_label]).copy()
        render_broadcast_detail(df2, selected_week)

except Exception as e:
    st.error(f"시스템 오류가 발생했습니다: {e}")
    st.write("힌트: CSV URL이 정확한지, 혹은 컬럼명이 코드와 일치하는지 확인해보세요.")


