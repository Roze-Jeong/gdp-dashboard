import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. 기본 설정 및 유틸리티
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NEWS&방송 플랫폼 트래픽 AI 대시보드", page_icon="📊", layout="wide")

@st.cache_data(ttl=300)
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 전처리: 컬럼명 정리 + 콤마 제거 및 숫자 변환"""
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.astype(str).str.strip()

    # ✅ 텍스트 컬럼(숫자 변환 제외) 규칙: '순위' 컬럼은 텍스트로 유지
    def is_text_col(col: str) -> bool:
        col = str(col)
        if col in ["주차", "날짜", "Date"]:
            return True
        # 키워드/기사 '순위'는 텍스트
        if col.endswith("순위") and ("키워드" in col or "기사" in col):
            return True
        # (선택) 기사 순위 컬럼 패턴이 더 있다면 여기 추가 가능
        return False

    for col in df_clean.columns:
        if is_text_col(col):
            df_clean[col] = df_clean[col].astype(str).str.strip()
            continue

        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)  # ✅ 비중 컬럼이 %로 들어오면 제거
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
st.title("NEWS&방송 플랫폼 트래픽 AI 대시보드")

if not csv_url:
    st.warning(
        "📌  좌측 사이드바에서 CSV URL(필수)을 입력하면 대시보드가 자동으로 로딩됩니다",
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
    # [드롭다운] 기준 주차 선택 (선택 주차에 따라 latest/prev 재정의)
    # -----------------------------------------------------------------------------
    st.subheader("기준 주차")  # ✅ divider 위가 아니라, 여기부터 시작
    
    # ✅ 최신 주차가 위로 보이도록 정렬된 리스트 생성
    weeks = df["주차"].astype(str).tolist()[::-1]
    selected_week = st.selectbox("주차", options=weeks, index=0, key="selected_week")
    
    st.caption("※ 선택한 주차를 기준으로 모든 지표와 AI 분석 결과가 업데이트됩니다.")
    
    # ✅ 여기서 KPI 영역과 명확히 구분
    st.divider()
    
    # -----------------------------------------------------------------------------
    # latest / prev 재정의
    # -----------------------------------------------------------------------------
    # df에서 선택 주차 row를 찾기
    mask = df["주차"].astype(str) == str(selected_week)
    if mask.any():
        idx = df.index[mask][0]
        latest = df.loc[idx]
        prev = df.loc[idx - 1] if (idx - 1) in df.index else None
    else:
        # fallback (이론상 거의 안 탐)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
    
    # -----------------------------------------------------------------------------
    # 앱 다운로드 합계(선택 주차 기준)
    # -----------------------------------------------------------------------------
    curr_app = latest.get("방송_AOS 다운로드", 0) + latest.get("방송_iOS 다운로드", 0)
    prev_app = (
        (prev.get("방송_AOS 다운로드", 0) + prev.get("방송_iOS 다운로드", 0))
        if prev is not None else None
    )

    # -----------------------------------------------------------------------------
    # [섹션 1] 주간 핵심 지표 (KPI) - 요청 레이아웃
    # - 좌측 상: 뉴스 3개
    # - 좌측 하: 방송 3개
    # - 우측: 회원 4개
    # -----------------------------------------------------------------------------
    st.markdown("### 주간 핵심 지표")
    
    # ✅ 뉴스 UV 컬럼 후보(원래 쓰던 규칙 유지)
    NEWS_UV_COL_CANDIDATES = ["뉴스_사용자", "뉴스_UV", "뉴스UV", "뉴스_사용자수"]
    news_uv_col_kpi = next((c for c in NEWS_UV_COL_CANDIDATES if c in df.columns), None)
    news_uv_val = latest.get(news_uv_col_kpi, 0) if news_uv_col_kpi else 0
    prev_news_uv_val = prev.get(news_uv_col_kpi, 0) if (prev is not None and news_uv_col_kpi) else None
    
    # ✅ 2열 레이아웃: 좌(트래픽 묶음) / 우(회원 묶음)
    left, right = st.columns([7, 5], gap="large")
    
    # -------------------------
    # 좌측: 뉴스(상) / 방송(하)
    # -------------------------
    with left:
        # 좌측 상단 박스(뉴스)
        with st.container(border=True):
            st.markdown("#### 📰 뉴스 지표")
            n1, n2, n3 = st.columns(3)
            with n1:
                st.metric(
                    "뉴스 PV",
                    f"{latest.get('뉴스_PV', 0):,.0f}",
                    fmt_delta(
                        latest.get("뉴스_PV", 0),
                        prev.get("뉴스_PV", 0) if prev is not None else None
                    )
                )
            with n2:
                st.metric(
                    "뉴스 UV",
                    f"{news_uv_val:,.0f}",
                    fmt_delta(news_uv_val, prev_news_uv_val)
                )
            with n3:
                # ※ 지금 curr_app/prev_app이 "방송 앱다운로드 합계"라면,
                #   여기서는 일단 공통 KPI로 두고 label만 중립적으로 둠
                st.metric(
                    "앱 다운로드",
                    f"{curr_app:,.0f}",
                    fmt_delta(curr_app, prev_app)
                )
    
        # 좌측 하단 박스(방송)
        with st.container(border=True):
            st.markdown("#### 📺 방송 지표")
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric(
                    "방송 PV",
                    f"{latest.get('방송_PV', 0):,.0f}",
                    fmt_delta(
                        latest.get("방송_PV", 0),
                        prev.get("방송_PV", 0) if prev is not None else None
                    )
                )
            with b2:
                st.metric(
                    "방송 UV",
                    f"{latest.get('방송_사용자', 0):,.0f}",
                    fmt_delta(
                        latest.get("방송_사용자", 0),
                        prev.get("방송_사용자", 0) if prev is not None else None
                    )
                )
            with b3:
                # 방송 지표 3개가 필요하니,
                # 방송 앱다운로드(정확히 보여주려면 별도 합계 계산 필요)
                # 단, 지금 변수명이 없어서 여기서는 안전하게 "0"으로 fallback
                # → 원하면 방송앱다운(=방송_AOS+방송_iOS) 변수를 만들어 꽂아줄 수 있음
                b_app = latest.get("방송_AOS 다운로드", 0) + latest.get("방송_iOS 다운로드", 0)
                prev_b_app = (prev.get("방송_AOS 다운로드", 0) + prev.get("방송_iOS 다운로드", 0)) if prev is not None else None
                st.metric(
                    "방송 앱다운",
                    f"{b_app:,.0f}",
                    fmt_delta(b_app, prev_b_app)
                )
    
    # -------------------------
    # 우측: 회원 지표(4개)
    # -------------------------
    with right:
        with st.container(border=True):
            st.markdown("#### 👤 회원 지표")
            # 2x2로 배치 (가독성 좋음)
            r1, r2 = st.columns(2)
            r3, r4 = st.columns(2)
    
            with r1:
                st.metric(
                    "총회원수",
                    f"{latest.get(TOTAL_MEM, 0):,.0f}",
                    fmt_delta(latest.get(TOTAL_MEM, 0), prev.get(TOTAL_MEM, 0) if prev is not None else None)
                )
            with r2:
                st.metric(
                    "누적전환회원",
                    f"{latest.get(CONV_MEM, 0):,.0f}",
                    fmt_delta(latest.get(CONV_MEM, 0), prev.get(CONV_MEM, 0) if prev is not None else None)
                )
            with r3:
                st.metric(
                    "신규회원",
                    f"{latest.get(NEW_MEM, 0):,.0f}",
                    fmt_delta(latest.get(NEW_MEM, 0), prev.get(NEW_MEM, 0) if prev is not None else None)
                )
            with r4:
                st.metric(
                    "탈퇴회원",
                    f"{latest.get(CHURN_MEM, 0):,.0f}",
                    fmt_delta(latest.get(CHURN_MEM, 0), prev.get(CHURN_MEM, 0) if prev is not None else None)
                )
    
    # ✅ KPI 섹션과 아래 영역 구분선
    st.divider()


    
    # -----------------------------------------------------------------------------
    # [추가 섹션] KPI 아래: 방송/뉴스 상세 탭 + 기간 선택
    # -----------------------------------------------------------------------------
    st.subheader("방송/뉴스 상세 보기")
    
    # ✅ 기간 선택 (탭보다 위에 있어야 탭 전체에 적용)
    st.markdown("### ⏱ 조회 기간")
    
    range_label = st.radio(
        "조회 기간",
        options=["최근 1년", "최근 6개월", "최근 3개월"],
        horizontal=True,
        index=0,
        key="range_label_main",
        label_visibility="collapsed"  # ✅ 라벨 숨김(중복 제거)
    )

    
    weeks_map = {"최근 1년": 52, "최근 6개월": 26, "최근 3개월": 13}
    n_weeks = weeks_map[range_label]
    
    df_range = df.tail(n_weeks).copy()
    df2 = df_range.copy()

    
    # 파생 컬럼 생성
    if "방송_AOS 다운로드" in df2.columns and "방송_iOS 다운로드" in df2.columns:
        df2["방송_앱다운로드"] = df2["방송_AOS 다운로드"] + df2["방송_iOS 다운로드"]
    else:
        df2["방송_앱다운로드"] = 0
    
    NEWS_UV_COL_CANDIDATES = ["뉴스_사용자", "뉴스_UV", "뉴스UV", "뉴스_사용자수"]
    news_uv_col = next((c for c in NEWS_UV_COL_CANDIDATES if c in df2.columns), None)
    
    if "뉴스_AOS 다운로드" in df2.columns and "뉴스_iOS 다운로드" in df2.columns:
        df2["뉴스_앱다운로드"] = df2["뉴스_AOS 다운로드"] + df2["뉴스_iOS 다운로드"]
    else:
        df2["뉴스_앱다운로드"] = 0
    
    # ✅ 탭 순서: 방송 먼저
    tab_n, tab_b = st.tabs(["뉴스", "방송"])

    with tab_n:
        st.markdown("#### 뉴스")
        st.caption("선택 주차 기준 뉴스 PV/UV/앱다운로드 · 키워드 · 유입을 확인합니다")
    
        fig_n_pv = px.line(df2, x="주차", y=["뉴스_PV"], markers=True, title="뉴스 PV 추이")
        fig_n_pv.update_layout(hovermode="x unified", xaxis_title=None, yaxis_title="PV", template="plotly_white")
        if str(selected_week) in df2["주차"].astype(str).tolist():
            fig_n_pv.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_n_pv, use_container_width=True)
    
        if news_uv_col:
            fig_n_uv = px.line(df2, x="주차", y=[news_uv_col], markers=True, title="뉴스 UV 추이")
            fig_n_uv.update_layout(hovermode="x unified", xaxis_title=None, yaxis_title="UV", template="plotly_white")
            if str(selected_week) in df2["주차"].astype(str).tolist():
                fig_n_uv.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(fig_n_uv, use_container_width=True)
        else:
            st.info("뉴스 UV 컬럼을 찾지 못했습니다 (예: 뉴스_사용자)")
    
        fig_n_app = px.bar(df2, x="주차", y=["뉴스_앱다운로드"], title="뉴스 앱 다운로드 추이")
        fig_n_app.update_layout(hovermode="x unified", xaxis_title=None, yaxis_title="다운로드", template="plotly_white")
        if str(selected_week) in df2["주차"].astype(str).tolist():
            fig_n_app.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_n_app, use_container_width=True)

        st.markdown("#### 🏷️ 주별 뉴스 키워드 TOP3")
        st.caption("선택 주차 기준 주요 키워드와 비중(%)을 표시합니다")
        
        # ✅ 변경된 컬럼명으로 맞춤
        kw_cols = ["뉴스_키워드1순위", "뉴스_키워드2순위", "뉴스_키워드3순위"]
        kw_share_cols = ["뉴스_키워드1비중", "뉴스_키워드2비중", "뉴스_키워드3비중"]
        
        # ✅ df 말고 df2 기준으로 체크해야 탭/기간필터가 일관됨
        missing = [c for c in kw_cols + kw_share_cols if c not in df2.columns]
        if missing:
            st.info(f"키워드 TOP3 컬럼을 찾지 못했습니다: {', '.join(missing)}")
        else:
            # latest 대신 선택주차 row를 확정하는게 안전 (없으면 마지막 주차)
            tmp = df2[df2["주차"].astype(str) == str(selected_week)] if "주차" in df2.columns else df2
            latest_row = tmp.iloc[-1] if len(tmp) else df2.iloc[-1]
        
            rows = []
            for i in range(3):
                kw = str(latest_row.get(kw_cols[i], "")).strip()
                share_raw = latest_row.get(kw_share_cols[i], 0)
        
                if not kw or kw.lower() == "nan":
                    continue
        
                try:
                    share_val = float(str(share_raw).replace(",", ""))
                except Exception:
                    share_val = 0.0
        
                rows.append({"순위": f"{i+1}위", "키워드": kw, "비중(%)": share_val})
        
            if not rows:
                st.caption("키워드 값이 비어 있습니다")
            else:
                top_df = pd.DataFrame(rows)
                st.dataframe(top_df, use_container_width=True, hide_index=True)
        
                fig_kw = px.bar(top_df, x="순위", y="비중(%)", text="키워드", title="키워드 비중(%)")
                fig_kw.update_layout(xaxis_title=None, yaxis_title="비중(%)", template="plotly_white")
                fig_kw.update_traces(textposition="outside")
                st.plotly_chart(fig_kw, use_container_width=True, key="news_kw_top3_bar")

        
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
                st.plotly_chart(fig_kw, use_container_width=True)

        st.markdown("#### 뉴스 유입 소스 (사용자/세션)")
        st.caption("소스: 다이렉트 / 네이버 / 다음 / 구글 / 기타 (전체는 KPI로만 표시)")
        
        # 1) 표시 순서 (차트용: 전체 제외)
        sources = ["다이렉트", "네이버", "다음", "구글", "기타"]
        
        # 2) 색상 고정 (요청 반영)
        # - Plotly는 색상 문자열을 받음(HEX 권장)
        color_map = {
            "네이버": "#2ECC71",     # 초록
            "구글":   "#1F77B4",     # 파랑
            "다음":   "#F1C40F",     # 노랑
            "다이렉트": "#7FDBFF",   # 하늘색
            "기타":   "#95A5A6",     # 회색
        }
        
        def to_num(x):
            try:
                return float(str(x).replace(",", "").strip())
            except Exception:
                return 0.0
        
        # 3) 최신(선택 주차) row를 사용 (없으면 마지막 row)
        tmp = df2[df2["주차"].astype(str) == str(selected_week)] if "주차" in df2.columns else df2
        latest_row = tmp.iloc[-1] if len(tmp) else df2.iloc[-1]
        
        # 4) 소스별 사용자/세션 데이터 구성 (전체 제외)
        rows = []
        for s in sources:
            u = to_num(latest_row.get(f"뉴스_유입_{s}_사용자", 0))
            se = to_num(latest_row.get(f"뉴스_유입_{s}_세션", 0))
            rows.append({"유입소스": s, "사용자": u, "세션": se})
        
        acq_df = pd.DataFrame(rows)
        
        # 5) '전체' KPI 값: 원본에 전체가 있으면 그걸 우선 사용, 없으면 합계로 대체
        #    (원본 시트에 전체 컬럼이 있든 없든 안정적으로 동작)
        total_users_raw = latest_row.get("뉴스_유입_전체_사용자", None)
        total_sessions_raw = latest_row.get("뉴스_유입_전체_세션", None)
        
        total_users = to_num(total_users_raw) if total_users_raw not in [None, ""] else acq_df["사용자"].sum()
        total_sessions = to_num(total_sessions_raw) if total_sessions_raw not in [None, ""] else acq_df["세션"].sum()
        
        # 6) KPI 먼저 노출
        k1, k2 = st.columns(2)
        k1.metric("뉴스 유입 사용자(전체)", f"{int(total_users):,}")
        k2.metric("뉴스 유입 세션(전체)", f"{int(total_sessions):,}")
        
        # 7) 차트 렌더
        #    값이 전부 0이면 안내
        if acq_df["사용자"].sum() == 0 and acq_df["세션"].sum() == 0:
            st.info("뉴스 유입 데이터가 모두 0입니다. 컬럼명/값 타입(쉼표 포함 숫자 등)을 확인해주세요.")
        else:
            c1, c2 = st.columns(2)
        
            # ✅ 사용자 기준: 막대 (색상 고정)
            with c1:
                fig_u = px.bar(
                    acq_df,
                    x="유입소스",
                    y="사용자",
                    title="사용자 기준",
                    category_orders={"유입소스": sources},
                    color="유입소스",
                    color_discrete_map=color_map
                )
                fig_u.update_layout(
                    xaxis_title=None,
                    yaxis_title="사용자",
                    template="plotly_white",
                    legend_title_text=None
                )
                st.plotly_chart(fig_u, use_container_width=True, key="news_acq_users_bar_fixed")
        
            # ✅ 세션 기준: 파이 (색상 고정)
            with c2:
                if acq_df["세션"].sum() == 0:
                    st.info("세션 값이 모두 0이라 원형차트를 그릴 수 없습니다.")
                else:
                    fig_s = px.pie(
                        acq_df,
                        names="유입소스",
                        values="세션",
                        title="세션 기준",
                        category_orders={"유입소스": sources},
                        color="유입소스",
                        color_discrete_map=color_map
                    )
                    fig_s.update_layout(template="plotly_white", legend_title_text=None)
                    st.plotly_chart(fig_s, use_container_width=True, key="news_acq_sessions_pie_fixed")


    
    with tab_b:
        st.markdown("#### 방송")
        st.caption("선택 주차 기준 방송 PV/UV/앱다운로드 추이를 확인합니다")
    
        fig_b_pv = px.line(df2, x="주차", y=["방송_PV"], markers=True, title="방송 PV 추이")
        fig_b_pv.update_layout(hovermode="x unified", xaxis_title=None, yaxis_title="PV", template="plotly_white")
        if str(selected_week) in df2["주차"].astype(str).tolist():
            fig_b_pv.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_b_pv, use_container_width=True)
    
        fig_b_uv = px.line(df2, x="주차", y=["방송_사용자"], markers=True, title="방송 UV 추이")
        fig_b_uv.update_layout(hovermode="x unified", xaxis_title=None, yaxis_title="UV", template="plotly_white")
        if str(selected_week) in df2["주차"].astype(str).tolist():
            fig_b_uv.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_b_uv, use_container_width=True)
    
        fig_b_app = px.bar(df2, x="주차", y=["방송_앱다운로드"], title="방송 앱 다운로드 추이")
        fig_b_app.update_layout(hovermode="x unified", xaxis_title=None, yaxis_title="다운로드", template="plotly_white")
        if str(selected_week) in df2["주차"].astype(str).tolist():
            fig_b_app.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_b_app, use_container_width=True)


    # -----------------------------------------------------------------------------
    # [섹션 2] 차트 분석 (선택 주차 기준선 표시)
    # -----------------------------------------------------------------------------
    st.subheader("채널별 트래픽 추이 분석")

    # ✅ [섹션2 전용] 조회 기간 필터 (3/6/12개월)
    st.markdown("### ⏱ 조회 기간 (채널별 추이)")
    range_label_ch = st.radio(
        "조회 기간 (채널별 추이)",
        options=["최근 1년", "최근 6개월", "최근 3개월"],
        horizontal=True,
        index=0,
        key="range_label_channel"  # 🔥 섹션1과 key 충돌 방지
    )
    
    weeks_map = {"최근 1년": 52, "최근 6개월": 26, "최근 3개월": 13}
    n_weeks_ch = weeks_map[range_label_ch]
    
    df_ch = df.tail(n_weeks_ch).copy()

    tab1, tab2, tab3 = st.tabs(["PV 추이 (통합)", "앱 다운로드 추이", "회원 지표 추이"])

    with tab1:
        fig_pv = px.line(
            df_ch,   # ✅ df → df_ch
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
    
        # ✅ 선택 주차가 df_ch에 있을 때만 기준선 표시
        if str(selected_week) in df_ch["주차"].astype(str).tolist():
            fig_pv.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
    
        st.plotly_chart(fig_pv, use_container_width=True, key="channel_pv_line")


    with tab2:
        fig_app = px.bar(
            df_ch,   # ✅ df → df_ch
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
    
        if str(selected_week) in df_ch["주차"].astype(str).tolist():
            fig_app.add_vline(x=selected_week, line_width=2, line_dash="dash", line_color="red")
    
        st.plotly_chart(fig_app, use_container_width=True, key="channel_app_bar")


    with tab3:
        mem_cols = [c for c in [TOTAL_MEM, CONV_MEM, NEW_MEM, CHURN_MEM] if c in df_ch.columns]
    
        if not mem_cols:
            st.warning("회원 지표 컬럼을 찾지 못했습니다. (총회원수/누적전환회원/신규회원/탈퇴회원 헤더 확인 필요)")
        else:
            fig_mem = px.line(
                df_ch,
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
    
            if str(selected_week) in df_ch["주차"].astype(str).tolist():
                fig_mem.add_vline(
                    x=selected_week,
                    line_width=2,
                    line_dash="dash",
                    line_color="red"
                )
    
            st.plotly_chart(
                fig_mem,
                use_container_width=True,
                key="channel_mem_line"
            )



    # -----------------------------------------------------------------------------
    # [섹션 3] 규칙 기반 자동 요약 (선택 주차 기준)
    # -----------------------------------------------------------------------------
    st.divider()
    st.subheader("트래픽 급등/급락 감지")

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
