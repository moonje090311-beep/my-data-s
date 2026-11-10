import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="데이터 탐색", page_icon="🔎", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "AqSolDB_missing_cleaned.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")

st.title("🔎 1. 데이터 탐색")
st.write("화합물 데이터의 기본적인 특징과 분포를 확인합니다.")

try:
    df = load_data()

    numeric_columns = [
        "Solubility",
        "MolWt",
        "MolLogP",
        "NumHAcceptors",
        "NumHDonors",
        "NumRotatableBonds",
        "TPSA"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    st.subheader("1. 데이터 기본 정보")

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터", f"{len(df):,}개")
    col2.metric("변수 개수", f"{len(numeric_columns)}개")
    col3.metric("결측값 개수", f"{df[numeric_columns].isna().sum().sum():,}개")

    st.divider()

    st.subheader("2. 용해도 분포")

    fig1 = px.histogram(
        df.dropna(subset=["Solubility"]),
        x="Solubility",
        nbins=50,
        title="화합물의 용해도 분포"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.write(
        "히스토그램을 통해 화합물의 용해도가 어떤 구간에 많이 분포하는지 확인할 수 있습니다."
    )

    st.divider()

    st.subheader("3. 분자 특성 분포")

    feature = st.selectbox(
        "확인할 분자 특성을 선택하세요.",
        [
            "MolWt",
            "MolLogP",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "TPSA"
        ]
    )

    fig2 = px.histogram(
        df.dropna(subset=[feature]),
        x=feature,
        nbins=40,
        title=f"{feature} 분포"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("4. 변수별 기초 통계")

    st.dataframe(
        df[numeric_columns].describe().T,
        use_container_width=True
    )

    st.divider()

    st.subheader("5. 결측값 확인")

    missing = pd.DataFrame({
        "변수": numeric_columns,
        "결측값 개수": [
            df[col].isna().sum() for col in numeric_columns
        ],
        "결측값 비율(%)": [
            df[col].isna().mean() * 100 for col in numeric_columns
        ]
    })

    st.dataframe(missing, use_container_width=True, hide_index=True)

    fig3 = px.bar(
        missing,
        x="변수",
        y="결측값 개수",
        title="변수별 결측값 개수"
    )

    st.plotly_chart(fig3, use_container_width=True)

except FileNotFoundError:
    st.error("CSV 파일을 찾을 수 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
