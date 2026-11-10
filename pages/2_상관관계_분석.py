import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="상관관계 분석", page_icon="📊", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "AqSolDB_missing_cleaned.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")

st.title("📊 2. 상관관계 분석")
st.write("분자 특성과 용해도 사이의 관계를 분석합니다.")

try:
    df = load_data()

    columns = [
        "Solubility",
        "MolWt",
        "MolLogP",
        "NumHAcceptors",
        "NumHDonors",
        "NumRotatableBonds",
        "TPSA"
    ]

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    data = df[columns].dropna()

    st.subheader("1. 상관관계 히트맵")

    corr = data.corr()

    fig1 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="분자 특성과 용해도의 상관관계"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.write("""
    상관계수는 두 변수 사이의 관계를 나타내는 값입니다.

    - 1에 가까울수록 강한 양의 상관관계
    - -1에 가까울수록 강한 음의 상관관계
    - 0에 가까울수록 선형적인 상관관계가 약함
    """)

    st.divider()

    st.subheader("2. 용해도와 각 분자 특성의 상관계수")

    target_corr = corr["Solubility"].drop("Solubility")
    target_corr = target_corr.sort_values()

    corr_df = target_corr.reset_index()
    corr_df.columns = ["분자 특성", "상관계수"]

    fig2 = px.bar(
        corr_df,
        x="분자 특성",
        y="상관계수",
        color="상관계수",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title="용해도와 분자 특성의 상관계수"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        corr_df.sort_values(
            "상관계수",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("3. 두 변수 사이의 관계 확인")

    feature = st.selectbox(
        "용해도와 비교할 분자 특성을 선택하세요.",
        [
            "MolWt",
            "MolLogP",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "TPSA"
        ]
    )

    fig3 = px.scatter(
        data,
        x=feature,
        y="Solubility",
        opacity=0.5,
        title=f"{feature}와 용해도의 관계",
        trendline=None
    )

    st.plotly_chart(fig3, use_container_width=True)

    value = corr.loc[feature, "Solubility"]

    st.metric(
        f"{feature}와 용해도의 상관계수",
        f"{value:.3f}"
    )

    if value > 0:
        st.info("두 변수 사이에는 양의 상관관계가 나타납니다.")
    elif value < 0:
        st.info("두 변수 사이에는 음의 상관관계가 나타납니다.")
    else:
        st.info("두 변수 사이의 선형 상관관계가 거의 나타나지 않습니다.")

    st.caption(
        "상관관계가 있다고 해서 한 변수가 다른 변수의 원인이라고 단정할 수는 없습니다."
    )

except FileNotFoundError:
    st.error("CSV 파일을 찾을 수 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
