import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="분자 특성 비교", page_icon="🧬", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "AqSolDB_missing_cleaned.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")

st.title("🧬 3. 분자 특성 비교")
st.write("분자 특성의 크기에 따라 화합물의 용해도가 어떻게 달라지는지 비교합니다.")

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

    data = df[columns].dropna().copy()

    st.subheader("1. 비교할 분자 특성 선택")

    feature = st.selectbox(
        "비교할 분자 특성을 선택하세요.",
        [
            "MolWt",
            "MolLogP",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "TPSA"
        ]
    )

    st.write(
        "선택한 분자 특성의 값을 기준으로 화합물을 낮은 그룹, 중간 그룹, 높은 그룹으로 나눕니다."
    )

    try:
        data["특성 그룹"] = pd.qcut(
            data[feature],
            q=3,
            labels=["낮은 그룹", "중간 그룹", "높은 그룹"],
            duplicates="drop"
        )

        data = data.dropna(subset=["특성 그룹"])

    except ValueError:
        st.warning("선택한 변수의 값이 충분히 다양하지 않아 그룹을 나눌 수 없습니다.")
        st.stop()

    if data["특성 그룹"].nunique() < 2:
        st.warning("그룹을 비교하기에 데이터의 값이 충분하지 않습니다.")
        st.stop()

    st.divider()

    st.subheader("2. 그룹별 용해도 비교")

    fig1 = px.box(
        data,
        x="특성 그룹",
        y="Solubility",
        color="특성 그룹",
        title=f"{feature} 그룹에 따른 용해도 비교",
        points="outliers"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.write(
        "상자그림을 통해 각 그룹의 용해도 중앙값과 데이터의 분포를 비교할 수 있습니다."
    )

    st.divider()

    st.subheader("3. 그룹별 평균 용해도")

    group_mean = (
        data.groupby("특성 그룹", observed=True)["Solubility"]
        .mean()
        .reset_index()
    )

    group_mean.columns = ["특성 그룹", "평균 용해도"]

    fig2 = px.bar(
        group_mean,
        x="특성 그룹",
        y="평균 용해도",
        color="특성 그룹",
        title=f"{feature} 그룹별 평균 용해도"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        group_mean,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("4. 그룹별 분자 특성 통계")

    group_stats = (
        data.groupby("특성 그룹", observed=True)
        .agg(
            화합물_개수=("Solubility", "count"),
            평균_특성값=(feature, "mean"),
            평균_용해도=("Solubility", "mean"),
            중앙값_용해도=("Solubility", "median")
        )
        .reset_index()
    )

    st.dataframe(
        group_stats,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("5. 분석 결과 해석")

    means = group_mean.set_index("특성 그룹")["평균 용해도"]

    if "낮은 그룹" in means.index and "높은 그룹" in means.index:
        low = means["낮은 그룹"]
        high = means["높은 그룹"]

        if high > low:
            st.info(
                f"{feature}가 높은 그룹의 평균 용해도가 낮은 그룹보다 높게 나타났습니다. "
                "다만 이것만으로 해당 특성이 용해도를 높이는 원인이라고 단정할 수는 없습니다."
            )
        elif high < low:
            st.info(
                f"{feature}가 높은 그룹의 평균 용해도가 낮은 그룹보다 낮게 나타났습니다. "
                "다른 분자 특성도 함께 고려할 필요가 있습니다."
            )
        else:
            st.info("두 그룹의 평균 용해도가 비슷하게 나타났습니다.")

except FileNotFoundError:
    st.error("CSV 파일을 찾을 수 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
