
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.title("1. 데이터 탐색")
st.write(
    "AqSolDB 데이터를 다양한 그래프로 살펴보고 "
    "화합물의 분자 특성과 용해도 분포를 탐구합니다."
)

st.divider()


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

DATA_PATH = (
    Path(__file__).parent.parent
    / "AqSolDB_missing_cleaned.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")


try:
    df = load_data()

except FileNotFoundError:
    st.error("AqSolDB_missing_cleaned.csv 파일을 찾을 수 없습니다.")
    st.stop()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 데이터 전처리
# --------------------------------------------------

feature_names = {
    "분자량": "MolWt",
    "분배계수": "MolLogP",
    "수소 결합 수용체 수": "NumHAcceptors",
    "수소 결합 공여체 수": "NumHDonors",
    "회전 가능한 결합 수": "NumRotatableBonds",
    "극성 표면적": "TPSA"
}

required_columns = ["Solubility"] + list(feature_names.values())

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"필요한 데이터 열이 없습니다: {missing_columns}")
    st.stop()

data = df[required_columns].copy()

for col in required_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")


# --------------------------------------------------
# 4. 데이터 기본 정보
# --------------------------------------------------

st.header("1번 구역: 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 화합물 수", f"{len(data):,}개")
col2.metric("분자 특성 수", f"{len(feature_names)}개")
col3.metric(
    "평균 용해도",
    f"{data['Solubility'].mean():.3f}"
)

with st.expander("데이터 미리 보기"):
    st.dataframe(data.head(10), use_container_width=True)

with st.expander("데이터 기초 통계"):
    st.dataframe(
        data.describe().round(3),
        use_container_width=True
    )


# --------------------------------------------------
# 5. 분자량과 용해도 분포
# --------------------------------------------------

st.divider()
st.header("2번 구역: 주요 데이터 분포")

st.write(
    "분자량과 용해도 값이 각각 어떤 범위에 많이 분포하는지 "
    "히스토그램으로 확인합니다."
)

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        data,
        x="MolWt",
        nbins=40,
        title="분자량 분포",
        labels={
            "MolWt": "분자량",
            "count": "화합물 수"
        },
        color_discrete_sequence=["#4C91D8"]
    )

    fig1.update_layout(
        xaxis_title="분자량",
        yaxis_title="화합물 수",
        height=400
    )

    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(
        data,
        x="Solubility",
        nbins=40,
        title="용해도 분포",
        labels={
            "Solubility": "용해도",
            "count": "화합물 수"
        },
        color_discrete_sequence=["#E58A54"]
    )

    fig2.update_layout(
        xaxis_title="용해도",
        yaxis_title="화합물 수",
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

st.subheader("이 그래프로 알 수 있는 것")

st.write(
    f"전체 화합물의 평균 분자량은 {data['MolWt'].mean():.2f}이며, "
    f"평균 용해도는 {data['Solubility'].mean():.2f}입니다. "
    "히스토그램을 통해 각 값이 어느 구간에 집중되어 있는지 "
    "확인할 수 있습니다."
)


# --------------------------------------------------
# 6. 용해도 구간별 분자 특성 비교
# --------------------------------------------------

st.divider()
st.header("3번 구역: 용해도 구간별 분자 특성 비교")

st.write(
    "화합물을 용해도에 따라 낮음, 중간, 높음의 세 그룹으로 나누고 "
    "각 그룹의 분자 특성을 상자그림으로 비교합니다."
)

comparison_data = data.dropna(subset=["Solubility"]).copy()

# 용해도의 1/3, 2/3 분위수를 기준으로 세 그룹 분류
q1 = comparison_data["Solubility"].quantile(1 / 3)
q2 = comparison_data["Solubility"].quantile(2 / 3)

comparison_data["용해도 그룹"] = pd.cut(
    comparison_data["Solubility"],
    bins=[
        float("-inf"),
        q1,
        q2,
        float("inf")
    ],
    labels=["용해도 낮음", "용해도 중간", "용해도 높음"],
    include_lowest=True
)

selected_feature_name = st.selectbox(
    "비교할 분자 특성을 선택하세요.",
    list(feature_names.keys()),
    key="boxplot_feature"
)

selected_feature = feature_names[selected_feature_name]

box_data = comparison_data[
    ["용해도 그룹", selected_feature]
].dropna()

fig3 = px.box(
    box_data,
    x="용해도 그룹",
    y=selected_feature,
    color="용해도 그룹",
    title=f"용해도 그룹별 {selected_feature_name} 비교",
    labels={
        "용해도 그룹": "용해도 그룹",
        selected_feature: selected_feature_name
    },
    color_discrete_sequence=[
        "#75AADB",
        "#80BFA0",
        "#E5A16B"
    ]
)

fig3.update_layout(
    height=500,
    showlegend=False
)

st.plotly_chart(fig3, use_container_width=True)

# 그룹별 평균값 표
mean_table = comparison_data.groupby(
    "용해도 그룹",
    observed=False
)[list(feature_names.values())].mean()

mean_table.columns = list(feature_names.keys())

st.subheader("용해도 그룹별 분자 특성 평균")

st.dataframe(
    mean_table.round(3),
    use_container_width=True
)

st.caption(
    "상자그림은 각 용해도 그룹에서 선택한 분자 특성의 "
    "중앙값과 데이터 분포를 비교하는 그래프입니다."
)


# --------------------------------------------------
# 7. 분자 특성별 용해도 상관관계
# --------------------------------------------------

st.divider()
st.header("4번 구역: 분자 특성과 용해도의 관계")

st.write(
    "분자 특성을 선택하여 용해도와 어떤 관계가 나타나는지 "
    "산점도로 확인합니다."
)

selected_name = st.selectbox(
    "비교할 분자 특성을 선택하세요.",
    list(feature_names.keys()),
    key="scatter_feature"
)

selected_column = feature_names[selected_name]

scatter_data = data[
    [selected_column, "Solubility"]
].dropna()

correlation = scatter_data[selected_column].corr(
    scatter_data["Solubility"]
)

fig4 = px.scatter(
    scatter_data,
    x=selected_column,
    y="Solubility",
    opacity=0.5,
    trendline="ols",
    title=f"{selected_name}과 용해도의 관계",
    labels={
        selected_column: selected_name,
        "Solubility": "용해도"
    },
    color_discrete_sequence=["#8B78C6"]
)

fig4.update_layout(
    height=500,
    xaxis_title=selected_name,
    yaxis_title="용해도"
)

st.plotly_chart(fig4, use_container_width=True)

st.metric(
    f"{selected_name}과 용해도의 상관계수",
    f"{correlation:.3f}"
)

if correlation > 0:
    relationship = "양의 상관관계"
elif correlation < 0:
    relationship = "음의 상관관계"
else:
    relationship = "선형 상관관계가 거의 없는 관계"

st.subheader("이 그래프로 알 수 있는 것")

st.write(
    f"현재 선택한 {selected_name}과 용해도 사이에는 "
    f"{relationship}가 나타납니다. "
    "상관계수는 두 변수 사이의 선형적인 관계를 나타내며, "
    "인과관계를 증명하는 것은 아닙니다."
)


# --------------------------------------------------
# 8. 분자 특성 사이의 상관관계
# --------------------------------------------------

st.divider()
st.header("5번 구역: 분자 특성 사이의 상관관계")

st.write(
    "여러 분자 특성이 서로 어떤 관계를 가지는지 "
    "상관관계 히트맵으로 비교합니다."
)

heatmap_data = data[
    required_columns
].corr()

korean_columns = {
    "Solubility": "용해도",
    "MolWt": "분자량",
    "MolLogP": "분배계수",
    "NumHAcceptors": "수소 결합 수용체 수",
    "NumHDonors": "수소 결합 공여체 수",
    "NumRotatableBonds": "회전 가능한 결합 수",
    "TPSA": "극성 표면적"
}

heatmap_data = heatmap_data.rename(
    index=korean_columns,
    columns=korean_columns
)

fig5 = px.imshow(
    heatmap_data,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="분자 특성 사이의 상관관계 히트맵"
)

fig5.update_layout(height=650)

st.plotly_chart(fig5, use_container_width=True)

st.subheader("이 그래프로 알 수 있는 것")

st.write(
    "상관계수가 1에 가까울수록 양의 상관관계가 강하고, "
    "-1에 가까울수록 음의 상관관계가 강합니다. "
    "0에 가까우면 선형적인 상관관계가 약하다는 의미입니다."
)


# --------------------------------------------------
# 9. 결측값 확인
# --------------------------------------------------

st.divider()
st.header("6번 구역: 결측값 확인")

st.write(
    "각 분자 특성에 비어 있는 데이터가 얼마나 있는지 확인합니다."
)

missing_table = pd.DataFrame({
    "분자 특성": required_columns,
    "결측값 개수": [
        int(data[col].isna().sum())
        for col in required_columns
    ],
    "결측값 비율(%)": [
        round(data[col].isna().mean() * 100, 2)
        for col in required_columns
    ]
})

st.dataframe(
    missing_table,
    use_container_width=True,
    hide_index=True
)

st.subheader("이 그래프로 알 수 있는 것")

st.write(
    "결측값 개수와 비율을 확인하여 데이터가 얼마나 완전한지 "
    "살펴볼 수 있습니다. 결측값이 존재하는 특성은 이후 분석에서 "
    "적절한 전처리 과정이 필요할 수 있습니다."
)


# --------------------------------------------------
# 10. 전체 분석 정리
# --------------------------------------------------

st.divider()
st.header("7번 구역: 탐색 결과 정리")

st.write(
    f"이번 탐색에서는 총 {len(data):,}개의 화합물 데이터를 활용했습니다. "
    "분자량과 용해도의 분포를 확인하고, 용해도 구간에 따라 "
    "분자 특성이 어떻게 달라지는지 비교했습니다. "
    "또한 산점도와 상관관계 히트맵을 통해 분자 특성과 용해도 사이의 "
    "통계적 관계를 살펴보았습니다."
)

st.info(
    "상관관계가 나타난다고 해서 특정 분자 특성이 용해도를 "
    "직접 변화시킨다는 뜻은 아닙니다. 그래프는 데이터에서 관찰되는 "
    "경향을 보여주는 탐색 도구입니다."
)
