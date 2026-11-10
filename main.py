
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="화합물 용해도 데이터 그래프 도감",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 화합물 용해도 데이터 그래프 도감")
st.write(
    "AqSolDB 데이터를 활용하여 분자 특성과 화합물의 용해도 사이의 "
    "관계를 다양한 그래프로 탐구합니다."
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

DATA_PATH = Path(__file__).parent / "AqSolDB_missing_cleaned.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(DATA_PATH)

except FileNotFoundError:
    st.error(
        "AqSolDB_missing_cleaned.csv 파일을 찾을 수 없습니다. "
        "main.py와 같은 폴더에 있는지 확인해 주세요."
    )
    st.stop()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 데이터 전처리
# --------------------------------------------------

target = "Solubility"

feature_names = {
    "분자량": "MolWt",
    "분배계수": "MolLogP",
    "수소 결합 수용체 수": "NumHAcceptors",
    "수소 결합 공여체 수": "NumHDonors",
    "회전 가능한 결합 수": "NumRotatableBonds",
    "극성 표면적": "TPSA"
}

required_columns = [target] + list(feature_names.values())

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

data = df[required_columns].copy()

for col in required_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=[target]).reset_index(drop=True)

if len(data) < 10:
    st.error("분석에 사용할 데이터가 부족합니다.")
    st.stop()


# --------------------------------------------------
# 4. 데이터 기본 정보
# --------------------------------------------------

st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 화합물 수", f"{len(data):,}개")
col2.metric("분석할 분자 특성", f"{len(feature_names)}개")
col3.metric(
    "용해도 범위",
    f"{data[target].min():.2f} ~ {data[target].max():.2f}"
)

with st.expander("데이터 미리 보기"):
    st.dataframe(data.head(10), use_container_width=True)

with st.expander("데이터 기초 통계"):
    st.dataframe(data.describe().round(3), use_container_width=True)


# --------------------------------------------------
# 5. 분자량과 용해도의 상관관계
# --------------------------------------------------

st.divider()
st.header("2. 분자량과 용해도의 관계")

correlation_data = data[["MolWt", "Solubility"]].dropna()

correlation = correlation_data["MolWt"].corr(
    correlation_data["Solubility"]
)

fig1 = px.scatter(
    correlation_data,
    x="MolWt",
    y="Solubility",
    opacity=0.45,
    labels={
        "MolWt": "분자량",
        "Solubility": "용해도"
    },
    title="분자량과 용해도의 상관관계"
)

fig1.update_layout(height=500)

st.plotly_chart(fig1, use_container_width=True)

if correlation > 0:
    relationship = "양의"
elif correlation < 0:
    relationship = "음의"
else:
    relationship = "선형"

st.subheader("이 그래프로 알 수 있는 것")

st.write(
    f"이 데이터에서는 분자량과 용해도 사이에 "
    f"{relationship} 선형 상관관계가 나타납니다 "
    f"(상관계수 r = {correlation:.2f}). "
    "이는 두 값이 함께 변하는 경향을 뜻하며, "
    "원인과 결과를 증명하는 것은 아닙니다."
)


# --------------------------------------------------
# 6. 용해도 데이터 분포
# --------------------------------------------------

st.divider()
st.header("3. 용해도 데이터 분포")

st.write(
    "전체 화합물의 용해도 값이 어떤 범위에 많이 분포하는지 "
    "히스토그램으로 확인합니다."
)

fig2 = px.histogram(
    data,
    x="Solubility",
    nbins=40,
    labels={
        "Solubility": "용해도",
        "count": "화합물 수"
    },
    title="화합물의 용해도 분포",
    color_discrete_sequence=["#4C91D8"]
)

fig2.update_layout(
    height=500,
    xaxis_title="용해도",
    yaxis_title="화합물 수",
    bargap=0.08
)

st.plotly_chart(fig2, use_container_width=True)

median_solubility = data["Solubility"].median()
mean_solubility = data["Solubility"].mean()

col1, col2 = st.columns(2)

col1.metric("평균 용해도", f"{mean_solubility:.3f}")
col2.metric("중앙값 용해도", f"{median_solubility:.3f}")

st.subheader("이 그래프로 알 수 있는 것")

st.write(
    f"용해도 데이터의 평균은 {mean_solubility:.2f}, "
    f"중앙값은 {median_solubility:.2f}입니다. "
    "히스토그램을 통해 화합물의 용해도가 어느 구간에 집중되어 있는지 "
    "확인할 수 있습니다."
)


# --------------------------------------------------
# 7. 분자 특성별 용해도 비교
# --------------------------------------------------

st.divider()
st.header("4. 분자 특성별 용해도 비교")

st.write(
    "분자 특성을 하나 선택하여 해당 특성과 용해도 사이의 "
    "관계를 비교할 수 있습니다."
)

selected_name = st.selectbox(
    "비교할 분자 특성을 선택하세요.",
    list(feature_names.keys())
)

selected_column = feature_names[selected_name]

comparison_data = data[
    [selected_column, "Solubility"]
].dropna()

selected_corr = comparison_data[selected_column].corr(
    comparison_data["Solubility"]
)

fig3 = px.scatter(
    comparison_data,
    x=selected_column,
    y="Solubility",
    opacity=0.5,
    trendline="ols",
    labels={
        selected_column: selected_name,
        "Solubility": "용해도"
    },
    title=f"{selected_name}과 용해도의 관계",
    color_discrete_sequence=["#E58A54"]
)

fig3.update_layout(
    height=500,
    xaxis_title=selected_name,
    yaxis_title="용해도"
)

st.plotly_chart(fig3, use_container_width=True)

st.metric(
    f"{selected_name}과 용해도의 상관계수",
    f"{selected_corr:.3f}"
)

st.subheader("이 그래프로 알 수 있는 것")

if selected_corr > 0:
    direction = "양의 상관관계"
elif selected_corr < 0:
    direction = "음의 상관관계"
else:
    direction = "뚜렷한 선형 상관관계가 거의 없는 관계"

st.write(
    f"{selected_name}과 용해도 사이에는 "
    f"{direction}가 나타납니다. "
    "상관계수는 두 변수 사이의 선형적인 관계를 나타내므로 "
    "이 값만으로 용해도에 영향을 미치는 원인을 단정할 수는 없습니다."
)


# --------------------------------------------------
# 8. 추가 그래프: 분자 특성 상관관계
# --------------------------------------------------

st.divider()
st.header("5. 추가 그래프 공간")

st.subheader("분자 특성 사이의 상관관계")

st.write(
    "여러 분자 특성들이 서로 어떤 관계를 가지고 있는지 "
    "상관관계 히트맵으로 확인합니다."
)

heatmap_data = data[
    list(feature_names.values()) + ["Solubility"]
].corr()

heatmap_data.index = [
    "분자량",
    "분배계수",
    "수소 결합 수용체 수",
    "수소 결합 공여체 수",
    "회전 가능한 결합 수",
    "극성 표면적",
    "용해도"
]

heatmap_data.columns = heatmap_data.index

fig4 = px.imshow(
    heatmap_data,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="분자 특성 사이의 상관관계 히트맵"
)

fig4.update_layout(height=600)

st.plotly_chart(fig4, use_container_width=True)

st.subheader("이 그래프로 알 수 있는 것")

st.write(
    "히트맵의 색과 숫자를 통해 각 분자 특성 사이의 상관관계를 "
    "한눈에 비교할 수 있습니다. 1에 가까울수록 양의 상관관계, "
    "-1에 가까울수록 음의 상관관계를 나타냅니다. "
    "0에 가까우면 선형적인 상관관계가 약하다는 뜻입니다."
)


# --------------------------------------------------
# 9. 다중 선형 회귀를 이용한 용해도 예측
# --------------------------------------------------

st.divider()
st.header("6. 다중 선형 회귀를 이용한 용해도 예측")

st.write(
    "분자 특성을 선택하여 학습에 사용할 변수를 결정하고 "
    "용해도를 예측합니다."
)

selected_features = []

feature_columns = st.columns(3)

for i, (korean_name, column_name) in enumerate(feature_names.items()):
    with feature_columns[i % 3]:
        checked = st.checkbox(
            korean_name,
            value=True,
            key=f"model_{column_name}"
        )

        if checked:
            selected_features.append(column_name)

if len(selected_features) == 0:
    st.warning("최소 한 가지 이상의 분자 특성을 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 10. 학습 데이터와 테스트 데이터 분리
# --------------------------------------------------

st.subheader("학습 데이터와 테스트 데이터 분리")

st.write(
    "데이터를 10개씩 나누고 각 묶음에서 앞의 3개는 테스트 데이터, "
    "나머지 7개는 학습 데이터로 사용합니다."
)

test_mask = (np.arange(len(data)) % 10) < 3

train_data = data.loc[~test_mask].copy()
test_data = data.loc[test_mask].copy()

col1, col2 = st.columns(2)

col1.metric("학습 데이터", f"{len(train_data):,}개")
col2.metric("테스트 데이터", f"{len(test_data):,}개")


# --------------------------------------------------
# 11. 모델 학습 및 예측
# --------------------------------------------------

X_train = train_data[selected_features]
y_train = train_data[target]

X_test = test_data[selected_features]
y_test = test_data[target]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.success("모델 학습이 완료되었습니다.")


# --------------------------------------------------
# 12. 모델 성능 평가
# --------------------------------------------------

st.subheader("모델 성능 평가")

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col1, col2, col3 = st.columns(3)

col1.metric("R² 결정계수", f"{r2:.4f}")
col2.metric("MAE 평균 절대 오차", f"{mae:.4f}")
col3.metric("RMSE 평균 제곱근 오차", f"{rmse:.4f}")

st.write("""
- R²: 1에 가까울수록 모델이 데이터를 잘 설명합니다. 음수가 나올 수도 있습니다.
- MAE: 실제값과 예측값의 평균적인 차이입니다. 낮을수록 좋습니다.
- RMSE: 큰 예측 오차에 더 큰 영향을 받는 평가 지표입니다. 낮을수록 좋습니다.
""")


# --------------------------------------------------
# 13. 실제값과 예측값 비교
# --------------------------------------------------

st.subheader("실제 용해도와 예측 용해도 비교")

result_df = pd.DataFrame({
    "실제 용해도": y_test.to_numpy(),
    "예측 용해도": y_pred
})

min_value = min(
    result_df["실제 용해도"].min(),
    result_df["예측 용해도"].min()
)

max_value = max(
    result_df["실제 용해도"].max(),
    result_df["예측 용해도"].max()
)

fig5 = px.scatter(
    result_df,
    x="실제 용해도",
    y="예측 용해도",
    title="실제 용해도와 예측 용해도 비교"
)

fig5.add_trace(
    go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode="lines",
        name="완벽한 예측 기준선",
        line=dict(dash="dash")
    )
)

fig5.update_layout(
    height=500,
    xaxis_title="실제 용해도",
    yaxis_title="예측 용해도"
)

st.plotly_chart(fig5, use_container_width=True)


# --------------------------------------------------
# 14. 예측 결과 표
# --------------------------------------------------

st.subheader("테스트 데이터 예측 결과")

result_df["예측 오차"] = (
    result_df["예측 용해도"] - result_df["실제 용해도"]
)

result_df["절대 오차"] = result_df["예측 오차"].abs()

result_df.insert(
    0,
    "데이터 번호",
    test_data.index.to_numpy() + 1
)

st.dataframe(
    result_df.round(4),
    use_container_width=True
)

csv = result_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "예측 결과 CSV 다운로드",
    data=csv,
    file_name="solubility_prediction_results.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 15. 새로운 화합물 용해도 예측
# --------------------------------------------------

st.divider()
st.header("7. 새로운 화합물의 용해도 예측")

st.write(
    "분자 특성값을 직접 입력하면 학습된 모델을 이용해 "
    "용해도를 예측할 수 있습니다."
)

input_values = {}

with st.form("prediction_form"):

    input_columns = st.columns(2)

    for i, (korean_name, column_name) in enumerate(feature_names.items()):
        with input_columns[i % 2]:
            median_value = data[column_name].median()

            input_values[column_name] = st.number_input(
                korean_name,
                value=float(median_value)
                if pd.notna(median_value) else 0.0,
                format="%.4f"
            )

    submitted = st.form_submit_button("용해도 예측하기")

if submitted:
    input_df = pd.DataFrame([input_values])[selected_features]

    prediction = model.predict(input_df)[0]

    st.success(f"예측된 용해도: {prediction:.4f}")

    st.caption(
        "모델의 예측값은 실제 실험으로 측정한 용해도와 다를 수 있습니다."
    )


# --------------------------------------------------
# 16. 분석 마무리
# --------------------------------------------------

st.divider()
st.header("분석 결과 정리")

st.write(
    f"총 {len(data):,}개의 화합물 데이터를 활용하여 "
    "분자 특성과 용해도의 관계를 분석했습니다. "
    "용해도 분포, 분자 특성별 비교, 상관관계 히트맵을 통해 "
    "데이터의 특징을 살펴보고 다중 선형 회귀 모델로 용해도를 예측했습니다."
)

st.info(
    "주의: 상관관계는 인과관계를 의미하지 않습니다. "
    "또한 학습 및 테스트 데이터 분리 결과는 CSV 파일의 행 순서에 영향을 받습니다."
)
