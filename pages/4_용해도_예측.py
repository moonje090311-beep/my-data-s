import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="용해도 예측", page_icon="🔬", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "AqSolDB_missing_cleaned.csv"

FEATURES = [
    "MolWt",
    "MolLogP",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "TPSA"
]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")

st.title("🔬 4. 용해도 예측")
st.write("분자 특성을 이용하여 화합물의 용해도를 예측합니다.")

try:
    df = load_data()

    for col in FEATURES + ["Solubility"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    data = df.dropna(subset=["Solubility"]).copy()

    st.subheader("1. 예측에 사용할 분자 특성 선택")

    selected_features = st.multiselect(
        "모델 학습에 사용할 특성을 선택하세요.",
        FEATURES,
        default=FEATURES
    )

    if len(selected_features) == 0:
        st.warning("최소 한 개 이상의 분자 특성을 선택해주세요.")
        st.stop()

    st.divider()

    st.subheader("2. 학습 데이터와 테스트 데이터 설정")

    test_size = st.slider(
        "테스트 데이터 비율(%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

    X = data[selected_features]
    y = data["Solubility"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size / 100,
        random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    st.success("모델 학습이 완료되었습니다.")

    st.divider()

    st.subheader("3. 모델 성능 평가")

    col1, col2, col3 = st.columns(3)

    col1.metric("R² 결정계수", f"{r2:.4f}")
    col2.metric("MAE 평균 절대 오차", f"{mae:.4f}")
    col3.metric("RMSE 평균 제곱근 오차", f"{rmse:.4f}")

    st.write("""
    - R²: 실제 용해도 데이터의 변동을 모델이 얼마나 설명하는지 나타냅니다.
    - MAE: 실제값과 예측값의 평균적인 차이입니다.
    - RMSE: 예측 오차를 제곱한 뒤 평균을 내고 제곱근을 구한 값입니다.

    MAE와 RMSE는 값이 작을수록 예측 오차가 작다는 의미입니다.
    """)

    st.divider()

    st.subheader("4. 실제 용해도와 예측 용해도 비교")

    result = pd.DataFrame({
        "실제 용해도": y_test.values,
        "예측 용해도": predictions
    })

    fig1 = px.scatter(
        result,
        x="실제 용해도",
        y="예측 용해도",
        title="실제값과 예측값 비교",
        opacity=0.6
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.dataframe(
        result.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("5. 새로운 화합물의 용해도 예측")

    st.write(
        "아래에 새로운 화합물의 분자 특성 값을 입력하면 용해도를 예측합니다."
    )

    input_values = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(selected_features):
        with (col1 if i % 2 == 0 else col2):
            median_value = float(data[feature].median())

            input_values[feature] = st.number_input(
                feature,
                value=median_value,
                format="%.4f"
            )

    if st.button("용해도 예측하기", type="primary"):
        input_df = pd.DataFrame([input_values])

        predicted_value = model.predict(input_df)[0]

        st.success(
            f"예측된 용해도는 {predicted_value:.4f}입니다."
        )

        st.caption(
            "이 결과는 선택한 분자 특성과 학습 데이터에 기반한 모델의 예측값이며, "
            "실제 실험 결과와 다를 수 있습니다."
        )

except FileNotFoundError:
    st.error("CSV 파일을 찾을 수 없습니다.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
