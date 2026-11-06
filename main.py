from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# 기본 설정
st.set_page_config(
    page_title="화합물의 용해도 데이터 그래프 도감 1 - 분자 특성",
    page_icon="🧪",
    layout="wide",
)

APP_TITLE = "화합물의 용해도 데이터 그래프 도감 1 - 분자 특성"
DATA_FILE = Path(__file__).parent / "AqSolDB_missing_cleaned.csv"

FEATURES = {
    "분자량": "MolWt",
    "소수성": "MolLogP",
    "수소 결합 수용체 수": "NumHAcceptors",
    "수소 결합 공여체 수": "NumHDonors",
    "회전 가능한 결합 수": "NumRotatableBonds",
    "극성 표면적": "TPSA",
}

REQUIRED_COLUMNS = ["Solubility", *FEATURES.values()]


# 1. 데이터 불러오기
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV 파일에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    if data.empty:
        raise ValueError(
            "숫자로 변환할 수 있는 유효한 데이터가 없습니다."
        )

    return data


# 데이터 기본 정보
def show_data_information(data):
    st.subheader("데이터 기본 정보")

    average_solubility = data["Solubility"].mean()
    minimum_solubility = data["Solubility"].min()
    maximum_solubility = data["Solubility"].max()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "전체 화합물 데이터 개수",
        f"{len(data):,}개"
    )

    col2.metric(
        "분자 특성 개수",
        f"{len(FEATURES)}개"
    )

    col3.metric(
        "용해도 평균",
        f"{average_solubility:.3f}"
    )

    col4.metric(
        "용해도 최솟값 ~ 최댓값",
        f"{minimum_solubility:.3f} ~ {maximum_solubility:.3f}"
    )


# 1번 구역: 분자 특성과 용해도의 관계
def describe_relationship(data, feature_column, feature_name):

    paired_data = data[
        [feature_column, "Solubility"]
    ].dropna()

    if len(paired_data) < 2:
        return "관계를 계산하기에 데이터가 충분하지 않습니다."

    correlation = paired_data[
        feature_column
    ].corr(paired_data["Solubility"])

    if pd.isna(correlation):
        return "값의 변화가 거의 없어 상관관계를 계산할 수 없습니다."

    if correlation >= 0.7:
        relation = "비교적 뚜렷한 양의 선형 상관관계"

    elif correlation >= 0.3:
        relation = "어느 정도의 양의 선형 상관관계"

    elif correlation > -0.3:
        relation = "뚜렷한 양의 또는 음의 선형 상관관계가 약한 관계"

    elif correlation > -0.7:
        relation = "어느 정도의 음의 선형 상관관계"

    else:
        relation = "비교적 뚜렷한 음의 선형 상관관계"

    return (
        f"이 데이터에서는 {feature_name}과(와) 용해도 사이에 "
        f"{relation}가 나타납니다(상관계수 r = {correlation:.2f}). "
        "이는 두 값이 함께 변하는 경향을 뜻하며, "
        "원인과 결과를 증명하는 것은 아닙니다."
    )


def show_feature_relationship(data):

    st.subheader("1번 구역: 분자 특성과 용해도의 관계")

    st.write(
        "아래 목록에서 살펴보고 싶은 분자 특성을 선택해 주세요."
    )

    selected_name = st.selectbox(
        "분자 특성 선택",
        options=list(FEATURES.keys()),
        index=0,
    )

    selected_column = FEATURES[selected_name]

    chart_data = data[
        [selected_column, "Solubility"]
    ].dropna()

    figure = px.scatter(
        chart_data,
        x=selected_column,
        y="Solubility",
        opacity=0.45,
        labels={
            selected_column: f"{selected_name} ({selected_column})",
            "Solubility": "용해도 (Solubility)",
        },
        title=f"{selected_name}과 용해도의 관계",
        hover_data={
            selected_column: ":.4f",
            "Solubility": ":.4f",
        },
    )

    figure.update_traces(
        marker={"size": 6},
        hovertemplate=(
            f"{selected_name}: %{{x:.4f}}<br>"
            "용해도: %{y:.4f}<extra></extra>"
        ),
    )

    figure.update_layout(
        title_x=0.5,
        xaxis_title=f"{selected_name} ({selected_column})",
        yaxis_title="용해도 (Solubility)",
        hovermode="closest",
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True
    )

    st.markdown("### 이 그래프로 알 수 있는 것")

    st.write(
        describe_relationship(
            data,
            selected_column,
            selected_name
        )
    )


# 2번 구역: 용해도 데이터 분포
def show_solubility_distribution(data):

    st.subheader("2번 구역: 용해도 데이터 분포")

    st.info(
        "추후 용해도 값이 어떤 범위에 많이 분포하는지 "
        "확인하는 히스토그램 등을 추가할 수 있습니다."
    )


# 3번 구역: 분자 특성별 용해도 비교
def show_feature_comparison(data):

    st.subheader("3번 구역: 분자 특성별 용해도 비교")

    st.info(
        "추후 분자 특성의 구간을 나누어 "
        "각 구간의 용해도를 비교하는 그래프 등을 추가할 수 있습니다."
    )


# 4번 구역: 추가 그래프 공간
def show_future_section():

    st.subheader("4번 구역: 추가 그래프 공간")

    st.info(
        "새로운 분석 아이디어가 생기면 "
        "이 구역에 그래프 함수를 추가하면 됩니다."
    )


# 앱 실행
def main():

    st.title(APP_TITLE)

    st.caption(
        "화합물의 분자 특성과 용해도 사이의 관계를 "
        "데이터로 살펴보는 앱입니다."
    )

    try:
        data = load_data(str(DATA_FILE))

    except FileNotFoundError:

        st.error(
            "데이터 파일을 찾을 수 없습니다. "
            "'AqSolDB_missing_cleaned.csv' 파일을 "
            "main.py와 같은 폴더에 넣었는지 확인해 주세요."
        )

        st.stop()

    except pd.errors.EmptyDataError:

        st.error(
            "CSV 파일에 데이터가 없습니다. "
            "파일 내용을 확인해 주세요."
        )

        st.stop()

    except (pd.errors.ParserError, UnicodeDecodeError):

        st.error(
            "CSV 파일을 읽는 중 문제가 발생했습니다. "
            "파일 형식과 인코딩을 확인해 주세요."
        )

        st.stop()

    except ValueError as error:

        st.error(
            f"데이터 형식을 확인해 주세요. {error}"
        )

        st.stop()

    except Exception as error:

        st.error(
            f"데이터를 불러오지 못했습니다: {error}"
        )

        st.stop()

    show_data_information(data)

    st.divider()

    show_feature_relationship(data)

    st.divider()

    show_solubility_distribution(data)

    st.divider()

    show_feature_comparison(data)

    st.divider()

    show_future_section()


if __name__ == "__main__":
    main()
