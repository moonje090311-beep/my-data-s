import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="화합물 용해도 분석",
    page_icon="🧪",
    layout="wide"
)

DATA_PATH = Path(__file__).parent / "AqSolDB_missing_cleaned.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8")

st.title("🧪 화합물 용해도 데이터 분석")
st.write("분자 특성을 이용하여 화합물의 용해도를 탐색하고 예측하는 프로젝트입니다.")

try:
    df = load_data()

    st.success("데이터를 성공적으로 불러왔습니다.")

    st.subheader("프로젝트 소개")

    st.write("""
    화합물의 용해도는 분자의 구조와 여러 물리화학적 특성에 영향을 받습니다.

    이 프로젝트에서는 AqSolDB 데이터를 활용하여
    분자량, 분배계수, 수소 결합 특성, 극성 표면적 등이
    화합물의 용해도와 어떤 관계가 있는지 분석합니다.

    또한 분자 특성을 입력하여 새로운 화합물의 용해도를 예측하는
    간단한 머신러닝 모델을 만들어 봅니다.
    """)

    st.divider()

    st.subheader("데이터 기본 정보")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("전체 화합물 수", f"{len(df):,}개")

    with col2:
        st.metric("전체 변수 수", f"{len(df.columns)}개")

    with col3:
        if "Solubility" in df.columns:
            solubility = pd.to_numeric(
                df["Solubility"], errors="coerce"
            )
            st.metric(
                "평균 용해도",
                f"{solubility.mean():.3f}"
            )

    st.divider()

    st.subheader("분석에 사용되는 변수")

    variable_info = pd.DataFrame({
        "변수": [
            "Solubility",
            "MolWt",
            "MolLogP",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "TPSA"
        ],
        "의미": [
            "화합물의 용해도",
            "분자의 분자량",
            "분자의 분배계수",
            "수소 결합 수용체 수",
            "수소 결합 공여체 수",
            "회전 가능한 결합 수",
            "분자의 극성 표면적"
        ]
    })

    st.dataframe(
        variable_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("데이터 미리보기")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.caption(
        "왼쪽 메뉴에서 탐색, 상관관계 분석, 분자 특성 비교, 용해도 예측을 선택할 수 있습니다."
    )

except FileNotFoundError:
    st.error(
        "CSV 파일을 찾을 수 없습니다. "
        "main.py와 같은 폴더에 AqSolDB_missing_cleaned.csv를 넣어주세요."
    )

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
