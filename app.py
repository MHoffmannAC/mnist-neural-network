import streamlit as st


def main() -> None:
    st.set_page_config(layout="wide")
    st.set_option("client.showErrorDetails", value=False)

    page = st.navigation(
        {
        "General Pages": [ 
            st.Page(
                "pages/start.py",
                title="Homepage",
                icon=":material/house:",
                default=True,
            ),
            st.Page(
                "pages/data.py",
                title="Data Sets",
                icon=":material/description:",
            ),
        ],
            "Unsupervised Machine Learning": [
                st.Page(
                    "pages/uml-kmeans.py",
                    title="KMeans",
                    icon=":material/home:",
                ),
                st.Page(
                    "pages/uml-dbscan.py",
                    title="DBSCAN",
                    icon=":material/blur_on:",
                ),
                st.Page(
                    "pages/uml-gmm.py",
                    title="GMM",
                    icon=":material/bubble_chart:",
                ),
            ],
            "Supervised Machine Learning": [
                st.Page(
                    "pages/sml-tree.py",
                    title="DecisionTree",
                    icon=":material/account_tree:",
                ),
                st.Page(
                    "pages/sml-svm.py",
                    title="SVM",
                    icon=":material/straighten:",
                ),
                st.Page(
                    "pages/sml-log_reg.py",
                    title="LogisticRegression",
                    icon=":material/show_chart:",
                ),
            ],
            "Time Series": [
                st.Page(
                    "pages/ts-sarima.py",
                    title="AR, ARIMA, SARIMA",
                    icon=":material/query_stats:",
                ),
                st.Page(
                    "pages/ts-prophet.py",
                    title="Prophet",
                    icon=":material/auto_awesome:",
                ),
            ],
            "Deep Learning": [
                st.Page(
                    "pages/dl-nn_mnist.py",
                    title="Neural Network",
                    icon=":material/memory:",
                ),
                st.Page(
                    "pages/dl-cnn.py",
                    title="Convolutional Neural Network",
                    icon=":material/image:",
                ),
            ],
        },
    )

    page.run()

if __name__ == "__main__":
    main()