import streamlit as st


def main() -> None:
    st.set_page_config(layout="wide")
    st.set_option("client.showErrorDetails", value=False)

    with st.sidebar:
        st.page_link("https://nfl-gameday.streamlit.app/", label="NFL Gameday Analyzer", icon=":material/sports_football:")
        st.page_link("https://nfl-ds-challenges.streamlit.app/", label="NFL DataScience Challenges", icon=":material/sports_football:")
        st.page_link("https://machine-learning-playgrounds.streamlit.app/", label="Other ML Playgrounds", icon=":material/person_play:")

    page = st.navigation(
        [
            st.Page(
                "pages/start.py",
                title="Homepage",
                icon=":material/house:",
                default=True,
            ),
            st.Page(
                "pages/data.py",
                title="Data",
                icon=":material/description:",
            ),
            st.Page(
                "pages/ts-prophet.py",
                title="Prophet",
                icon=":material/auto_awesome:",
            ),
        ],
        expanded=True,
    )

    page.run()


if __name__ == "__main__":
    main()
