from script.recommender import Recommender
import streamlit as st

st.title("Recommendation Engine")

user = st.number_input("User ID", min_value=1)

if st.button("Get recommendations"):
    recommender = Recommender()

    recommendations = recommender.get_recommendations()
    recommendations = recommender.filter_user(recommendations, user)

    recommendations_list = recommendations.collect()

    for row in recommendations_list:
        st.write(f"Title: {row['title']}, Genre: {row['genres']}")

    recommender.stop()