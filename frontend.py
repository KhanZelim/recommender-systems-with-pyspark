from script.recommender import Recommender
import streamlit as st

st.title("Recommendation Engine")

user = st.number_input("User ID", min_value=1)

if st.button("Get recommendations"):
    recommender = Recommender()

    recommendations = recommender.get_recommendations()
    recommendations = recommendations.filter(recommendations.userId == user)

    recommendations_list = recommendations.collect()

    for row in recommendations_list:
        st.write(f"Movie ID: {row['movieId']}, Recommendation Score: {row['rating']}, Genre: {row['genres']}")

    recommender.stop()