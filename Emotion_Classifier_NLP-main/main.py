import streamlit as st 
import altair as alt

import numpy as np
import pandas as pd


import joblib
from joblib import load
import pickle

pipe = joblib.load(open('models/model_pickel', 'rb'))

def predict_emotions(docx):
    results = pipe.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe.predict_proba([docx])
    return results

def main():
    st.title("Emotion Classifier")
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Emotion in Text")

        with st.form(key = "emotion_clf_form"):
            raw_text = st.text_area("Type Area")
            submit_text = st.form_submit_button(label = "Submit")

        if submit_text:
            col1, col2 = st.columns(2)


            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)
                

                st.success("Confidence")
                st.write("Confidence: {}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                # st.write(probability)

                prob_df = pd.DataFrame(probability, columns = pipe.classes_)
                # st.write(prob_df.T)
                
                prob_df_clean = prob_df.T.reset_index()
                prob_df_clean.columns = ["emotions","probability"]

                fig = alt.Chart(prob_df_clean).mark_bar().encode(x = "emotions", y = "probability", color = "emotions")
                st.altair_chart(fig, use_container_width=True)

    else:
        st.subheader("About")
        st.write("This predicts the emotion or sentiment of the sentence or dialogue for the emotions such as 'anger', 'fear', 'joy', 'love', 'sadness', 'surprise'.")
        st.write("The probability that it can predict is 70%")
        st.write("It cannot also predict out of vocabulary(oov) words")
        st.caption("The data used to make this is from the Kaggle")

if __name__ == "__main__":
    main()
