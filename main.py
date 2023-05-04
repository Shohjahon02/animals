import streamlit as st
from fastai.vision.all import *
import plotly.express as px


st.set_page_config(page_title="Jonivorlarni klassifikatsiya qiluvchi model",
                   page_icon=":bar_chart",
                   layout="wide")
 
st.title(":bar_chart: Jonivorlarni klassifikatsiya qiluvchi model")
st.markdown("##")

file = st.file_uploader("Rasm yuklash", type=('png', 'jpeg', 'gif', 'svg', 'jpg'))



if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('animals.pkl')


    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')


    fig =px.histogram(x=model.dls.vocab, y=probs*100)
    st.plotly_chart(fig)
