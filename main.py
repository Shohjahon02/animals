import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import wikipedia


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

    result = wikipedia.summary(pred)

    fig =px.histogram(x=model.dls.vocab, y=probs*100)
    left_column, right_column = st.columns(2)
    left_column.plotly_chart(fig, use_container_width=True)
    right_column.write(result)