
import pandas as pd
import streamlit as st
import plotly.express as px


#loading data
@st.cache_data
def load_data():
    df = pd.read_csv("tweets_with_linguistics.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    return df

df = load_data()
st.set_page_config(page_title="Crisis emotion dashboard")


#sidebar
st.sidebar.title("Filters")
events = sorted(df["event"].unique())
selected_event = st.sidebar.selectbox("Select event", events)

emotion_cols = [
    "plutchik_joy","plutchik_sadness","plutchik_fear","plutchik_anger",
    "plutchik_surprise","plutchik_trust","plutchik_disgust","plutchik_anticipation"
]
selected_emotions = st.sidebar.multiselect(
    "Select emotions to display", emotion_cols, default=emotion_cols
)


#main graph
event_df = df[df["event"] == selected_event].copy()
event_df["date"] = event_df["timestamp"].dt.date
agg = event_df.groupby("date")[["vader_score","empathy_score"] + emotion_cols].mean().reset_index()

st.title(f"Emotion and empathy timeline: {selected_event}")

fig_sent = px.line(
    agg, x="date", y=["vader_score","empathy_score"],
    title="Sentiment vs empathy over time",
    markers=True
)
fig_sent.update_layout(yaxis_title="Average score", legend_title="Metric")
st.plotly_chart(fig_sent, use_container_width=True)

#emotion graphs
fig_emo = px.line(
    agg, x="date", y=selected_emotions,
    title="Intensity of emotions over time",
    markers=False
)
fig_emo.update_layout(yaxis_title="Average intensity", legend_title="Emotion")
st.plotly_chart(fig_emo, use_container_width=True)

#stats
st.subheader("Event statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Tweets", len(event_df))
with col2:
    st.write("**Date range**")
    st.markdown(f"{event_df['date'].min():%Y-%m-%d} → {event_df['date'].max():%Y-%m-%d}")
col3.metric("Dominant emotion", event_df['plutchik_top'].mode()[0] if not event_df.empty else "N/A")

st.caption("Data from CrisisLexT26, available at https://github.com/sajao/CrisisLex/tree/master/data/CrisisLexT26/")
