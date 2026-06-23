import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.predict import predict_sms

st.set_page_config(
    page_title="Premium Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
"""
<style>

html,body,[class*="css"]{

    background:#050816;

}

.main{

    background:#050816;

}

h1,h2,h3,h4,h5,h6,p,label{

    color:white;

}

.hero{

    background:linear-gradient(90deg,#4F46E5,#7C3AED,#2563EB);

    padding:30px;

    border-radius:20px;

    text-align:center;

    margin-bottom:25px;

    box-shadow:0px 0px 20px rgba(80,80,255,.4);

}

.hero h1{

    color:white;

    font-size:52px;

}

.hero p{

    color:#E5E7EB;

    font-size:22px;

}

.metricCard{

    background:#111827;

    padding:18px;

    border-radius:15px;

    text-align:center;

    border:1px solid #374151;

}

.metricTitle{

    color:#9CA3AF;

    font-size:18px;

}

.metricValue{

    color:white;

    font-size:30px;

    font-weight:bold;

}

.resultSpam{

    background:#3B0A0A;

    padding:25px;

    border-radius:15px;

    border-left:8px solid red;

}

.resultHam{

    background:#053B23;

    padding:25px;

    border-radius:15px;

    border-left:8px solid lime;

}

.footer{

    text-align:center;

    color:#9CA3AF;

    margin-top:60px;

}

textarea{

    font-size:20px !important;

}

</style>
""",
unsafe_allow_html=True
)

st.markdown(

"""
<div class="hero">

<h1>📧 Premium Email Spam Detector</h1>

<p>

AI Powered NLP + Machine Learning + TF-IDF + Naive Bayes

</p>

</div>

""",

unsafe_allow_html=True

)

st.sidebar.title("📊 Dashboard")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("### 🤖 Algorithm")

st.sidebar.write("Multinomial Naive Bayes")

st.sidebar.markdown("### 🧠 NLP")

st.sidebar.write("TF-IDF Vectorizer")

st.sidebar.markdown("### 📚 Dataset")

st.sidebar.write("SMS Spam Collection")

st.sidebar.markdown("### 🎯 Accuracy")

st.sidebar.metric(

"Accuracy",

"97.58%"

)

st.sidebar.markdown("---")

st.sidebar.markdown(

"""
### 💡 Tips

✔ Genuine messages usually contain normal conversations.

✔ Spam messages often contain:

• Win money

• Click here

• Claim reward

• Urgent offer

• Free gift

"""
)

col1,col2,col3,col4=st.columns(4)

with col1:

    st.markdown(

    """

    <div class="metricCard">

    <div class="metricTitle">

    Accuracy

    </div>

    <div class="metricValue">

    97.58%

    </div>

    </div>

    """,

    unsafe_allow_html=True

    )

with col2:

    st.markdown(

    """

    <div class="metricCard">

    <div class="metricTitle">

    Dataset

    </div>

    <div class="metricValue">

    5574

    </div>

    </div>

    """,

    unsafe_allow_html=True

    )

with col3:

    st.markdown(

    """

    <div class="metricCard">

    <div class="metricTitle">

    Classes

    </div>

    <div class="metricValue">

    2

    </div>

    </div>

    """,

    unsafe_allow_html=True

    )

with col4:

    st.markdown(

    """

    <div class="metricCard">

    <div class="metricTitle">

    Model

    </div>

    <div class="metricValue">

    NB

    </div>

    </div>

    """,

    unsafe_allow_html=True

    )

st.write("")

st.subheader("✉ Enter Email / SMS")

message=st.text_area(

"",

height=180,

placeholder="Type your email or SMS here..."

)

predictButton=st.button(

"🔍 Predict Message",

use_container_width=True

)

if predictButton:

    if message.strip() == "":

        st.warning("⚠ Please enter an Email or SMS message.")

    else:

        prediction, spam_probability, ham_probability = predict_sms(message)

        st.write("")

        if prediction == "SPAM":

            st.markdown(
            f"""

            <div class="resultSpam">

            <h2>🚨 SPAM MESSAGE DETECTED</h2>

            <h3>Risk Level : HIGH</h3>

            <p>

            This message looks suspicious.

            Avoid clicking unknown links or sharing personal information.

            </p>

            </div>

            """,

            unsafe_allow_html=True

            )

        else:

            st.markdown(
            f"""

            <div class="resultHam">

            <h2>✅ GENUINE MESSAGE</h2>

            <h3>Risk Level : LOW</h3>

            <p>

            This message appears to be safe.

            </p>

            </div>

            """,

            unsafe_allow_html=True

            )

        st.write("")

        left,right=st.columns(2)

        with left:

            st.subheader("📊 Prediction Confidence")

            confidence=max(spam_probability,ham_probability)

            st.progress(int(confidence*100))

            st.metric(

                "Confidence",

                f"{confidence*100:.2f}%"

            )

        with right:

            df=pd.DataFrame({

                "Class":[

                    "Spam",

                    "Ham"

                ],

                "Probability":[

                    spam_probability,

                    ham_probability

                ]

            })

            st.subheader("📈 Probability")

            st.bar_chart(

                df.set_index("Class")

            )

st.divider()

st.subheader("📋 Try Example Messages")

example1="""
Hi Arpan,

Your order has been delivered successfully.

Thank you for shopping with us.
"""

example2="""
Congratulations!!

You have won ₹50,000 cash prize.

Click here now to claim your reward.

Limited Offer!!
"""

example3="""
Dear Student,

Tomorrow's class will begin at 10:30 AM.

Please bring your laboratory notebook.
"""

c1,c2,c3=st.columns(3)

with c1:

    st.info("✅ Genuine")

    st.code(example1)

with c2:

    st.error("🚨 Spam")

    st.code(example2)

with c3:

    st.success("✅ Genuine")

    st.code(example3)

st.divider()

st.subheader("📊 Dataset Overview")

dataset=pd.read_csv(

    "data/spam.csv",

    encoding="latin1"

)

dataset=dataset.iloc[:,0:2]

dataset.columns=[

    "label",

    "message"

]

spamCount=(dataset["label"]=="spam").sum()

hamCount=(dataset["label"]=="ham").sum()

pieData=pd.DataFrame({

    "Label":[

        "Ham",

        "Spam"

    ],

    "Count":[

        hamCount,

        spamCount

    ]

})

chart1,chart2=st.columns(2)

with chart1:

    st.subheader("📈 Spam vs Ham Distribution")

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(

        pieData["Count"],

        labels=pieData["Label"],

        autopct="%1.1f%%",

        startangle=90,

        explode=(0,0.08)

    )

    ax.axis("equal")

    st.pyplot(fig)

with chart2:

    st.subheader("📊 Dataset Statistics")

    st.metric(

        "Total Messages",

        len(dataset)

    )

    st.metric(

        "Ham Messages",

        hamCount

    )

    st.metric(

        "Spam Messages",

        spamCount

    )

dataset["length"]=dataset["message"].apply(len)

st.divider()

st.subheader("📏 Message Length Analysis")

fig2,ax2=plt.subplots(figsize=(10,4))

ax2.hist(

    dataset["length"],

    bins=40

)

ax2.set_xlabel(

    "Characters"

)

ax2.set_ylabel(

    "Frequency"

)

st.pyplot(fig2)

st.divider()

st.subheader("☁️ Word Clouds")

spam_text=" ".join(

    dataset[

        dataset["label"]=="spam"

    ]["message"]

)

ham_text=" ".join(

    dataset[

        dataset["label"]=="ham"

    ]["message"]

)

spamCloud=WordCloud(

    width=900,

    height=500,

    background_color="black"

).generate(

    spam_text

)

hamCloud=WordCloud(

    width=900,

    height=500,

    background_color="white"

).generate(

    ham_text

)

left,right=st.columns(2)

with left:

    st.subheader("🚨 Spam Word Cloud")

    fig3,ax3=plt.subplots(figsize=(6,4))

    ax3.imshow(

        spamCloud

    )

    ax3.axis("off")

    st.pyplot(fig3)

with right:

    st.subheader("✅ Ham Word Cloud")

    fig4,ax4=plt.subplots(figsize=(6,4))

    ax4.imshow(

        hamCloud

    )

    ax4.axis("off")

    st.pyplot(fig4)

st.divider()

st.subheader("💡 About this Project")

st.info(

"""

This project detects whether an Email or SMS is **Spam** or **Genuine**

using Natural Language Processing (NLP) and Machine Learning.



### Technologies Used

• Python

• Pandas

• Scikit-Learn

• TF-IDF Vectorizer

• Multinomial Naive Bayes

• Streamlit



### Features

✅ Real-time Prediction

✅ Prediction Confidence

✅ Spam/Ham Probability

✅ Word Cloud Visualization

✅ Dataset Analytics

✅ Modern Dashboard

"""

)

st.divider()

footer="""

<div class="footer">

<h3>

📧 Premium Email Spam Detector

</h3>

<p>

AICTE Oasis Infobyte Internship Project

</p>

<p>

Machine Learning • NLP • Streamlit

</p>

<hr>

<p>

Developed using Python, Scikit-Learn & Streamlit

</p>

</div>

"""

st.markdown(

footer,

unsafe_allow_html=True

)