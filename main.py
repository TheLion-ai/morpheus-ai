import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import hydralit_components as hc
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier


det_input = {
        'BAT': 0,
        'EOT': 0,
        'LYT': 0,
        'MOT': 0,
        'HGB': 0,
        'MCHC': 0,
        'MCV': 0,
        'PLT': 0,
        'WBC': 0,
        'Age': 0,
        'Sex': 0
    }

prog_input = {
        'LYT': 0,
        'HGB': 0,
        'PLT': 0,
        'WBC': 0,
        'Age': 0,
        'Sex': 0
    }

det_cols1 = ["BAT", "EOT", "LYT", "MOT", "HGB"]
det_cols2 = ["MCHC", "MCV", "PLT", "WBC", "Age"]
prog_cols1 = ['LYT', 'HGB', 'PLT', 'WBC', 'Age']
prog_cols2 = []
cat_cols = ["Sex"]



st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
)


menu_data = [
    {"id": "Detection", "icon": "🩸", "label": "SARS-CoV-2 detection"},
    {"id": "Prognosis", "icon": "🩸", "label": "COVID-19 prognosis"},
]

over_theme = {"txc_inactive": "#FFFFFF"}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name="Home",
    hide_streamlit_markers=False,
    sticky_nav=True,
    sticky_mode="pinned",
)

clf_det = TabNetClassifier()
clf_det.load_model('tabnet_detection.zip')

clf_prog = TabNetClassifier()
clf_prog.load_model('tabnet_prognosis.zip')

scalar = StandardScaler()


def preprocess_sex(my_dict):
    if my_dict['Sex']=='M':
        my_dict['Sex'] = 1
    elif my_dict['Sex']=='F':
        my_dict['Sex'] = 0
    else:
        st.error("Incorrect Sex. Correct the input and try again.")
    return my_dict


def predict_det(**det_input):
        covid = False
        det_input = preprocess_sex(det_input)
        try:
            predict_arr = np.array([[float(det_input[col]) if det_input[col] else 0.0 for col in [*det_cols1, *det_cols2, *cat_cols]]])
            predict_arr = scalar.fit_transform(predict_arr)
            print(predict_arr )

            covid = clf_det.predict(predict_arr) > 0.5
            if covid:
                col2.markdown('<h1 style="color:red">COV+</h1>',  unsafe_allow_html=True)
            else:
                col2.markdown('<h1 style="color:green">COV-</h1>',  unsafe_allow_html=True)
        except:
            st.error("Incorrect data format in the form. Correct the input and try again.")


def predict_prog(**prog_input):
        care = -1

        prog_input = preprocess_sex(prog_input)
        try:
            predict_arr = np.array([[float(prog_input[col]) if prog_input[col] else None for col in [*prog_cols1, *cat_cols]]])
            predict_arr = scalar.fit_transform(predict_arr)
            print(predict_arr )
            care = clf_prog.predict(predict_arr)
            if care==0:
                col2.markdown('<h1 style="color:green">Not admitted to a hospital</h1>',  unsafe_allow_html=True)
            elif care == 1:
                col2.markdown('<h1 style="color:orange">Admitted to a hospital with moderate COVID</h1>',  unsafe_allow_html=True)
            elif care == 2:
                col2.markdown('<h1 style="color:red">Admitted to a hospital with critical COVID!</h1>',  unsafe_allow_html=True)
        except:
            st.error("Incorrect data format in the form. Correct the input and try again.")



if menu_id == 'Home':
    st.title(
        "Machine-aided detection of SARS-CoV-2 and prediction of the course of COVID-19 based on laboratory results of patients"
    )
    st.text('Welcome to the page hosting prototypes of machine-aided models for SARS-CoV-2 detection and COVID-19 prognosis based on Complete Blood Count results. Select tab in the menu bar to try out the models.')
    st.text('Disclaimer: The models presented on the page are prototypes and are not intended for clinical use.')

elif menu_id == 'Detection':

    _, col1, col2, _ = st.columns(4)
    col1.title('SARS-CoV-2 detection')
    col1.text('Press predict after filling in the form below.')
    col2.markdown("#")
    col2.markdown("#")
    col2.write("##")
    col2.write("##")

    for col in det_cols1:
        det_input[col] = col1.number_input(col)

    for col in det_cols2:
        det_input[col] = col2.number_input(col)

    for col in cat_cols:
        det_input[col] = col1.selectbox(col, ('F', 'M'))

    col2.write("##")
    col2.write("##")

    col1.button("PREDICT", on_click=predict_det, kwargs=det_input)


elif menu_id == 'Prognosis':
    _, col1, col2, _ = st.columns(4)
    col1.title('SARS-CoV-2 detection')
    col1.text('Press predict after filling in the form below.')
    col2.markdown("#")
    col2.markdown("#")
    col2.write("##")
    col2.write("##")

    for col in prog_cols1:
        prog_input[col] = col1.number_input(col)
        col2.text("")

    for col in cat_cols:
        prog_input[col] = col1.selectbox(col, ('F', 'M'))
        col2.text("")

    col2.write("##")
    col2.write("##")

    col1.button("PREDICT", on_click=predict_prog, kwargs=prog_input)
