import joblib
import streamlit as st
import plotly_express as px
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import date

# custom imports
import ann_prediction


def comparison_df(lr_output, ann_output, actual):
    predicted_ann = ann_output['prediction']
    predicted_lr = lr_output['prediction']

    outputs = [predicted_ann, predicted_lr, actual]
    labels = ['ANN Prediction', 'LR Prediction', 'Actual']

    data = {'kWh': outputs, 'Labels': labels}
    df = pd.DataFrame(data)
    return df


st.title("Energy usage for buildings")

# Load data / models
daily_df = pd.read_pickle('daily_df.pkl')
daily_df_ann = pd.read_pickle('daily_df_ann.pkl')
ann_model = load_model('ann_model.h5')

daily_df = pd.read_pickle('daily_df.pkl')
lr_winter = joblib.load('lr_winter.joblib')
lr_summer = joblib.load('lr_summer.joblib')

lr_models = {'winter': lr_winter,
             'summer': lr_summer}

# Set the app layout
col1, col2 = st.columns([3, 1])

### Input
# Get the selected date from the restricted calendar input
# Set the minimum and maximum allowed dates
min_date = date(2022, 1, 1)
max_date = date(2022, 12, 31)
selected_date = col1.date_input("Select a date", value=min_date, min_value=min_date, max_value=max_date)
selected_building = col1.selectbox("Select a building", (1, 2, 3, 4))

ann_dict = ann_prediction.ann_prediction(date_ann=selected_date,
                                         df_ann=daily_df_ann,
                                         ann_model=ann_model)
lr_dict = ann_prediction.linear_regression_prediction(date_lr=selected_date.strftime("%Y-%m-%d"),
                                                      models=lr_models,
                                                      df_lr=daily_df)

ground_truth = ann_prediction.get_actual_consumption(date=selected_date, df_ann=daily_df_ann)

### Visualizaitons
with col1:
    plot_data = comparison_df(lr_dict, ann_dict, ground_truth)
    fig = px.bar(data_frame=plot_data, x='Labels', y='kWh',
                 color='Labels',
                 color_discrete_map={'ANN Prediction': 'blue',
                                     'LR Prediction': 'blue',
                                     'Actual': 'green'})
    st.plotly_chart(fig, use_container_width=True)


with col2:
    r2_ann = ann_dict['r2']
    mae_ann = ann_dict['mae']
    mse_ann = ann_dict['mse']

    r2_lr = lr_dict['r2']
    mae_lr = lr_dict['mae']
    mse_lr = lr_dict['mse']

    st.markdown("##### ANN Model")
    st.write("- R2 score:", round(r2_ann, 2))
    st.write("- MAE:", round(mae_ann, 2))
    st.write("- MSE:", round(mse_ann, 2))

    st.markdown("##### Linear Regression Model")
    st.write("- R2 score:", round(r2_lr, 2))
    st.write("- MAE:", round(mae_lr, 2))
    st.write("- MSE:", round(mse_lr, 2))
