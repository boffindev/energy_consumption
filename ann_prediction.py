import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from datetime import date, datetime
import joblib


def get_actual_consumption(date, df_ann):
    input_data = df_ann.loc[df_ann['date_time'].dt.date == pd.to_datetime(date).date()]
    actual = input_data['kwh_flux_cmb_win'].values[0]
    return actual


def get_outputs_ann(date):
    predicted = predict_energy_consumption(date)
    actual = get_actual_consumption(date)

    outputs = [predicted, actual]
    labels = ['ANN Prediction', 'Actual Consumption']

    data = {'kWh': outputs, 'Labels': labels}
    df = pd.DataFrame(data)
    return df


def predict_energy_consumption(date, df_ann, model_ann):
    input_data = df_ann.loc[df_ann['date_time'].dt.date == pd.to_datetime(date).date()]
    input_data_scaled = scaler.transform(input_data[features_yearly])
    prediction = ann_model.predict(input_data_scaled)
    return prediction[0][0]


def model_error(actual, prediction):
    r2 = r2_score(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    mae = mean_absolute_error(actual, prediction)

    return r2, mse, mae


def get_season_split(date):
    if date.month in [11, 12, 1, 2, 3]:
        return 'winter'
    else:
        return 'summer'


def linear_regression_prediction(date_lr, models, df_lr):
    date_lr = datetime.strptime(date_lr, "%Y-%m-%d")
    winter_feature = ['T', 'is_weekend', 'is_holiday', 'SQ', 'month_11', 'month_12', 'month_1', 'month_2', 'month_3']
    summer_feature = ['T', 'is_weekend', 'FF', 'R', 'T10N', 'T10X', 'month_4', 'month_5', 'month_6', 'month_7',
                      'month_8', 'month_9', 'month_10']

    lr_features = {'winter': winter_feature,
                   'summer': summer_feature}

    season = get_season_split(date_lr)
    features = lr_features[season]

    ### Single Prediction
    specific_season = df_lr.loc[df_lr['date_time'].dt.date == date_lr.date(), 'season'].values[0]
    X_specific = df_lr[(df_lr['date_time'].dt.date == date_lr.date()) & (df_lr['season'] == specific_season)][features]
    y_specific_pred = models[season].predict(X_specific)

    ### Overall Predictive Power
    combined_predictions = pd.DataFrame()

    for season in ['winter', 'summer']:
        features = lr_features[season]

        X_test = df_lr[(df_lr['date_time'].dt.year == 2022) & (df_lr['season'] == season)][features]
        y_pred = models[season].predict(X_test)

        temp_df = pd.DataFrame(y_pred, columns=['predicted'], index=X_test.index)
        temp_df['date_time'] = df_lr.loc[X_test.index, 'date_time']

        combined_predictions = pd.concat([combined_predictions, temp_df])

    combined_predictions.sort_values('date_time', inplace=True)

    y_actual = df_lr[df_lr['date_time'].dt.year == 2022]['kwh_flux_cmb_win']

    mse_lr = mean_squared_error(y_actual, combined_predictions['predicted'])
    r2_lr = r2_score(y_actual, combined_predictions['predicted'])
    mae_lr = mean_absolute_error(y_actual, combined_predictions['predicted'])

    return_dict = {'prediction': y_specific_pred[0][0],
                   'mse': mse_lr,
                   'mae': mae_lr,
                   'r2': r2_lr}

    return return_dict


######### ANN Model ###############################

def ann_prediction(date_ann, df_ann, ann_model):

    features_yearly = ['T', 'is_weekend', 'is_holiday', 'SQ', 'R']

    X_train = df_ann[df_ann['date_time'].dt.year.isin([2020, 2021])][features_yearly]
    X_test = df_ann[df_ann['date_time'].dt.year == 2022][features_yearly]
    y_train = df_ann[df_ann['date_time'].dt.year.isin([2020, 2021])][['kwh_flux_cmb_win']]
    y_test = df_ann[df_ann['date_time'].dt.year == 2022][['kwh_flux_cmb_win']]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred = ann_model.predict(X_test_scaled)

    r2_ann = r2_score(y_test, y_pred)
    mse_ann = mean_squared_error(y_test, y_pred)
    mae_ann = mean_absolute_error(y_test, y_pred)

    ### predict for a given day

    input_data = df_ann.loc[df_ann['date_time'].dt.date == pd.to_datetime(date_ann).date()]
    input_data_scaled = scaler.transform(input_data[features_yearly])
    prediction = ann_model.predict(input_data_scaled)

    return_dict = {'prediction': prediction[0][0],
                   'mse': mse_ann,
                   'mae': mae_ann,
                   'r2': r2_ann}
    return return_dict

