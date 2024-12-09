from algorithm import *
from utils import read_excel_data, create_window

# Create rolling windows and check anomalies
def check_anomalies_in_windows(detected_anomaly_data):
    window_data = create_window(data=detected_anomaly_data)
    for window in window_data:
        anomaly_panel = None
        # Use list comprehension instead of apply for better performance
        window['panel_anomaly'] = [
            x.split(',') if x is not None else ['no_anomaly']
            for x in window['panel_anomaly']
        ]
        # Convert lists to sets
        sets_value = [set(x) for x in window['panel_anomaly']]
        # Flatten the list of sets to get unique values
        anomaly_panel = list(set.intersection(*sets_value))
        if len(anomaly_panel) > 0 and 'no_anomaly' not in anomaly_panel and '' not in anomaly_panel:
            print(f"Thời gian từ {window.iloc[0]['datetime']} tới {window.iloc[-1]['datetime']} phát hiện bất thường ở các panel {anomaly_panel}")
            anomaly_panel.insert(0, 'datetime')
            print(window[anomaly_panel])
            nomal_panel = [col for col in window.columns if col not in anomaly_panel and col not in ['panel_anomaly']]
            print(window[nomal_panel])

def detect_anomaly_value_in_row(data):
    for idx, row in data.iterrows():
        list_anomaies_panel = []
        row_values = row[data.columns[1:-1]]  # Extracting panel data only, excluding the first and last columns
        if row_values.sum() > 0:
            non_zero_data = row_values[row_values > 0]
            zero_outliers_panel = row_values[row_values == 0].index.tolist()
            negative_outliers_panel = row_values[row_values < 0].index.tolist()

            # Assuming this returns a list of indices (column names) from the data
            non_zero_outliers = detect_with_mean_and_std(non_zero_data, threshold = 3)

            # non_zero_outliers = detect_with_iqr(non_zero_data)

            # Convert these into column names which indicate panel names
            non_zero_outliers_panel = ','.join([col for col in data.columns if row[col] in non_zero_outliers])
            list_anomaies_panel = zero_outliers_panel + negative_outliers_panel
            list_anomaies_panel.append(non_zero_outliers_panel)
            if len(list_anomaies_panel)>0:
                data.loc[idx, 'panel_anomaly'] = ','.join(list_anomaies_panel)
    return data

if __name__ == "__main__":
    data = read_excel_data(data_path='data/DataMakeOurlies.xlsx')
    # data = data[(data['datetime'] >= '2024-08-01 06:20:00') & (data['datetime'] <= '2024-08-04 09:20:00')]
    data['panel_anomaly'] = None
    detected_anomaly_data = detect_anomaly_value_in_row(data)
    check_anomalies_in_windows(detected_anomaly_data)