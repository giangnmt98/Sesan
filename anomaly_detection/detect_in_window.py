from algorithm_base_on_distribution import *
from utils import read_excel_data, create_window

if __name__ == "__main__":
    data = read_excel_data(data_path='data/DataMakeOurlies.xlsx')
    data = data[(data['datetime'] >= '2024-08-01 06:20:00') & (data['datetime'] <= '2024-08-02 09:20:00')]
    data_windows = create_window(data, window_size = 6, step_size=1)
    for window in data_windows:
        print()
        # Select all panel columns except 'datetime', 'Day', and 'Time'
        panel_columns = [column for column in window.columns if column not in ['datetime', 'Day', 'Time']]
        # Capture the 'datetime' column
        datetime_column = window['datetime'].to_numpy()
        # Convert each panel column to a numpy array and store it in a list
        data = [window[panel].to_numpy() for panel in panel_columns]
        # result = t_test_anomaly(data, threshold=0.005)
        result = perform_hierarchical_clustering(data)
        if len(result) > 0:
            print(result)
        #     list_panel = [item['index'] for item in result]
        #     list_anomaly_col = [panel_columns[i] for i in list_panel]
        #     list_normal_col = [item for item in panel_columns if item not in list_anomaly_col]
        #
        #     list_anomaly_col.insert(0, 'datetime')
        #     list_normal_col.insert(0, 'datetime')
        #     print(window[list_anomaly_col])
        #     print(window[list_normal_col])
            # print(window)
            # print(result)
