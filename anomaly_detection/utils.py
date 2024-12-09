import pandas as pd

def read_excel_data(data_path='data/DataMakeOurlies.xlsx'):
    data = pd.read_excel(data_path)
    data['Day'] = data['Day'].astype(str)
    data['Time'] = data['Time'].astype(str)
    data['datetime'] = pd.to_datetime(data['Day'] + ' ' + data['Time'])
    list_panel_col = [col for col in data.columns if col not in ['datetime','Day', 'Time']]
    data.rename(columns={i: f"panel_{i}" for i in list_panel_col}, inplace=True)
    columns = ['datetime'] + [col for col in data.columns if col != 'datetime']
    data = data[columns]
    data = data.drop(['Day', 'Time'], axis=1)
    return data

def create_window(data, window_size = 6, step_size = 1):
    # Sử dụng hàm để tạo cửa sổ dữ liệu
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data.iloc[start:start + window_size])

    # Biến các cửa sổ thành danh sách DataFrame
    data_windows = [window.reset_index(drop=True) for window in windows]
    return data_windows

def transpose_data(data_path):
    data = pd.read_csv(data_path)

    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame(data)

    # Chuyển đổi dữ liệu từ wide format sang long format
    df_melted = pd.melt(
        df,
        id_vars=['Day', 'Time'],  # Cột giữ nguyên
        var_name='panel_id',  # Tên cột mới chứa tên các cột được unpivot
        value_name='value'  # Tên cột chứa giá trị
    )
    df_melted['created_time'] = df_melted['Day'] + ' ' + df_melted['Time']
    df_melted['created_time'] = pd.to_datetime(df_melted['created_time'])
    df_melted = ['created_time', 'panel_id', 'value']
    return df_melted