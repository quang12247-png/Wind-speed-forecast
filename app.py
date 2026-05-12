import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

# CẤU HÌNH TRANG
st.set_page_config(
    page_title="BT1 WIND FARM - CÔNG CỤ LẤY DỮ LIỆU THỜI TIẾT PHỤC VỤ SẢN XUẤT",
    page_icon="🌤️",
    layout="wide"
)

# URL của Open-Meteo API
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# HÀM LẤY DỮ LIỆU TỪ OPEN-METEO API
# ============================================================
def get_wind_data_openmeteo(lat, lon, days=5, wind_height=10):
    """
    Lấy dữ liệu từ Open-Meteo API
    - Nếu days <= 7: sử dụng minutely_15 (dữ liệu 15 phút gốc)
    - Nếu days > 7: sử dụng hourly (dữ liệu 1 giờ)
    - wind_gusts CHỈ có ở độ cao 10m
    - wind_speed có ở các độ cao: 10m, 80m, 120m, 180m
    """
    
    # Xác định tham số độ cao cho tốc độ gió thường
    if wind_height == 10:
        wind_param = "wind_speed_10m"
        height_label = "10m"
    else:
        wind_param = f"wind_speed_{wind_height}m"
        height_label = f"{wind_height}m"
    
    # Gió giật CHỈ lấy ở 10m (API không hỗ trợ ở độ cao khác)
    gust_param = "wind_gusts_10m"
    
    # QUYẾT ĐỊNH ENDPOINT DỰA TRÊN SỐ NGÀY
    if days <= 7:
        # Dùng minutely_15 cho chi tiết cao nhất
        time_resolution = "minutely_15"
        freq_text = "15 phút (gốc)"
        params = {
            "latitude": lat,
            "longitude": lon,
            "minutely_15": [
                "temperature_2m",
                wind_param,
                gust_param
            ],
            "forecast_days": days,
            "timezone": "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms"
        }
    else:
        # Dùng hourly cho dữ liệu dài hơn (tối đa 16 ngày)
        time_resolution = "hourly"
        freq_text = "1 giờ (gốc)"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                wind_param,
                gust_param
            ],
            "forecast_days": days,
            "timezone": "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms"
        }
    
    try:
        with st.spinner(f'🌐 Đang kết nối Open-Meteo API (độ cao: {height_label}, độ phân giải: {freq_text})...'):
            response = requests.get(OPENMETEO_URL, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"Lỗi API: {response.status_code}")
            st.error(response.text)
            return None, None, None, None, None, None, None, None
        
        data = response.json()
        time_data = data.get(time_resolution, {})
        
        # Lấy các mảng dữ liệu
        timestamps_str = time_data.get("time", [])
        temperatures = time_data.get("temperature_2m", [])
        wind_speeds = time_data.get(wind_param, [])
        wind_gusts = time_data.get(gust_param, [])
        
        if not timestamps_str:
            st.error("Không có dữ liệu chuỗi thời gian")
            return None, None, None, None, None, None, None, None
        
        # Chuyển đổi chuỗi thời gian ISO 8601 sang timestamp
        timestamps_ms = []
        datetime_objects = []
        for ts_str in timestamps_str:
            dt = datetime.fromisoformat(ts_str)
            timestamps_ms.append(int(dt.timestamp() * 1000))
            datetime_objects.append(dt)
        
        # Xác định độ phân giải gốc
        original_freq = 15 if days <= 7 else 60
        
        st.success(f"✅ Đã lấy thành công {len(timestamps_str)} mốc dữ liệu (gốc: {freq_text})")
        st.info(f"📌 Tốc độ gió thường: độ cao {height_label} | Gió giật: độ cao 10m")
        
        return timestamps_ms, wind_speeds, wind_gusts, temperatures, datetime_objects, height_label, original_freq, freq_text
        
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")
        return None, None, None, None, None, None, None, None


# ============================================================
# HÀM NỘI SUY DỮ LIỆU (CHO TRƯỜNG HỢP >7 NGÀY)
# ============================================================
def interpolate_data(timestamps, wind_speed, wind_gust, temperature, datetime_objects, original_freq, target_freq_minutes):
    """
    Nội suy dữ liệu từ tần suất thô (1 giờ) xuống tần suất mịn hơn (30 phút hoặc 15 phút)
    Sử dụng phương pháp nội suy bậc 3 (cubic) cho dữ liệu liên tục
    """
    from scipy.interpolate import interp1d
    
    # Chuyển đổi dữ liệu thành mảng numpy
    timestamps_seconds = np.array([ts / 1000 for ts in timestamps])  # Chuyển về giây
    wind_speed_array = np.array(wind_speed)
    wind_gust_array = np.array(wind_gust)
    temperature_array = np.array(temperature)
    
    # Tạo timestamp mới với tần suất mong muốn
    start_time = timestamps_seconds[0]
    end_time = timestamps_seconds[-1]
    
    # Tính khoảng thời gian giữa các điểm nội suy (phút -> giây)
    interval_seconds = target_freq_minutes * 60
    
    # Tạo mảng thời gian mới
    new_timestamps_seconds = np.arange(start_time, end_time + interval_seconds, interval_seconds)
    new_datetime_objects = [datetime.fromtimestamp(ts) for ts in new_timestamps_seconds]
    new_timestamps_ms = [int(ts * 1000) for ts in new_timestamps_seconds]
    
    # Loại bỏ các giá trị NaN
    valid_idx = ~(np.isnan(wind_speed_array) | np.isnan(wind_gust_array) | np.isnan(temperature_array))
    
    if np.sum(valid_idx) < 2:
        st.error("Không đủ dữ liệu hợp lệ để nội suy")
        return None, None, None, None, None, None
    
    timestamps_valid = timestamps_seconds[valid_idx]
    wind_speed_valid = wind_speed_array[valid_idx]
    wind_gust_valid = wind_gust_array[valid_idx]
    temperature_valid = temperature_array[valid_idx]
    
    try:
        # Nội suy tốc độ gió thường (cubic spline)
        f_speed = interp1d(timestamps_valid, wind_speed_valid, kind='cubic', fill_value='extrapolate')
        new_wind_speed = f_speed(new_timestamps_seconds)
        
        # Nội suy gió giật (cubic spline)
        f_gust = interp1d(timestamps_valid, wind_gust_valid, kind='cubic', fill_value='extrapolate')
        new_wind_gust = f_gust(new_timestamps_seconds)
        
        # Nội suy nhiệt độ (linear cho ổn định hơn)
        f_temp = interp1d(timestamps_valid, temperature_valid, kind='linear', fill_value='extrapolate')
        new_temperature = f_temp(new_timestamps_seconds)
        
        # Đảm bảo không có giá trị âm cho tốc độ gió
        new_wind_speed = np.maximum(new_wind_speed, 0)
        new_wind_gust = np.maximum(new_wind_gust, 0)
        
        st.info(f"📊 Đã nội suy: {len(new_timestamps_ms)} mốc ({target_freq_minutes} phút/lần) từ {len(timestamps)} mốc gốc (1 giờ)")
        st.success(f"✨ Phương pháp nội suy: Bậc 3 (cubic) cho gió, Tuyến tính (linear) cho nhiệt độ")
        
        return new_timestamps_ms, new_wind_speed.tolist(), new_wind_gust.tolist(), new_temperature.tolist(), new_datetime_objects
        
    except Exception as e:
        st.error(f"Lỗi khi nội suy dữ liệu: {str(e)}")
        return None, None, None, None, None, None


# ============================================================
# HÀM TỔNG HỢP DỮ LIỆU (RESAMPLE VÀ NỘI SUY)
# ============================================================
def resample_data(timestamps, wind_speed, wind_gust, temperature, datetime_objects, original_freq, target_freq_minutes):
    """
    Tổng hợp hoặc nội suy dữ liệu từ tần suất gốc lên/xuống tần suất mong muốn
    original_freq: 15 hoặc 60 (phút)
    target_freq_minutes: 15, 30, 60
    """
    
    # Kiểm tra dữ liệu đầu vào
    if not datetime_objects or len(datetime_objects) == 0:
        st.error("Không có dữ liệu datetime để xử lý")
        return None, None, None, None, None, None
    
    # Trường hợp 1: Giữ nguyên tần suất gốc
    if original_freq == target_freq_minutes:
        st.info(f"📊 Giữ nguyên dữ liệu gốc: {len(timestamps)} mốc ({original_freq} phút/lần)")
        
        # Tính gió trung bình
        wind_avg = []
        for speed, gust in zip(wind_speed, wind_gust):
            if speed is not None and gust is not None and not np.isnan(speed) and not np.isnan(gust):
                wind_avg.append((speed + gust) / 2)
            else:
                wind_avg.append(None)
        
        return timestamps, wind_speed, wind_gust, wind_avg, temperature, datetime_objects
    
    # Trường hợp 2: Dữ liệu gốc tần suất cao hơn target (resample xuống)
    elif original_freq < target_freq_minutes:
        # Chuyển dữ liệu thành DataFrame
        df_orig = pd.DataFrame({
            'datetime': datetime_objects,
            'wind_speed': wind_speed,
            'wind_gust': wind_gust,
            'temperature': temperature
        })
        
        # Đặt cột datetime làm index
        df_orig.set_index('datetime', inplace=True)
        df_orig.index = pd.to_datetime(df_orig.index)
        
        # Xác định frequency string cho resample
        if target_freq_minutes == 30:
            freq_string = '30min'
        elif target_freq_minutes == 60:
            freq_string = '1h'
        else:
            freq_string = f'{target_freq_minutes}min'
        
        # Resample: lấy giá trị trung bình trong mỗi khoảng
        df_resampled = df_orig.resample(freq_string).agg({
            'wind_speed': 'mean',
            'wind_gust': 'mean',
            'temperature': 'mean'
        })
        
        # Làm sạch dữ liệu
        df_resampled = df_resampled.dropna(how='any')
        
        if len(df_resampled) == 0:
            st.warning(f"⚠️ Không có dữ liệu sau khi tổng hợp. Giữ nguyên dữ liệu gốc.")
            wind_avg = []
            for speed, gust in zip(wind_speed, wind_gust):
                if speed is not None and gust is not None and not np.isnan(speed) and not np.isnan(gust):
                    wind_avg.append((speed + gust) / 2)
                else:
                    wind_avg.append(None)
            return timestamps, wind_speed, wind_gust, wind_avg, temperature, datetime_objects
        
        # Chuyển đổi về định dạng cho code hiện tại
        timestamps_res = [int(dt.timestamp() * 1000) for dt in df_resampled.index]
        datetime_res = df_resampled.index.to_list()
        wind_speed_res = df_resampled['wind_speed'].tolist()
        wind_gust_res = df_resampled['wind_gust'].tolist()
        temp_res = df_resampled['temperature'].tolist()
        
        # Tính gió trung bình
        wind_avg_res = []
        for speed, gust in zip(wind_speed_res, wind_gust_res):
            if speed is not None and gust is not None and not np.isnan(speed) and not np.isnan(gust):
                wind_avg_res.append((speed + gust) / 2)
            else:
                wind_avg_res.append(None)
        
        st.info(f"📊 Đã tổng hợp: {len(timestamps_res)} mốc ({target_freq_minutes} phút/lần) từ {len(timestamps)} mốc gốc ({original_freq} phút)")
        
        return timestamps_res, wind_speed_res, wind_gust_res, wind_avg_res, temp_res, datetime_res
    
    # Trường hợp 3: Dữ liệu gốc tần suất thấp hơn target (nội suy lên)
    else:  # original_freq > target_freq_minutes
        st.info(f"🔄 Đang nội suy dữ liệu từ {original_freq} phút xuống {target_freq_minutes} phút...")
        
        # Sử dụng hàm nội suy
        result = interpolate_data(timestamps, wind_speed, wind_gust, temperature, 
                                 datetime_objects, original_freq, target_freq_minutes)
        
        if result[0] is None:
            st.warning(f"⚠️ Không thể nội suy dữ liệu. Sử dụng phương pháp thay thế (linear với dữ liệu gốc).")
            # Fallback: sử dụng phương pháp đơn giản hơn
            timestamps_res, wind_speed_res, wind_gust_res, temp_res, datetime_res = result
            if timestamps_res is None:
                # Giữ nguyên dữ liệu gốc nếu không thể nội suy
                wind_avg = []
                for speed, gust in zip(wind_speed, wind_gust):
                    if speed is not None and gust is not None and not np.isnan(speed) and not np.isnan(gust):
                        wind_avg.append((speed + gust) / 2)
                    else:
                        wind_avg.append(None)
                return timestamps, wind_speed, wind_gust, wind_avg, temperature, datetime_objects
        
        # Tính gió trung bình sau nội suy
        wind_avg_res = []
        for speed, gust in zip(result[1], result[2]):
            if speed is not None and gust is not None and not np.isnan(speed) and not np.isnan(gust):
                wind_avg_res.append((speed + gust) / 2)
            else:
                wind_avg_res.append(None)
        
        return result[0], result[1], result[2], wind_avg_res, result[3], result[4]


# ============================================================
# HÀM TẠO FILE EXCEL
# ============================================================
def create_excel_file(timestamps, wind_speed, wind_gust, wind_avg, temperature, lat, lon, freq_minutes, wind_height, data_source_info):
    """Tạo file Excel với 3 sheet và định dạng số 1 chữ số thập phân"""
    if not timestamps or len(timestamps) == 0:
        st.error("Không có dữ liệu để tạo file Excel")
        return None
    
    data_rows = []
    for i, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts / 1000)
        
        # Định dạng số với 1 chữ số thập phân
        gió_giật = ''
        if i < len(wind_gust) and wind_gust[i] is not None and not np.isnan(wind_gust[i]):
            gió_giật = f"{wind_gust[i]:.1f}"
        
        gió_thường = ''
        if i < len(wind_speed) and wind_speed[i] is not None and not np.isnan(wind_speed[i]):
            gió_thường = f"{wind_speed[i]:.1f}"
        
        gió_tb = ''
        if i < len(wind_avg) and wind_avg[i] is not None and not np.isnan(wind_avg[i]):
            gió_tb = f"{wind_avg[i]:.1f}"
        
        nhiệt_độ = ''
        if i < len(temperature) and temperature[i] is not None and not np.isnan(temperature[i]):
            nhiệt_độ = f"{temperature[i]:.1f}"
        
        data_rows.append({
            'Thời gian': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'Gió giật (m/s)': gió_giật,
            'Gió thường (m/s)': gió_thường,
            'Gió trung bình (m/s)': gió_tb,
            'Nhiệt độ (°C)': nhiệt_độ
        })
    
    df_main = pd.DataFrame(data_rows)
    
    # Thống kê
    valid_speed = [s for s in wind_speed if s is not None and not np.isnan(s)]
    valid_gust = [g for g in wind_gust if g is not None and not np.isnan(g)]
    valid_avg = [a for a in wind_avg if a is not None and not np.isnan(a)]
    valid_temp = [t for t in temperature if t is not None and not np.isnan(t)]
    
    stats_data = {
        'Chỉ số': ['Gió thường (m/s)', 'Gió giật (m/s)', 'Gió trung bình (m/s)', 'Nhiệt độ (°C)'],
        'Trung bình': [
            f"{sum(valid_speed)/len(valid_speed):.1f}" if valid_speed else 'N/A',
            f"{sum(valid_gust)/len(valid_gust):.1f}" if valid_gust else 'N/A',
            f"{sum(valid_avg)/len(valid_avg):.1f}" if valid_avg else 'N/A',
            f"{sum(valid_temp)/len(valid_temp):.1f}" if valid_temp else 'N/A'
        ],
        'Tối đa': [
            f"{max(valid_speed):.1f}" if valid_speed else 'N/A',
            f"{max(valid_gust):.1f}" if valid_gust else 'N/A',
            f"{max(valid_avg):.1f}" if valid_avg else 'N/A',
            f"{max(valid_temp):.1f}" if valid_temp else 'N/A'
        ],
        'Tối thiểu': [
            f"{min(valid_speed):.1f}" if valid_speed else 'N/A',
            f"{min(valid_gust):.1f}" if valid_gust else 'N/A',
            f"{min(valid_avg):.1f}" if valid_avg else 'N/A',
            f"{min(valid_temp):.1f}" if valid_temp else 'N/A'
        ]
    }
    
    df_stats = pd.DataFrame(stats_data)
    
    freq_text = {15: "15 phút", 30: "30 phút", 60: "1 giờ"}.get(freq_minutes, f"{freq_minutes} phút")
    
    metadata = [
        {'Thông tin': 'Thời gian xuất', 'Giá trị': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
        {'Thông tin': 'Vĩ độ', 'Giá trị': lat},
        {'Thông tin': 'Kinh độ', 'Giá trị': lon},
        {'Thông tin': 'Nguồn dữ liệu', 'Giá trị': 'Open-Meteo API (miễn phí)'},
        {'Thông tin': 'Độ cao gió thường', 'Giá trị': wind_height},
        {'Thông tin': 'Độ cao gió giật', 'Giá trị': '10m (cố định - API chỉ hỗ trợ 10m)'},
        {'Thông tin': 'Độ cao nhiệt độ', 'Giá trị': '2 mét'},
        {'Thông tin': 'Tần suất dữ liệu', 'Giá trị': freq_text},
        {'Thông tin': 'Số mốc dữ liệu', 'Giá trị': len(timestamps)},
        {'Thông tin': 'Ngày bắt đầu', 'Giá trị': datetime.fromtimestamp(timestamps[0]/1000).strftime('%Y-%m-%d %H:%M')},
        {'Thông tin': 'Ngày kết thúc', 'Giá trị': datetime.fromtimestamp(timestamps[-1]/1000).strftime('%Y-%m-%d %H:%M')},
        {'Thông tin': 'Ghi chú', 'Giá trị': data_source_info},
    ]
    
    df_metadata = pd.DataFrame(metadata)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name=f'Chi tiết ({freq_text})', index=False)
        df_stats.to_excel(writer, sheet_name='Thống kê', index=False)
        df_metadata.to_excel(writer, sheet_name='Thông tin', index=False)
    
    output.seek(0)
    return output


# ============================================================
# HÀM TẠO BIỂU ĐỒ
# ============================================================
def create_chart(df, days, freq_minutes, wind_height):
    """Tạo biểu đồ tương tác với Plotly"""
    if df is None or len(df) == 0:
        st.warning("Không có dữ liệu để vẽ biểu đồ")
        return None
    
    # Giới hạn số điểm hiển thị để tránh quá tải
    max_points = 300
    if len(df) > max_points:
        step = len(df) // max_points
        df_chart = df.iloc[::step]
        st.caption(f"📊 Hiển thị {len(df_chart)}/{len(df)} điểm dữ liệu")
    else:
        df_chart = df
    
    freq_text = {15: "15 phút", 30: "30 phút", 60: "1 giờ"}.get(freq_minutes, f"{freq_minutes} phút")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Tốc độ gió (thường: {wind_height}m, giật: 10m)', 'Nhiệt độ (độ cao 2m)'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Chuyển đổi cột số từ string sang number cho biểu đồ
    df_chart_safe = df_chart.copy()
    for col in ['Gió thường (m/s)', 'Gió giật (m/s)', 'Gió TB (m/s)', 'Nhiệt độ (°C)']:
        df_chart_safe[col] = pd.to_numeric(df_chart_safe[col], errors='coerce')
    
    # Biểu đồ gió
    fig.add_trace(
        go.Scatter(x=df_chart_safe['Thời gian'], y=df_chart_safe['Gió thường (m/s)'],
                  mode='lines', name=f'Gió thường ({wind_height}m)',
                  line=dict(color='#2196F3', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_chart_safe['Thời gian'], y=df_chart_safe['Gió giật (m/s)'],
                  mode='lines', name='Gió giật (10m)',
                  line=dict(color='#F44336', width=2, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_chart_safe['Thời gian'], y=df_chart_safe['Gió TB (m/s)'],
                  mode='lines', name='Gió trung bình',
                  line=dict(color='#4CAF50', width=2)),
        row=1, col=1
    )
    
    # Biểu đồ nhiệt độ
    fig.add_trace(
        go.Scatter(x=df_chart_safe['Thời gian'], y=df_chart_safe['Nhiệt độ (°C)'],
                  mode='lines+markers', name='Nhiệt độ',
                  line=dict(color='#FF9800', width=2),
                  marker=dict(size=4)),
        row=2, col=1
    )
    
    # Cập nhật layout
    fig.update_xaxes(title_text="Thời gian", row=2, col=1)
    fig.update_yaxes(title_text="Tốc độ (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Nhiệt độ (°C)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_text=f"Dự báo thời tiết {days} ngày tới (tần suất: {freq_text})",
        hovermode='x unified'
    )
    
    return fig


# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================
def main():
    # Sidebar
    with st.sidebar:
        st.title("🌤️ BT1 WIND FARM")
        st.markdown("---")
        
        st.subheader("📍 Nhập tọa độ")
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Vĩ độ", value=17.348, format="%.6f")
        with col2:
            lon = st.number_input("Kinh độ", value=106.733, format="%.6f")
        
        st.markdown("---")
        
        st.subheader("⚙️ Tùy chọn")
        days = st.slider("📅 Số ngày dự báo", 1, 15, 5, 
                        help="1-7 ngày: dữ liệu 15 phút gốc | 8-15 ngày: dữ liệu 1 giờ gốc (hỗ trợ nội suy xuống 30 phút hoặc 15 phút)")
        
        # Tùy chọn độ cao gió THƯỜNG
        height_options = {
            "10 mét ": 10,
            "80 mét ": 80,
            "120 mét ": 120,
            "180 mét ": 180
        }
        height_choice = st.selectbox(
            "🌬️ Độ cao lấy tốc độ gió thường",
            options=list(height_options.keys()),
            index=0,
            help="Chọn độ cao lấy tốc độ gió thường. Gió giật chỉ có ở 10m."
        )
        wind_height = height_options[height_choice]
        
        st.caption("⚠️ **Lưu ý:** Gió giật (gust) chỉ được API hỗ trợ ở độ cao 10m")
        
        # Tùy chọn độ phân giải - HIỂN THỊ ĐẦY ĐỦ CHO MỌI TRƯỜNG HỢP
        freq_options = {
            "15 phút": 15,
            "30 phút": 30,
            "1 giờ": 60
        }
        
        if days <= 7:
            freq_choice = st.selectbox(
                "⏱️ Độ phân giải dữ liệu",
                options=list(freq_options.keys()),
                index=0,
                help="Dữ liệu gốc 15 phút, có thể tổng hợp lên hoặc nội suy xuống"
            )
        else:
            freq_choice = st.selectbox(
                "⏱️ Độ phân giải dữ liệu",
                options=list(freq_options.keys()),
                index=1,
                help="Dữ liệu gốc 1 giờ, có thể nội suy xuống 30 phút hoặc 15 phút (dùng Cubic Spline)"
            )
        
        freq_minutes = freq_options[freq_choice]
        
        # Hiển thị phương pháp xử lý
        if days > 7 and freq_minutes < 60:
            st.success("✨ **Sẽ sử dụng nội suy Cubic Spline** để tạo dữ liệu chi tiết từ dữ liệu 1 giờ")
            st.caption("📐 Phương pháp: Nội suy bậc 3 (cubic) cho gió, tuyến tính cho nhiệt độ")
        elif days <= 7 and freq_minutes > 15:
            st.info("📊 Sẽ tổng hợp dữ liệu từ 15 phút lên tần suất cao hơn")
        
        st.markdown("---")
        download_btn = st.button("🚀 TẢI DỮ LIỆU", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.info("💡 **Thông tin chi tiết:**")
        st.caption("✅ **Miễn phí 100%** - Không cần API key")
        st.caption(f"✅ **Tốc độ gió thường:** {wind_height}m")
        st.caption("✅ **Gió giật:** 10m (cố định - giới hạn API)")
        st.caption("✅ **Nhiệt độ:** 2m")
        if days <= 7:
            st.caption("✅ **Dữ liệu gốc:** 15 phút/lần")
            st.caption("✅ **Xử lý:** Tổng hợp hoặc nội suy linh hoạt")
        else:
            st.caption("✅ **Dữ liệu gốc:** 1 giờ/lần")
            st.caption("✅ **Xử lý:** Nội suy Cubic Spline xuống 30/15 phút")
        st.caption("✅ **Định dạng số:** 1 chữ số thập phân")
        
        st.markdown("---")
        st.caption("📊 **Nguồn:** Open-Meteo API")
        st.caption("🔗 Dữ liệu từ NOAA, DWD, Météo-France")
    
    # Main content
    st.title("🌊 CÔNG CỤ LẤY DỮ LIỆU THỜI TIẾT PHỤC VỤ SẢN XUẤT - BT1 Wind Farm")
    st.markdown("**Hoàn toàn miễn phí | Dữ liệu chính xác từ Open-Meteo | Phát triển bởi: O&M TEAM**")
    st.markdown("---")
    
    # Hiển thị thông tin tọa độ
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Vĩ độ", f"{lat:.4f}°")
    with col2:
        st.metric("📍 Kinh độ", f"{lon:.4f}°")
    with col3:
        st.metric("📅 Số ngày", f"{days} ngày")
    with col4:
        st.metric("🌬️ Độ cao gió thường", f"{wind_height}m")
    
    if download_btn:
        try:
            # Bước 1: Lấy dữ liệu từ API
            result = get_wind_data_openmeteo(lat, lon, days, wind_height)
            
            if result[0] is None:
                st.warning("⚠️ Không thể lấy dữ liệu. Vui lòng kiểm tra lại tọa độ và kết nối internet.")
                return
            
            timestamps_orig, wind_speed_orig, wind_gust_orig, temp_orig, datetime_orig, actual_height, original_freq, freq_text = result
            
            # Bước 2: Resample hoặc nội suy dữ liệu theo tần suất đã chọn
            result_resample = resample_data(
                timestamps_orig, wind_speed_orig, wind_gust_orig, temp_orig, 
                datetime_orig, original_freq, freq_minutes
            )
            
            # Kiểm tra kết quả
            if result_resample[0] is None:
                st.error("Không thể xử lý dữ liệu. Vui lòng thử lại với độ phân giải khác.")
                return
            
            timestamps, wind_speed, wind_gust, wind_avg, temperature, datetime_resampled = result_resample
            
            # Kiểm tra dữ liệu sau xử lý
            if not timestamps or len(timestamps) == 0:
                st.warning("Không có dữ liệu sau khi xử lý. Vui lòng thử với độ phân giải khác.")
                return
            
            # Tạo thông tin ghi chú cho file Excel
            if days <= 7:
                if freq_minutes == 15:
                    data_source_info = f"Dữ liệu gốc 15 phút, giữ nguyên tần suất gốc"
                elif freq_minutes > 15:
                    data_source_info = f"Dữ liệu gốc 15 phút, đã tổng hợp lên {freq_minutes} phút (lấy trung bình)"
                else:
                    data_source_info = f"Dữ liệu gốc 15 phút, đã nội suy xuống {freq_minutes} phút"
            else:
                if freq_minutes == 60:
                    data_source_info = f"Dữ liệu gốc 1 giờ (do chọn {days} ngày > 7 ngày), giữ nguyên tần suất"
                else:
                    data_source_info = f"Dữ liệu gốc 1 giờ (do chọn {days} ngày > 7 ngày), đã nội suy xuống {freq_minutes} phút bằng phương pháp Cubic Spline"
            
            # Bước 3: Tạo file Excel
            excel_file = create_excel_file(
                timestamps, wind_speed, wind_gust, wind_avg, temperature, 
                lat, lon, freq_minutes, actual_height, data_source_info
            )
            
            if excel_file is None:
                st.error("Không thể tạo file Excel")
                return
            
            # Hiển thị thông báo thành công
            freq_display = f"{freq_minutes} phút" if freq_minutes < 60 else "1 giờ"
            st.success(f"✅ Thành công! Đã lấy {len(timestamps)} mốc dữ liệu (tần suất: {freq_display})")
            
            # Bước 4: Tạo DataFrame để hiển thị
            df_display = pd.DataFrame({
                'Thời gian': [dt.strftime('%Y-%m-%d %H:%M') for dt in datetime_resampled],
                'Gió giật (m/s)': [f"{g:.1f}" if g is not None and not np.isnan(g) else '' for g in wind_gust],
                'Gió thường (m/s)': [f"{s:.1f}" if s is not None and not np.isnan(s) else '' for s in wind_speed],
                'Gió TB (m/s)': [f"{a:.1f}" if a is not None and not np.isnan(a) else '' for a in wind_avg],
                'Nhiệt độ (°C)': [f"{t:.1f}" if t is not None and not np.isnan(t) else '' for t in temperature]
            })
            
            # Bước 5: Hiển thị thống kê nhanh
            st.subheader("📊 Thống kê nhanh")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            valid_speed = [s for s in wind_speed if s is not None and not np.isnan(s)]
            valid_gust = [g for g in wind_gust if g is not None and not np.isnan(g)]
            valid_avg = [a for a in wind_avg if a is not None and not np.isnan(a)]
            valid_temp = [t for t in temperature if t is not None and not np.isnan(t)]
            
            with stat_col1:
                avg_speed = sum(valid_speed)/len(valid_speed) if valid_speed else 0
                st.metric("🌬️ Gió thường TB", f"{avg_speed:.1f} m/s")
                st.caption(f"Max: {max(valid_speed):.1f} m/s" if valid_speed else "")
            
            with stat_col2:
                avg_gust = sum(valid_gust)/len(valid_gust) if valid_gust else 0
                st.metric("💨 Gió giật TB", f"{avg_gust:.1f} m/s")
                st.caption(f"Max: {max(valid_gust):.1f} m/s" if valid_gust else "")
            
            with stat_col3:
                avg_avg = sum(valid_avg)/len(valid_avg) if valid_avg else 0
                st.metric("📊 Gió TB", f"{avg_avg:.1f} m/s")
            
            with stat_col4:
                avg_temp = sum(valid_temp)/len(valid_temp) if valid_temp else 0
                st.metric("🌡️ Nhiệt độ TB", f"{avg_temp:.1f} °C")
                st.caption(f"Max: {max(valid_temp):.1f}°C / Min: {min(valid_temp):.1f}°C" if valid_temp else "")
            
            # Bước 6: Vẽ biểu đồ
            st.subheader("📈 Biểu đồ dự báo")
            fig = create_chart(df_display, days, freq_minutes, actual_height)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Bước 7: Hiển thị bảng dữ liệu
            with st.expander("📊 Xem dữ liệu chi tiết", expanded=False):
                freq_display_text = f"{freq_minutes} phút" if freq_minutes < 60 else "1 giờ"
                st.info(f"📊 Tổng số dòng: {len(df_display)} dòng dữ liệu (tần suất: {freq_display_text})")
                st.dataframe(df_display, use_container_width=True, height=400)
            
            # Bước 8: Nút tải file
            st.markdown("---")
            filename = f"BT1_wind_data_{lat}_{lon}_{wind_height}m_{freq_minutes}min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Tải file Excel (đầy đủ 3 sheet)",
                    data=excel_file,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Hiển thị thông tin file
            st.caption(f"📁 Tên file: {filename}")
            st.caption(f"📊 Dung lượng: {len(excel_file.getvalue()) / 1024:.1f} KB")
            
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
            st.info("Vui lòng kiểm tra lại tọa độ và kết nối internet.")


# ============================================================
# CHẠY ỨNG DỤNG
# ============================================================
if __name__ == "__main__":
    main()
