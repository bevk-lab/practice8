import streamlit as st
import gzip
import json
import queue
import threading
import time
import uuid
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from confluent_kafka import Consumer, Producer
from transformers import pipeline

# Cấu hình giao diện trang
st.set_page_config(page_title="Amazon Fashion Sentiment Streaming", layout="wide")

# ==========================================
# YÊU CẦU CỦA THẦY: CHÈN LINK GOOGLE COLAB
# ==========================================
# ==========================================
# YÊU CẦU CỦA THẦY: CHÈN LINK GOOGLE COLAB
# ==========================================
st.markdown("🔗 **Link Google Colab:** [Xem chi tiết mã nguồn trên Google Colab tại đây](https://colab.research.google.com/drive/1Oebj9Ukgu26NcK0uT1v8956Ns_8BkDMB?usp=sharing)")

st.title("Ứng dụng Big Data Streaming Phân Tích Cảm Xúc Khách Hàng")


# 1. KHỞI TẠO MÔ HÌNH AI (Dùng cache để không load lại nhiều lần)
@st.cache_resource(show_spinner="Đang tải mô hình AI (RoBERTa)...")
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


sentiment_analyzer = load_model()

# 2. CẤU HÌNH HỆ THỐNG
DATA_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/AMAZON_FASHION.json.gz'
BOOTSTRAP_SERVERS = 'cell-1.streaming.sa-saopaulo-1.oci.oraclecloud.com:9092'
TOPIC = 'DemoStreamingFashion'
SASL_USERNAME = 'dant49/dant@uel.edu.vn/ocid1.streampool.oc1.sa-saopaulo-1.amaaaaaai6jti5aa4m3x5a53b3n4uk6smo2cs6wi7vnues4kmrsoy5pr6mcq'
OCI_AUTH_TOKEN = 'AIg(4_0#Xm_sR_u3y251'

RUN_ID = uuid.uuid4().hex[:8]
COMMON_KAFKA_CONF = {
    'bootstrap.servers': BOOTSTRAP_SERVERS,
    'security.protocol': 'SASL_SSL',
    'sasl.mechanism': 'PLAIN',
    'sasl.username': SASL_USERNAME,
    'sasl.password': OCI_AUTH_TOKEN,
}
PRODUCER_CONF = {**COMMON_KAFKA_CONF, 'client.id': f'prod_{RUN_ID}', 'linger.ms': 10, 'acks': '1'}
CONSUMER_CONF = {**COMMON_KAFKA_CONF, 'client.id': f'cons_{RUN_ID}', 'group.id': f'fashion_stream_{RUN_ID}',
                 'auto.offset.reset': 'latest'}

# Biến toàn cục cho Threading
producer_lock = threading.Lock()
producer_stats = {'generated': 0, 'delivered': 0, 'failed': 0, 'error': 'Đang thiết lập kết nối OCI...'}
local_queue = queue.Queue()
use_fallback = False


def delivery_report(err, msg):
    with producer_lock:
        if err:
            producer_stats['failed'] += 1
            producer_stats['error'] = str(err)
        else:
            producer_stats['delivered'] += 1


def file_streaming_producer_worker(producer, stop_event):
    global use_fallback
    try:
        response = requests.get(DATA_URL, stream=True, timeout=30)
        with gzip.GzipFile(fileobj=response.raw) as gz:
            for line in gz:
                if stop_event.is_set(): break
                if not line.strip(): continue

                try:
                    raw_record = json.loads(line.decode('utf-8'))
                except:
                    continue

                amazon_rating = float(raw_record.get('overall', 0.0))
                # YÊU CẦU CỦA THẦY: ĐỔI 'summary' THÀNH 'reviewText'
                review_text = str(raw_record.get('reviewText', 'Không có nội dung')).strip()
                if not review_text: review_text = "Không có nội dung"

                ai_result = sentiment_analyzer(review_text[:500])[0]
                ai_label = ai_result['label'].lower()

                if amazon_rating <= 2.0 and 'pos' in ai_label:
                    effective_sentiment = 'negative'
                elif amazon_rating >= 4.0 and 'neg' in ai_label:
                    effective_sentiment = 'negative'
                else:
                    if 'pos' in ai_label:
                        effective_sentiment = 'positive'
                    elif 'neg' in ai_label:
                        effective_sentiment = 'negative'
                    else:
                        effective_sentiment = 'neutral'

                if effective_sentiment == 'positive':
                    emotion_text, text_color, bg_color = (
                    'Rất tích cực' if amazon_rating >= 4.5 else 'Tích cực', '#052c11', '#a3cfbb')
                elif effective_sentiment == 'negative':
                    emotion_text, text_color, bg_color = (
                    'Rất tiêu cực' if amazon_rating <= 1.5 else 'Tiêu cực', '#58151c', '#f1aeb5')
                else:
                    emotion_text, text_color, bg_color = ('Trung lập', '#41464b', '#e2e3e5')

                event = {
                    'run_id': RUN_ID,
                    'amazon_rating': amazon_rating,
                    'reviewText': review_text,
                    'emotion': emotion_text,
                    't_color': text_color,
                    'b_color': bg_color
                }
                payload = json.dumps(event).encode('utf-8')
                local_queue.put(payload)

                if not use_fallback:
                    producer.produce(TOPIC, value=payload, on_delivery=delivery_report)
                    producer.poll(0)

                with producer_lock:
                    producer_stats['generated'] += 1

    except Exception as exc:
        with producer_lock:
            producer_stats['error'] = f"Lỗi hệ thống tập tin: {exc}"
    if not use_fallback: producer.flush(5)


# Hàm vẽ biểu đồ trả về figure cho Streamlit
def create_3d_chart(rows):
    categories = ['Rất tích cực', 'Tích cực', 'Trung lập', 'Tiêu cực', 'Rất tiêu cực']
    counts = {cat: 0 for cat in categories}
    for row in rows:
        emo = row['emotion']
        if emo in counts: counts[emo] += 1

    y_vals = [counts[cat] for cat in categories]
    fig = plt.figure(figsize=(6, 4), facecolor='white')
    ax = fig.add_subplot(projection='3d')
    x = np.arange(len(categories))
    y = np.zeros(len(categories))
    z = np.zeros(len(categories))
    dx = np.ones(len(categories)) * 0.4
    dy = np.ones(len(categories)) * 0.4
    bar_colors = ['#2b8a3e', '#51cf66', '#ced4da', '#ff6b6b', '#c92a2a']

    ax.bar3d(x - 0.2, y, z, dx, dy, y_vals, color=bar_colors, shade=True, alpha=0.9)
    ax.view_init(elev=28, azim=-55)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, rotation=30)
    ax.set_zlabel('Số lượng')
    return fig


# 3. GIAO DIỆN STREAMING
if st.button("Khởi động Streaming", type="primary"):
    stop_event = threading.Event()
    producer = Producer(PRODUCER_CONF)
    consumer = Consumer(CONSUMER_CONF)
    try:
        consumer.subscribe([TOPIC])
    except:
        pass

    # Chạy worker trong thread
    threading.Thread(target=file_streaming_producer_worker, args=(producer, stop_event), daemon=True).start()

    # Chuẩn bị khung hiển thị (Placeholders)
    status_placeholder = st.empty()
    col1, col2, col3, col4 = st.columns(4)
    metric_gen = col1.empty()
    metric_del = col2.empty()
    metric_recv = col3.empty()
    metric_time = col4.empty()

    col_table, col_chart = st.columns([1.5, 1])
    table_placeholder = col_table.empty()
    chart_placeholder = col_chart.empty()

    rows = []
    consumed_count = 0
    started_at = time.monotonic()

    try:
        while time.monotonic() - started_at < 60:  # Chạy demo 60 giây (bạn có thể tăng lên)
            if not use_fallback and time.monotonic() - started_at > 12:
                with producer_lock:
                    if producer_stats['delivered'] == 0: use_fallback = True


            class MockMessage:
                def __init__(self, val): self._val = val

                def value(self): return self._val

                def error(self): return None


            if not use_fallback:
                message = consumer.poll(0.1)
            else:
                try:
                    message = MockMessage(local_queue.get(timeout=0.05))
                except queue.Empty:
                    message = None

            if message is not None and not message.error():
                try:
                    event = json.loads(message.value().decode('utf-8'))
                    if event.get('run_id') == RUN_ID:
                        consumed_count += 1
                        rows.append(event)
                        if len(rows) > 500: rows.pop(0)
                except:
                    pass

            # Cập nhật giao diện
            with producer_lock:
                pstats = dict(producer_stats)
            elapsed = time.monotonic() - started_at

            status_placeholder.success("TRẠNG THÁI: ĐANG HOẠT ĐỘNG")
            metric_gen.metric("Đã xử lý", f"{pstats['generated']:,}")
            metric_del.metric("Đã chuyển OCI", f"{pstats['delivered']:,}")
            metric_recv.metric("Đã nhận", f"{consumed_count:,}")
            metric_time.metric("Thời gian", f"{elapsed:.1f}s")

            if len(rows) > 0:
                recent_rows = rows[-7:]
                # Tạo HTML Table
                table_html = "<table style='width:100%; text-align:left; border-collapse:collapse;'>"
                table_html += "<tr style='border-bottom:2px solid #ccc;'><th>Rating</th><th>Nội dung (reviewText)</th><th>Cảm xúc</th></tr>"
                for r in reversed(recent_rows):
                    stars = "⭐" * int(r['amazon_rating'])
                    table_html += f"<tr style='border-bottom:1px solid #eee;'>"
                    table_html += f"<td style='padding:8px;'><b>{r['amazon_rating']}</b><br><span style='font-size:10px;'>{stars}</span></td>"
                    table_html += f"<td>{r['reviewText'][:150]}...</td>"
                    table_html += f"<td><span style='background-color:{r['b_color']}; color:{r['t_color']}; padding:4px 8px; border-radius:4px;'>{r['emotion']}</span></td>"
                    table_html += "</tr>"
                table_html += "</table>"
                table_placeholder.markdown(table_html, unsafe_allow_html=True)

                # Cập nhật biểu đồ
                chart_placeholder.pyplot(create_3d_chart(rows))

            time.sleep(0.5)  # Chờ 0.5s để UI không bị giật lag

    except Exception as e:
        st.error(f"Đã dừng: {e}")
    finally:
        stop_event.set()
        status_placeholder.warning("ĐÃ DỪNG STREAMING")
