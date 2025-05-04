from flask import Flask, render_template, request, Response, jsonify, send_file
import cv2
import numpy as np
import os
import json
from datetime import datetime
import pandas as pd
from io import BytesIO
import base64
import threading
import time
from fpdf import FPDF
from ultralytics import YOLO
import torch

app = Flask(__name__)

# Создание директорий для хранения данных
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('static/results'):
    os.makedirs('static/results')

# Загрузка модели YOLO
import pickle  # Add this import to fix the _pickle error

# Monkey patch the torch.load function to always use weights_only=False
original_torch_load = torch.load

def custom_load(*args, **kwargs):
    # Force weights_only to False for all torch.load calls
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Replace the original function with our custom one
torch.load = custom_load

# Now load the model normally
try:
    model = YOLO('best.pt', task='detect')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
# model = YOLO('best.pt', task='detect')

# Глобальная переменная для хранения истории запросов
history = []
if os.path.exists('history.json'):
    with open('history.json', 'r') as f:
        try:
            history = json.load(f)
        except:
            history = []

# Глобальные переменные для управления стримом с камеры
camera = None
camera_stream_active = False
camera_thread = None

def save_history():
    """Сохранение истории запросов в JSON файл"""
    with open('history.json', 'w') as f:
        json.dump(history, f, indent=4, default=str)

def detect_elephants(image):
    """Обнаружение слонов на изображении с помощью YOLOv8n"""
    results = model(image)
    result = results[0]
    
    # Фильтрация результатов - оставляем только класс "elephant" (id=22 в COCO dataset) но у нас id=0
    elephants = []
    for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
        cls_id = int(cls.item())
        if cls_id == 0:
            elephants.append({
                'id': i,
                'box': box.tolist(),
                'confidence': conf.item()
            })
    
    # Отрисовка результатов на изображении
    annotated_img = results[0].plot()
    
    return elephants, annotated_img

def camera_stream():
    """Функция для стриминга с камеры"""
    global camera, camera_stream_active
    
    while camera_stream_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Здесь можно обрабатывать кадр (например, обнаруживать слонов)
        # Но это будет происходить отдельно от основного потока для производительности
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)  # Небольшая задержка для снижения нагрузки

@app.route('/')
def index():
    """Главная страница приложения"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загруженных файлов"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Сохранение загруженного файла
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Определение типа файла (изображение или видео)
    file_ext = file.filename.split('.')[-1].lower()
    
    detection_result = {
        'id': len(history) + 1,
        'timestamp': datetime.now(),
        'file_name': file.filename,
        'type': 'image' if file_ext in ['jpg', 'jpeg', 'png'] else 'video',
        'elephants_detected': 0,
        'processed_file': ''
    }
    
    if file_ext in ['jpg', 'jpeg', 'png']:
        # Обработка изображения
        image = cv2.imread(file_path)
        elephants, annotated_img = detect_elephants(image)
        
        # Сохранение обработанного изображения
        result_path = os.path.join('static/results', f"processed_{file.filename}")
        cv2.imwrite(result_path, annotated_img)
        
        detection_result['elephants_detected'] = len(elephants)
        detection_result['processed_file'] = f"results/processed_{file.filename}"
        detection_result['elephants'] = elephants
        
    elif file_ext in ['mp4', 'avi', 'mov']:
        # Обработка видео
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        result_path = os.path.join('results', f"processed_{file.filename}")
        out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        total_elephants = 0
        frame_count = 0
        elephants_per_frame = []
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Обрабатываем каждый 5-й кадр для ускорения
            if frame_count % 5 == 0:
                elephants, annotated_frame = detect_elephants(frame)
                total_elephants = max(total_elephants, len(elephants))
                elephants_per_frame.append(len(elephants))
                out.write(annotated_frame)
            else:
                out.write(frame)
            
            frame_count += 1
        
        video.release()
        out.release()
        
        # Создаем превью первого кадра для отображения в веб-интерфейсе
        video = cv2.VideoCapture(result_path)
        ret, first_frame = video.read()
        if ret:
            preview_path = os.path.join('static/results', f"preview_{file.filename.split('.')[0]}.jpg")
            cv2.imwrite(preview_path, first_frame)
            detection_result['preview'] = f"results/preview_{file.filename.split('.')[0]}.jpg"
        video.release()
        
        detection_result['elephants_detected'] = total_elephants
        detection_result['processed_file'] = f"../results/processed_{file.filename}"
        detection_result['frames_processed'] = frame_count
        detection_result['elephants_per_frame'] = elephants_per_frame
    
    # Добавление результата в историю и сохранение
    history.append(detection_result)
    save_history()
    
    return jsonify({
        'success': True,
        'result': detection_result
    })

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Запуск потока с камеры"""
    global camera, camera_stream_active, camera_thread
    
    if camera_stream_active:
        return jsonify({'error': 'Camera stream already active'}), 400
    
    try:
        camera_id = int(request.form.get('camera_id', 0))
        camera = cv2.VideoCapture(camera_id)
        
        if not camera.isOpened():
            return jsonify({'error': f'Could not open camera with ID {camera_id}'}), 400
        
        camera_stream_active = True
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Остановка потока с камеры"""
    global camera, camera_stream_active
    
    if camera:
        camera_stream_active = False
        camera.release()
        camera = None
        
    return jsonify({'success': True})

@app.route('/video_feed')
def video_feed():
    """Маршрут для потоковой передачи видео"""
    if not camera_stream_active:
        # Если камера не активна, возвращаем пустое изображение
        blank_image = np.zeros((480, 640, 3), np.uint8)
        ret, buffer = cv2.imencode('.jpg', blank_image)
        frame_bytes = buffer.tobytes()
        return Response(
            (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    return Response(
        camera_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/capture_and_process', methods=['POST'])
def capture_and_process():
    """Захват кадра с камеры и его обработка"""
    global camera
    
    if not camera or not camera_stream_active:
        return jsonify({'error': 'Camera is not active'}), 400
    
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture image from camera'}), 500
    
    # Сохранение и обработка кадра
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"camera_capture_{timestamp}.jpg"
    file_path = os.path.join('uploads', file_name)
    cv2.imwrite(file_path, frame)
    
    # Обнаружение слонов
    elephants, annotated_img = detect_elephants(frame)
    
    # Сохранение обработанного изображения
    result_path = os.path.join('static/results', f"processed_{file_name}")
    cv2.imwrite(result_path, annotated_img)
    
    detection_result = {
        'id': len(history) + 1,
        'timestamp': datetime.now(),
        'file_name': file_name,
        'type': 'camera_capture',
        'elephants_detected': len(elephants),
        'processed_file': f"results/processed_{file_name}",
        'elephants': elephants
    }
    
    # Добавление результата в историю и сохранение
    history.append(detection_result)
    save_history()
    
    return jsonify({
        'success': True,
        'result': detection_result
    })

@app.route('/history')
def get_history():
    """Получение истории запросов"""
    global history
    return jsonify(history)

@app.route('/stats')
def get_stats():
    """Получение статистики обнаружений"""
    global history
    
    if not history:
        return jsonify({
            'total_detections': 0,
            'total_elephants': 0,
            'avg_elephants_per_detection': 0,
            'detections_by_type': {}
        })
    
    total_elephants = sum(item.get('elephants_detected', 0) for item in history)
    detections_by_type = {}
    
    for item in history:
        file_type = item.get('type', 'unknown')
        if file_type not in detections_by_type:
            detections_by_type[file_type] = {
                'count': 0,
                'elephants': 0
            }
        
        detections_by_type[file_type]['count'] += 1
        detections_by_type[file_type]['elephants'] += item.get('elephants_detected', 0)
    
    return jsonify({
        'total_detections': len(history),
        'total_elephants': total_elephants,
        'avg_elephants_per_detection': total_elephants / len(history) if history else 0,
        'detections_by_type': detections_by_type,
        'detection_history': [
            {
                'timestamp': str(item.get('timestamp')),
                'elephants': item.get('elephants_detected', 0)
            }
            for item in history
        ]
    })

@app.route('/generate_excel_report')
def generate_excel_report():
    """Генерация отчета в формате Excel"""
    if not history:
        return jsonify({'error': 'No data available to generate report'}), 400
    
    # Создаем DataFrame из истории
    df_data = []
    for item in history:
        df_data.append({
            'ID': item.get('id'),
            'Timestamp': item.get('timestamp'),
            'File Name': item.get('file_name'),
            'Type': item.get('type'),
            'Elephants Detected': item.get('elephants_detected')
        })
    
    df = pd.DataFrame(df_data)
    
    # Создаем отчет Excel в памяти
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='ElephantDetections', index=False)
        
        # Создаем лист со статистикой
        stats_data = {
            'Metric': ['Total Detections', 'Total Elephants', 'Average Elephants per Detection'],
            'Value': [
                len(history),
                sum(item.get('elephants_detected', 0) for item in history),
                sum(item.get('elephants_detected', 0) for item in history) / len(history) if history else 0
            ]
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)
        
        # Улучшаем форматирование
        workbook = writer.book
        worksheet = writer.sheets['ElephantDetections']
        
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 15)
    
    output.seek(0)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'elephant_detection_report_{timestamp}.xlsx'
    )

@app.route('/generate_pdf_report')
def generate_pdf_report():
    """Генерация отчета в формате PDF"""
    if not history:
        return jsonify({'error': 'No data available to generate report'}), 400
    
    # Создаем PDF-отчет
    pdf = FPDF()
    pdf.add_page()
    
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    # pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)

    # Заголовок
    pdf.set_font('DejaVu', '', 16)
    pdf.cell(0, 10, 'Отчет по обнаружению слонов', 0, 1, 'C')
    pdf.ln(10)
    
    # Основная статистика
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, 'Сводная статистика:', 0, 1)
    
    pdf.set_font('DejaVu', '', 12)
    total_detections = len(history)
    total_elephants = sum(item.get('elephants_detected', 0) for item in history)
    avg_elephants = total_elephants / total_detections if total_detections else 0
    
    pdf.cell(0, 8, f'Всего обработано материалов: {total_detections}', 0, 1)
    pdf.cell(0, 8, f'Всего обнаружено слонов: {total_elephants}', 0, 1)
    pdf.cell(0, 8, f'Среднее количество слонов на материал: {avg_elephants:.2f}', 0, 1)
    pdf.ln(5)
    
    # Таблица обнаружений
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, 'История обнаружений:', 0, 1)
    
    # Заголовки таблицы
    column_widths = [15, 45, 50, 30, 50]
    pdf.set_font('DejaVu', '', 10)
    pdf.cell(column_widths[0], 10, 'ID', 1, 0, 'C')
    pdf.cell(column_widths[1], 10, 'Дата и время', 1, 0, 'C')
    pdf.cell(column_widths[2], 10, 'Имя файла', 1, 0, 'C')
    pdf.cell(column_widths[3], 10, 'Тип', 1, 0, 'C')
    pdf.cell(column_widths[4], 10, 'Обнаружено слонов', 1, 1, 'C')
    
    # Данные таблицы
    pdf.set_font('DejaVu', '', 10)
    for item in history:
        pdf.cell(column_widths[0], 10, str(item.get('id')), 1, 0, 'C')
        timestamp = item.get('timestamp')
        if isinstance(timestamp, str):
            time_str = timestamp
        else:
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        pdf.cell(column_widths[1], 10, time_str, 1, 0, 'C')
        
        # Обрезаем имя файла если оно слишком длинное
        file_name = item.get('file_name', '')
        if len(file_name) > 20:
            file_name = file_name[:17] + '...'
        pdf.cell(column_widths[2], 10, file_name, 1, 0, 'C')
        
        pdf.cell(column_widths[3], 10, item.get('type', ''), 1, 0, 'C')
        pdf.cell(column_widths[4], 10, str(item.get('elephants_detected', 0)), 1, 1, 'C')
    
   # Создаем временный файл для PDF
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_path = temp_file.name
    temp_file.close()
    
    # Сначала сохраняем PDF в файл
    pdf.output(pdf_path)
    
    # Читаем файл в память
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # Удаляем временный файл
    os.unlink(pdf_path)
    
    # Создаем BytesIO объект с данными PDF
    output = BytesIO(pdf_data)
    output.seek(0)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(
        output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'elephant_detection_report_{timestamp}.pdf'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)