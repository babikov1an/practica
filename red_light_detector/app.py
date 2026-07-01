import os
import json
import sqlite3
import cv2
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from detector import RedLightDetector
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openpyxl import Workbook

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['DATABASE'] = 'history.db'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

detector = RedLightDetector()

def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            result_filename TEXT,
            stats TEXT,
            processing_time REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(filename, result_filename, stats, processing_time):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO requests (timestamp, filename, result_filename, stats, processing_time)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, result_filename, json.dumps(stats), processing_time))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "Файл не загружен"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Файл не выбран"}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    start_time = datetime.now()
    output_img, stats = detector.detect(filepath)
    processing_time = (datetime.now() - start_time).total_seconds()

    if output_img is None:
        return jsonify(stats), 400

    result_filename = f"result_{unique_filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, output_img)

    stats['processing_time'] = round(processing_time, 2)
    stats['original_image'] = f"static/uploads/{unique_filename}"
    stats['result_image'] = f"static/results/{result_filename}"

    save_to_db(unique_filename, result_filename, stats, processing_time)

    return jsonify(stats)

@app.route('/history')
def get_history():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM requests ORDER BY id DESC LIMIT 50')
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "filename": row[2],
            "result_filename": row[3],
            "stats": json.loads(row[4]),
            "processing_time": row[5]
        })

    return jsonify(history)

@app.route('/export/pdf')
def export_pdf():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM requests ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()

    pdf_path = os.path.join(app.config['RESULT_FOLDER'], 'report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Отчет по детекции нарушений")
    c.drawString(50, height - 70, "Проезд на красный свет")

    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(50, y, f"Дата отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30
    c.drawString(50, y, f"Всего обработано изображений: {len(rows)}")
    y -= 30

    total_violations = sum(json.loads(row[4]).get('violations', 0) for row in rows)
    c.drawString(50, y, f"Всего нарушений обнаружено: {total_violations}")

    y -= 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Детализация по изображениям:")
    y -= 25
    c.setFont("Helvetica", 10)

    for row in rows[:20]:
        stats = json.loads(row[4])
        text = f"{row[1]} | Авто: {stats.get('total_cars', 0)} | Нарушения: {stats.get('violations', 0)}"
        c.drawString(50, y, text)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    return send_from_directory(app.config['RESULT_FOLDER'], 'report.pdf', as_attachment=True)

@app.route('/export/excel')
def export_excel():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM requests ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "Отчет"

    headers = ["ID", "Дата", "Исходное фото", "Авто", "Светофоры", "Нарушения", "Время (с)"]
    ws.append(headers)

    for row in rows:
        stats = json.loads(row[4])
        ws.append([
            row[0],
            row[1],
            row[2],
            stats.get('total_cars', 0),
            stats.get('total_traffic_lights', 0),
            stats.get('violations', 0),
            row[5]
        ])

    excel_path = os.path.join(app.config['RESULT_FOLDER'], 'report.xlsx')
    wb.save(excel_path)
    return send_from_directory(app.config['RESULT_FOLDER'], 'report.xlsx', as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5001, debug=True)
