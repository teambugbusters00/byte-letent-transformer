#!/usr/bin/env python3
"""
Simple Flask server for serving BLT visualizations.
This provides a web interface to view the generated HTML visualizations and plots.
"""

import os
from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify
import json

app = Flask(__name__)

# Path to demo_output directory
DEMO_OUTPUT_DIR = Path(__file__).parent / "demo_output"
PLOT_DATA_DIR = Path(__file__).parent / "plot_data"

@app.route('/')
def index():
    """Main page showing available visualizations."""
    html_files = []
    image_files = []
    data_files = []

    # Scan demo_output directory
    if DEMO_OUTPUT_DIR.exists():
        for file in DEMO_OUTPUT_DIR.iterdir():
            if file.suffix == '.html':
                html_files.append(file.name)
            elif file.suffix in ['.png', '.pdf', '.jpg', '.jpeg']:
                image_files.append(file.name)
            elif file.suffix in ['.json', '.txt']:
                data_files.append(file.name)

    # Scan plot_data directory
    if PLOT_DATA_DIR.exists():
        for file in PLOT_DATA_DIR.iterdir():
            if file.suffix == '.json':
                data_files.append(f"plot_data/{file.name}")

    return render_template('index.html',
                         html_files=html_files,
                         image_files=image_files,
                         data_files=data_files)

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve HTML visualization files."""
    return send_from_directory(DEMO_OUTPUT_DIR, filename)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve image files."""
    return send_from_directory(DEMO_OUTPUT_DIR, filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve data files as JSON."""
    if filename.startswith('plot_data/'):
        directory = PLOT_DATA_DIR
        filename = filename.replace('plot_data/', '')
    else:
        directory = DEMO_OUTPUT_DIR

    file_path = directory / filename
    if file_path.suffix == '.json':
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except:
            return send_from_directory(directory, filename)
    else:
        return send_from_directory(directory, filename)

@app.route('/api/visualizations')
def list_visualizations():
    """API endpoint to list all available visualizations."""
    visualizations = {
        'html': [],
        'images': [],
        'data': []
    }

    if DEMO_OUTPUT_DIR.exists():
        for file in DEMO_OUTPUT_DIR.iterdir():
            if file.suffix == '.html':
                visualizations['html'].append(file.name)
            elif file.suffix in ['.png', '.pdf', '.jpg', '.jpeg']:
                visualizations['images'].append(file.name)
            elif file.suffix in ['.json', '.txt']:
                visualizations['data'].append(file.name)

    return jsonify(visualizations)

if __name__ == '__main__':
    # Create templates directory and index.html
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)

    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BLT Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .section { margin-bottom: 30px; }
        .file-list { display: flex; flex-wrap: wrap; gap: 10px; }
        .file-item { padding: 8px 12px; background: #f0f0f0; border-radius: 4px; text-decoration: none; color: #333; }
        .file-item:hover { background: #e0e0e0; }
        h1, h2 { color: #2c3e50; }
    </style>
</head>
<body>
    <h1>Byte Latent Transformer (BLT) Visualizations</h1>

    <div class="section">
        <h2>Interactive Visualizations</h2>
        <div class="file-list">
            {% for file in html_files %}
            <a href="/visualizations/{{ file }}" class="file-item" target="_blank">{{ file }}</a>
            {% endfor %}
        </div>
    </div>

    <div class="section">
        <h2>Static Images</h2>
        <div class="file-list">
            {% for file in image_files %}
            <a href="/images/{{ file }}" class="file-item" target="_blank">{{ file }}</a>
            {% endfor %}
        </div>
    </div>

    <div class="section">
        <h2>Data Files</h2>
        <div class="file-list">
            {% for file in data_files %}
            <a href="/data/{{ file }}" class="file-item" target="_blank">{{ file }}</a>
            {% endfor %}
        </div>
    </div>
</body>
</html>
    """

    with open(templates_dir / "index.html", "w") as f:
        f.write(index_html)

    print("Starting BLT Visualization Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)