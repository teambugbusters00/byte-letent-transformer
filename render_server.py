#!/usr/bin/env python3
"""
Simple Flask server for serving BLT visualizations.
This provides a web interface to view the generated HTML visualizations and plots.
Production-ready for deployment on Render.
"""

import os
from pathlib import Path
from flask import Flask, render_template_string, send_from_directory, jsonify
import json

app = Flask(__name__)

# Path to demo_output directory
DEMO_OUTPUT_DIR = Path(__file__).parent / "demo_output"
PLOT_DATA_DIR = Path(__file__).parent / "plot_data"

# HTML template as string for production deployment
INDEX_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BLT Visualizations</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .section {
            margin: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .section:last-child {
            border-bottom: none;
        }
        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            font-weight: 400;
        }
        .file-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }
        .file-item {
            display: block;
            padding: 15px 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 8px;
            text-decoration: none;
            color: #2c3e50;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            font-weight: 500;
        }
        .file-item:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .file-item.html {
            border-left: 4px solid #e74c3c;
        }
        .file-item.image {
            border-left: 4px solid #27ae60;
        }
        .file-item.data {
            border-left: 4px solid #3498db;
        }
        .empty-state {
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            padding: 40px;
        }
        .footer {
            background: #34495e;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }
        .footer a {
            color: #3498db;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Byte Latent Transformer</h1>
            <p>Interactive Visualizations & Analysis Dashboard</p>
        </div>

        <div class="section">
            <h2>📊 Interactive Visualizations</h2>
            {% if html_files %}
            <div class="file-grid">
                {% for file in html_files %}
                <a href="/visualizations/{{ file }}" class="file-item html" target="_blank">
                    📄 {{ file }}
                </a>
                {% endfor %}
            </div>
            {% else %}
            <div class="empty-state">
                No interactive visualizations available. Run the BLT demos to generate HTML outputs.
            </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>🖼️ Static Images & Plots</h2>
            {% if image_files %}
            <div class="file-grid">
                {% for file in image_files %}
                <a href="/images/{{ file }}" class="file-item image" target="_blank">
                    🖼️ {{ file }}
                </a>
                {% endfor %}
            </div>
            {% else %}
            <div class="empty-state">
                No static images available. Run the BLT plotting scripts to generate visualizations.
            </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>📋 Data Files & Results</h2>
            {% if data_files %}
            <div class="file-grid">
                {% for file in data_files %}
                <a href="/data/{{ file }}" class="file-item data" target="_blank">
                    📋 {{ file }}
                </a>
                {% endfor %}
            </div>
            {% else %}
            <div class="empty-state">
                No data files available. Run BLT processing to generate analysis results.
            </div>
            {% endif %}
        </div>

        <div class="footer">
            <p>
                Powered by <a href="https://github.com/teambugbusters00/byte-letent-transformer" target="_blank">Byte Latent Transformer</a> |
                Built with Flask & Deployed on Render
            </p>
        </div>
    </div>
</body>
</html>
"""

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

    return render_template_string(INDEX_HTML_TEMPLATE,
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
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get('PORT', 5000))

    print("Starting BLT Visualization Server...")
    print(f"Server will run on port {port}")
    print(f"Open http://localhost:{port} in your browser")

    # Production settings
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )