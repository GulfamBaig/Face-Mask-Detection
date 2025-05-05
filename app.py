/* style.css */
/* Main styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Upload container */
.upload-container {
    background: #f5f5f5;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Result containers */
.result-container {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    transition: all 0.3s ease;
}

.result-container.safe {
    background: #e8f5e9;
    border-left: 5px solid #4caf50;
}

.result-container.danger {
    background: #ffebee;
    border-left: 5px solid #f44336;
}

/* Confidence meter */
.confidence-meter {
    height: 25px;
    background: #e0e0e0;
    border-radius: 12px;
    margin: 15px 0;
    position: relative;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 12px;
    transition: width 0.5s ease;
}

.confidence-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-weight: bold;
    text-shadow: 0 0 3px rgba(0,0,0,0.3);
}

.processing-time {
    font-size: 14px;
    color: #757575;
    text-align: right;
    margin-top: 10px;
}

/* Info cards */
.info-container {
    padding: 15px;
}

.info-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.info-card h4 {
    color: #1e88e5;
    margin-top: 0;
}

.info-card.warning {
    background: #fff3e0;
    border-left: 5px solid #ffa000;
}

/* Button styles */
.stButton>button {
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 500;
    transition: all 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* File uploader customization */
.stFileUploader>div>div>div>div {
    border: 2px dashed #1e88e5;
    border-radius: 10px;
    padding: 30px;
    background: #f5f5f5;
}

.stFileUploader>div>div>div>div:hover {
    border-color: #0d47a1;
}
