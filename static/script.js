// Constants
const API_ENDPOINTS = {
    SINGLE: '/ocr',
    BATCH: '/batch'
};

const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB
const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff'];

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const errorSection = document.getElementById('errorSection');
const errorText = document.getElementById('errorText');
const copyAllBtn = document.getElementById('copyAllBtn');
const downloadBtn = document.getElementById('downloadBtn');

// Event Listeners
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
copyAllBtn.addEventListener('click', copyAllResults);
downloadBtn.addEventListener('click', downloadResults);

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
    
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
}

// File Processing
function processFiles(files) {
    // Validate files
    const validFiles = files.filter(file => {
        if (!ALLOWED_TYPES.includes(file.type)) {
            showError(`File type not supported: ${file.name}`);
            return false;
        }
        if (file.size > MAX_FILE_SIZE) {
            showError(`File too large: ${file.name} (max 16MB)`);
            return false;
        }
        return true;
    });

    if (validFiles.length === 0) return;

    // Show progress
    showProgress();
    updateProgress(0);

    // Get processing options
    const options = {
        enhance: document.getElementById('enhanceImage').checked,
        denoise: document.getElementById('denoiseImage').checked,
        correct_skew: document.getElementById('correctSkew').checked
    };

    // Process files
    if (validFiles.length === 1) {
        processSingleFile(validFiles[0], options);
    } else {
        processBatchFiles(validFiles, options);
    }
}

async function processSingleFile(file, options) {
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('enhance', options.enhance);
        formData.append('denoise', options.denoise);
        formData.append('correct_skew', options.correct_skew);

        const response = await fetch(API_ENDPOINTS.SINGLE, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResults([{ filename: file.name, ...result }]);
    } catch (error) {
        showError(`Error processing ${file.name}: ${error.message}`);
    } finally {
        hideProgress();
    }
}

async function processBatchFiles(files, options) {
    try {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files[]', file);
        });
        formData.append('enhance', options.enhance);
        formData.append('denoise', options.denoise);
        formData.append('correct_skew', options.correct_skew);

        const response = await fetch(API_ENDPOINTS.BATCH, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result.results);
    } catch (error) {
        showError(`Error processing batch: ${error.message}`);
    } finally {
        hideProgress();
    }
}

// Results Display
function displayResults(results) {
    resultsContainer.innerHTML = '';
    results.forEach(result => {
        const resultElement = document.createElement('div');
        resultElement.className = 'result-item';
        
        const header = document.createElement('div');
        header.className = 'result-header';
        header.innerHTML = `
            <h4>${result.filename}</h4>
            <button class="copy-btn" onclick="copyText(this)">
                <i class="fas fa-copy"></i>
            </button>
        `;
        
        const content = document.createElement('div');
        content.className = 'result-content';
        content.innerHTML = `
            <pre>${result.text || 'No text extracted'}</pre>
            ${result.error ? `<p class="error">${result.error}</p>` : ''}
            ${result.metadata ? `
                <div class="metadata">
                    <p>Model: ${result.metadata.model}</p>
                    <p>Device: ${result.metadata.device}</p>
                    <p>Image Size: ${result.metadata.image_size?.join('x') || 'N/A'}</p>
                </div>
            ` : ''}
        `;
        
        resultElement.appendChild(header);
        resultElement.appendChild(content);
        resultsContainer.appendChild(resultElement);
    });
    
    resultsSection.style.display = 'block';
}

// Utility Functions
function showProgress() {
    progressSection.style.display = 'block';
    errorSection.style.display = 'none';
}

function hideProgress() {
    progressSection.style.display = 'none';
}

function updateProgress(percent) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = `Processing... ${percent}%`;
}

function showError(message) {
    errorText.textContent = message;
    errorSection.style.display = 'block';
    setTimeout(() => {
        errorSection.style.display = 'none';
    }, 5000);
}

function copyText(button) {
    const textContainer = button.parentElement.nextElementSibling.querySelector('pre');
    if (!textContainer) return;
    
    const text = textContainer.textContent;
    navigator.clipboard.writeText(text).then(() => {
        button.innerHTML = '<i class="fas fa-check"></i>';
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-copy"></i>';
        }, 2000);
    });
}

function copyAllResults() {
    const allText = Array.from(resultsContainer.querySelectorAll('pre'))
        .map(pre => pre.textContent)
        .join('\n\n');
    
    navigator.clipboard.writeText(allText).then(() => {
        copyAllBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        setTimeout(() => {
            copyAllBtn.innerHTML = '<i class="fas fa-copy"></i> Copy All';
        }, 2000);
    });
}

function downloadResults() {
    const allText = Array.from(resultsContainer.querySelectorAll('pre'))
        .map(pre => pre.textContent)
        .join('\n\n');
    
    const blob = new Blob([allText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ocr-results.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}