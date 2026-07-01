const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');

let selectedFile = null;

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

dropzone.addEventListener('click', (e) => {
    if (e.target === dropzone || e.target.closest('.dropzone-content')) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        processBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    processBtn.disabled = true;
    processBtn.textContent = 'Обработка...';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert('Ошибка: ' + data.error);
            return;
        }

        document.getElementById('totalCars').textContent = data.total_cars || 0;
        document.getElementById('totalLights').textContent = data.total_traffic_lights || 0;
        document.getElementById('violations').textContent = data.violations || 0;
        document.getElementById('processingTime').textContent = data.processing_time || 0;
        document.getElementById('redLights').textContent = data.red_lights || 0;
        document.getElementById('yellowLights').textContent = data.yellow_lights || 0;
        document.getElementById('greenLights').textContent = data.green_lights || 0;

        resultImage.src = data.result_image + '?' + Date.now();
        resultsSection.style.display = 'block';

        loadHistory();
    } catch (error) {
        alert('Ошибка при обработке: ' + error.message);
    } finally {
        processBtn.disabled = false;
        processBtn.textContent = 'Запустить обработку';
    }
});

async function loadHistory() {
    try {
        const response = await fetch('/history');
        const history = await response.json();

        const container = document.getElementById('historyContainer');

        if (history.length === 0) {
            container.innerHTML = '<p class="loading">Нет записей в истории</p>';
            return;
        }

        container.innerHTML = history.map(item => `
            <div class="history-item">
                <span>${item.timestamp}</span>
                <span>Авто: ${item.stats.total_cars || 0} | Нарушения: ${item.stats.violations || 0}</span>
                <span>Время: ${item.processing_time}с</span>
                <img src="${item.stats.result_image}" alt="Результат">
                <a href="${item.stats.result_image}" download class="btn-secondary">Скачать</a>
            </div>
        `).join('');
    } catch (error) {
        console.error('Ошибка загрузки истории:', error);
    }
}

loadHistory();
