import {
    modelFeatures,
    catFeatures,
    featureOptions,
    loadModelInfo,
    loadFeatureOptions,
    submitForm
} from './loads.js';


// DOM элементы
let formFields, randomizeBtn, submitBtn, loading, result;

let autofillBtn;
let isAutofillRunning = false;
let autofillInterval;
const TARGET_PROBABILITY = 0.5;
let speedControl, speedValue;

// Инициализация приложения
export async function initPredictor() {
    try {
        // Получаем DOM элементы
        formFields = document.getElementById('form-fields');
        randomizeBtn = document.getElementById('randomize');
        submitBtn = document.getElementById('submit');
        loading = document.getElementById('loading');
        result = document.getElementById('result');
        autofillBtn = document.getElementById('autofill');
        autofillBtn.addEventListener('click', startAutofill);
        speedControl = document.getElementById('speed');
        speedValue = document.getElementById('speed-value');

        speedControl.addEventListener('input', () => {
        speedValue.textContent = `${speedControl.value} мс`;
    });
        // Загружаем информацию о модели
        await loadModelInfo();
        
        // Загружаем доступные значения для фичей
        await loadFeatureOptions();
        
        // Генерируем поля формы
        generateFormFields();
        
        // Назначаем обработчики событий
        randomizeBtn.addEventListener('click', submitRandomly);
        submitBtn.addEventListener('click', submitForm);
        
    } catch (error) {
        console.error('Ошибка инициализации:', error);
        alert('Произошла ошибка при загрузке данных. Пожалуйста, обновите страницу.');
    }
}

function submitRandomly() {
    fillRandomly()
    submitForm()
}


// Генерация полей формы
function generateFormFields() {
    let html = '';

    modelFeatures.forEach(feature => {
        html += `<div class="form-group">
                  <label for="${feature}">${feature}</label>`;

        if (catFeatures.includes(feature) && featureOptions[feature]) {
            // Для категориальных фичей создаем select
            html += `<select id="${feature}" required>
                      <option value="">Выберите значение</option>`;

            featureOptions[feature].forEach(value => {
                html += `<option value="${value}">${value}</option>`;
            });

            html += `</select>`;
        } else {
            // Для числовых фичей создаем input
            html += `<input type="${feature === 'visit_number' ? 'number' : 'text'}" 
                      id="${feature}" required>`;
        }

        html += `</div>`;
    });

    formFields.innerHTML = html;
}

// Случайное заполнение формы
function fillRandomly() {
    modelFeatures.forEach(feature => {
        const element = document.getElementById(feature);
        if (featureOptions[feature]) {
            const randomIndex = Math.floor(Math.random() * featureOptions[feature].length);
            element.value = featureOptions[feature][randomIndex];
        } else if (feature === 'visit_number') {
            element.value = 1;
        } else {
            element.value = 'test_value';
        }
    });
}

function startAutofill() {
    if (isAutofillRunning) {
        stopAutofill();
        return;
    }

    isAutofillRunning = true;
    autofillBtn.textContent = 'Остановить автоподбор';
    autofillBtn.classList.add('running');
    loading.style.display = 'block';

    const speed = parseInt(speedControl.value);

    autofillInterval = setInterval(async () => {
        fillRandomly();
        const proba = await submitForm(true);

        if (proba >= TARGET_PROBABILITY) {
            stopAutofill();
            showNotification(`Найдено значение с вероятностью ${proba.toFixed(4)}`);
        }
    }, speed);
}

function showNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 500);
    }, 3000);
}

function stopAutofill() {
    isAutofillRunning = false;
    clearInterval(autofillInterval);
    autofillBtn.textContent = 'Автоподбор (≥0.5)';
    loading.style.display = 'none';
}
