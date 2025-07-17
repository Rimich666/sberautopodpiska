export let modelFeatures = [];
export let catFeatures = [];
export let featureOptions = {};

const base_url = 'http://localhost:8800/api/v1'

export async function loadModelInfo() {
    const response = await fetch(`${base_url}/model_info`);
    const data = await response.json();

    document.getElementById('model-type').textContent = data.model_type;
    document.getElementById('current-model').textContent = data.current_model;
    document.getElementById('loaded-at').textContent = data.loaded_at;

    modelFeatures = data.features;
    catFeatures = data.cat_features;
}

export async function loadFeatureOptions() {
  const response = await fetch(`${base_url}/features`);
  const data = await response.json();
  console.log(data);
  featureOptions = data;
}

export async function submitForm(isAutofill = false) {
    const formData = {};
    modelFeatures.forEach(feature => {
        const element = document.getElementById(feature);
        formData[feature] = element.value;
    });

    // Для автоподбора пропускаем проверку заполненности
    if (!isAutofill && Object.values(formData).some(value => !value)) {
        alert('Пожалуйста, заполните все поля формы');
        return;
    }

    try {
        if (!isAutofill) {
            loading.style.display = 'block';
            result.style.display = 'none';
        }

        const response = await fetch(`${base_url}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error(`Ошибка HTTP: ${response.status}`);

        const data = await response.json();

        document.getElementById('probability').textContent = data.probability.toFixed(16);
        document.getElementById('prediction').textContent = data.prediction;
        document.getElementById('timestamp').textContent = data.timestamp;
        result.style.display = 'block';

        return data.probability;

    } catch (error) {
        console.error('Ошибка при отправке формы:', error);
        if (!isAutofill) alert(`Произошла ошибка: ${error.message}`);
        return 0;
    } finally {
        if (!isAutofill) loading.style.display = 'none';
    }
}