const form = document.querySelector('#app-form');
const answer = document.querySelector('#answer');
const status = document.querySelector('#status');
const openedAsFile = window.location.protocol === 'file:';

if (openedAsFile) {
  status.textContent = 'Open http://localhost:3000';
  answer.textContent = 'This app should be tested through the local server. In Terminal or PowerShell, run npm start from the project folder, then open http://localhost:3000. Do not double-click public/index.html.';
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (openedAsFile) {
    status.textContent = 'Open http://localhost:3000';
    answer.textContent = 'This app cannot call the model from a raw file page. Run npm start, then open http://localhost:3000.';
    return;
  }

  status.textContent = 'Calling Gemini...';
  answer.textContent = '';

  const data = Object.fromEntries(new FormData(form).entries());
  data.useKnowledgeBase = form.elements.useKnowledgeBase.checked;

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Request failed');

    status.textContent = `${result.model}${result.usedKnowledgeBase ? ' + docs' : ''}`;
    answer.textContent = result.answer;
  } catch (error) {
    status.textContent = 'Error';
    answer.textContent = error.message;
  }
});
