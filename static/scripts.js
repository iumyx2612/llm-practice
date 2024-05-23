document.getElementById('query-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const queryInput = document.getElementById('query-input').value;

    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = 'Loading...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: queryInput }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        responseDiv.innerHTML = `Response: ${data.response}`;
    } catch (error) {
        responseDiv.innerHTML = `Error: ${error.message}`;
    }
});
