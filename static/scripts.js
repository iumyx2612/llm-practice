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

document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('file-input').files[0];

    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = 'Uploading...';

    const formData = new FormData();
    formData.append('file', fileInput);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        responseDiv.innerHTML = `File uploaded successfully: ${data.filename}`;
    } catch (error) {
        responseDiv.innerHTML = `Error: ${error.message}`;
    }
});
