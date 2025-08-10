document.addEventListener('DOMContentLoaded', function () {
    const uploadContainer = document.getElementById('uploadContainer');
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');

    let isProcessing = false;
    let spinner = null;

    const initIcons = () => {
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewArea.innerHTML = `
                <div class="space-y-4">
                    <img src="${e.target.result}" alt="Preview" class="max-h-64 max-w-full mx-auto rounded-lg object-contain">
                    <p class="text-sm text-gray-500">Click to change image</p>
                </div>
            `;
            initIcons();
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    };

    const uploadImage = (file) => {
        if (isProcessing) return;
        isProcessing = true;

        // Show spinner
        if (spinner && document.body.contains(spinner)) {
            document.body.removeChild(spinner);
        }

        spinner = document.createElement('div');
        spinner.id = 'loading-spinner';
        spinner.className = 'fixed inset-0 bg-black/30 flex items-center justify-center z-50';
        spinner.innerHTML = `
            <div class="bg-white p-6 rounded-lg text-center">
                <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-green-500 mx-auto"></div>
                <p class="mt-4 text-gray-700">Processing image...</p>
            </div>
        `;
        document.body.appendChild(spinner);

        const formData = new FormData();
        formData.append('image', file);

        // Prevent default form submission behavior
        event.preventDefault();

        fetch('/recipe-finder', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // Check if the response is ok
            if (!response.ok) {
                throw new Error(`Failed to process the image. Status: ${response.status}`);
            }
            return response.text();
        })
        .then(html => {
            // Update the page content
            document.body.innerHTML = html;
        })
        .catch(error => {
            console.error('Error during image processing:', error);
            alert(`Error processing image: ${error.message}`);
        })
        .finally(() => {
            isProcessing = false;
            if (spinner && document.body.contains(spinner)) {
                document.body.removeChild(spinner);
                spinner = null;
            }
        });
    };

    // Add listeners if upload container is present (only on upload page)
    if (uploadContainer && fileInput && previewArea) {
        uploadContainer.addEventListener('click', () => {
            if (!isProcessing) fileInput.click();
        });

        fileInput.addEventListener('change', (event) => {
            if (!isProcessing && event.target.files[0]) {
                handleFileChange(event);
            }
        });
    }

    // Always initialize icons (for results.html or others)
    initIcons();
});
