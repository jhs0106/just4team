
function previewFile(input, previewId, infoId) {
    if (input.files && input.files[0]) {
        const file = input.files[0];

        const reader = new FileReader();
        reader.onload = function(e) {

            const img = document.getElementById(previewId);
            img.src = e.target.result;

            const tempImg = new Image();
            tempImg.onload = function() {
                const width = tempImg.width;
                const height = tempImg.height;
                const sizeKB = (file.size / 1024).toFixed(2);

                document.getElementById(infoId).innerHTML =
                    `📏 ${width}x${height}px | 💾 ${sizeKB} KB`;
            };

            tempImg.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }
}

function setupDropZone(dropZoneId, inputId, previewId, infoId) {
    const dropZone = document.getElementById(dropZoneId);
    const input = document.getElementById(inputId);

    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('bg-primary', 'bg-opacity-10');
    });

    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.classList.remove('bg-primary', 'bg-opacity-10');
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('bg-primary', 'bg-opacity-10');

        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            previewFile(input, previewId, infoId);
        }
    });
}

// activar ambos
setupDropZone("dropZone1", "deskImage1", "previewImage1", "info1");
setupDropZone("dropZone2", "deskImage2", "previewImage2", "info2");