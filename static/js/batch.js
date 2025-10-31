// ==================== Batch Analysis Page JavaScript ====================

(function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const loadingMessage = document.getElementById('loadingMessage');

    // Exit if not on upload page
    if (!uploadArea || !fileInput) return;

    let isUploading = false;  // Flag to prevent duplicate uploads

    // Click upload area to trigger file selection
    uploadArea.addEventListener('click', function(e) {
        if (e.target !== fileInput && !isUploading) {
            fileInput.click();
        }
    });

    // Drag upload - dragover
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('drag-over');
    });

    // Drag upload - dragleave
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');
    });

    // Drag upload - drop
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');

        if (!isUploading && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File selection
    fileInput.addEventListener('change', function(e) {
        if (!isUploading && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    /**
     * Handle file upload
     */
    function handleFile(file) {
        // Prevent duplicate uploads
        if (isUploading) {
            console.log('Upload in progress, please do not repeat');
            return;
        }

        // Validate file format
        const fileExt = file.name.split('.').pop().toLowerCase();
        if (!['csv', 'xlsx', 'xls'].includes(fileExt)) {
            showMessage('Unsupported file format, please upload CSV or Excel file', 'error');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            showMessage('File size exceeds 16MB limit', 'error');
            return;
        }

        // Set upload status
        isUploading = true;
        uploadArea.style.display = 'none';
        loadingMessage.style.display = 'block';

        // Create FormData
        const formData = new FormData();
        formData.append('file', file);

        // Upload file
        fetch('/batch', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showMessage('Analysis complete! Redirecting...', 'success');
                setTimeout(() => {
                    window.location.href = '/batch?id=' + data.analysis_id;
                }, 500);
            } else {
                throw new Error(data.message || 'Analysis failed');
            }
        })
        .catch(error => {
            console.error('Upload error:', error);
            showMessage(error.message || 'Upload failed, please try again', 'error');

            // Restore upload interface
            uploadArea.style.display = 'block';
            loadingMessage.style.display = 'none';
            isUploading = false;
            fileInput.value = '';  // Clear file selection
        });
    }
})();
