// ==================== History Page JavaScript ====================

/**
 * Delete a single record
 */
function deleteRecord(id) {
    if (!confirm('Are you sure you want to delete this record?')) {
        return;
    }

    fetch(`/history/delete/${id}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage('Deleted successfully', 'success');
            setTimeout(() => {
                location.reload();
            }, 500);
        } else {
            showMessage('Delete failed', 'error');
        }
    })
    .catch(error => {
        console.error('Delete error:', error);
        showMessage('Delete failed, please try again', 'error');
    });
}

/**
 * Clear all records
 */
function clearAllRecords() {
    if (!confirm('Are you sure you want to clear all history? This action cannot be undone!')) {
        return;
    }

    // TODO: Implement clear all records API
    showMessage('Feature under development...', 'info');
}

/**
 * Export records
 */
function exportRecords() {
    // TODO: Implement export functionality
    showMessage('Export feature under development...', 'info');
}
