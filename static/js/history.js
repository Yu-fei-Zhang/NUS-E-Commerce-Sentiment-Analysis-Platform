// ==================== History Page JavaScript ====================

/**
 * View details of a record
 */
function viewDetails(id, type) {
    if (type === 'single') {
        window.location.href = `/single?id=${id}`;
    } else if (type === 'batch') {
        window.location.href = `/batch?id=${id}`;
    }
}

/**
 * Delete a single record
 */
function deleteRecord(id) {
    // Stop event propagation to prevent row click
    event.stopPropagation();

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
