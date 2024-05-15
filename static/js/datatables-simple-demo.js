window.addEventListener('DOMContentLoaded', event => {
    const datatablesSimple = document.getElementById('datatablesSimple');
    if (datatablesSimple) {
        new simpleDatatables.DataTable(datatablesSimple, {
            perPage: 10, // Set number of rows per page
            perPageSelect: [5, 10, 15, 20] // Dropdown options for rows per page
        });
    }
});
