function copyText(button) {
    const text = button.parentElement.innerText;
    navigator.clipboard.writeText(text).then(() => {
        button.innerHTML = '✓';
        setTimeout(() => {
            button.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                </svg>`;
        }, 2000);
    });
}