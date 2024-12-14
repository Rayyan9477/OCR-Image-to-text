function copyText(button) {
    const textContainer = button.parentElement.querySelector('pre');
    const text = textContainer.innerText;
    navigator.clipboard.writeText(text).then(() => {
        button.innerHTML = '✓';
        setTimeout(() => {
            button.innerHTML = '📋';
        }, 2000);
    });
}