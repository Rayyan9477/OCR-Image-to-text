// This function will be called when the copy button is clicked
function copyText(button) {
    // Find the text container (pre element) within the parent element of the button
    const textContainer = button.parentElement.querySelector('pre');
    if (!textContainer) {
        console.error('Text container not found');
        button.innerHTML = '❌';
        setTimeout(() => button.innerHTML = '📋', 2000);
        return;
    }
    
    // Get the text to copy
    const text = textContainer.innerText || textContainer.textContent;
    
    if (navigator.clipboard && window.isSecureContext) {
        // Modern browsers in secure context
        navigator.clipboard.writeText(text)
            .then(() => {
                button.innerHTML = '✓';
                setTimeout(() => button.innerHTML = '📋', 2000);
            })
            .catch(err => {
                console.error('Failed to copy:', err);
                fallbackCopyText(text, button);
            });
    } else {
        // Fallback for older browsers or non-secure context
        fallbackCopyText(text, button);
    }
}

function fallbackCopyText(text, button) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    
    try {
        textArea.select();
        document.execCommand('copy');
        button.innerHTML = '✓';
        setTimeout(() => button.innerHTML = '📋', 2000);
    } catch (err) {
        console.error('Fallback copy failed:', err);
        button.innerHTML = '❌';
        setTimeout(() => button.innerHTML = '📋', 2000);
    } finally {
        document.body.removeChild(textArea);
    }
}