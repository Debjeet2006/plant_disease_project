// Language switching function
function changeLanguage(lang) {
    const elements = document.querySelectorAll("[data-en]");

    elements.forEach(el => {
        let text = el.getAttribute(`data-${lang}`);
        if (text) {
            el.textContent = text;
        }
    });
}

// Optional: fallback to English if invalid
function safeChangeLanguage(lang) {
    if (["en", "hi"].includes(lang)) {
        changeLanguage(lang);
    } else {
        changeLanguage("en");
    }
}