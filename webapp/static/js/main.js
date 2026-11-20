// Quant Strategy System - Shared JS
function updateClock() {
    const now = new Date();
    const el = document.getElementById('clock');
    if (el) el.textContent = now.toLocaleTimeString('zh-CN');
}
setInterval(updateClock, 1000);
updateClock();

// Strategy toggle
async function toggleStrategy(sid, button) {
    button.disabled = true;
    try {
        const response = await fetch('/api/strategy/' + sid + '/toggle', {method: 'POST'});
        if (!response.ok) throw new Error();
        location.reload();
    } catch {
        button.disabled = false;
        button.textContent = '重试';
    }
}

async function refreshQuotes(button) {
    button.disabled = true;
    button.textContent = '刷新中…';
    try {
        const response = await fetch('/api/refresh', {cache: 'no-store'});
        const result = await response.json();
        if (response.ok && result.status === 'ok') location.reload();
        else button.textContent = result.previous_cache_kept ? '刷新失败，已保留快照' : '刷新失败，无可用缓存';
    } catch {
        button.textContent = '刷新失败，请重试';
    } finally {
        button.disabled = false;
    }
}
