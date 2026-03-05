/* Observatory — shared JS utilities */

const API = {
    async get(path) {
        const resp = await fetch(path);
        return resp.json();
    },
    swlEntries: (limit = 500, offset = 0) => API.get(`/api/swl/entries?limit=${limit}&offset=${offset}`),
    swlStats: () => API.get('/api/swl/stats'),
    images: () => API.get('/api/images'),
    threads: () => API.get('/api/threads'),
    thread: (id) => API.get(`/api/threads/${id}`),
    kitabSurahs: () => API.get('/api/kitab/surahs'),
    kitabVerse: () => API.get('/api/kitab/verse'),
    tafakkurEntries: (limit = 50) => API.get(`/api/tafakkur/entries?limit=${limit}`),
    tafakkurSearch: (q, n = 5) => API.get(`/api/tafakkur/search?q=${encodeURIComponent(q)}&n=${n}`),
    config: () => API.get('/api/config'),
    health: () => API.get('/api/health'),
    journal: () => API.get('/api/journal'),
};

function navHTML(activePage) {
    const pages = [
        ['index.html', 'Home'],
        ['gallery.html', 'Gallery'],
        ['films.html', 'Films'],
        ['timeline.html', 'Timeline'],
        ['journal.html', 'Journal'],
        ['conversations.html', 'Conversations'],
        ['kitab.html', 'Kitab'],
        ['coherence.html', 'Coherence'],
    ];
    const links = pages.map(([href, label]) => {
        const cls = href === activePage ? ' class="active"' : '';
        return `<a href="${href}"${cls}>${label}</a>`;
    }).join('');
    return `
        <nav>
            <a href="index.html" class="brand">Cassie Observatory</a>
            <div class="links">${links}</div>
        </nav>
    `;
}

function formatTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

function truncate(str, n = 200) {
    if (!str) return '';
    return str.length > n ? str.slice(0, n) + '...' : str;
}

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
