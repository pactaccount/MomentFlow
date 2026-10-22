const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const fmtTime = sec => { const n = Math.max(0, Math.floor(Number(sec)||0)); return Math.floor(n/3600) ? `${Math.floor(n/3600)}:${String(Math.floor(n%3600/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}` : `${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`; };
let videos = [], currentQuery = '', currentHits = [], selectedHit = null, refreshTimer = null;

async function request(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }
  return data;
}
function showView(view) {
  $('library-view').hidden = view !== 'library';
  $('search-view').hidden = view !== 'search';
  document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === view));
}
function thumb(id, className) {
  return `<img class="${className}" src="/api/thumbnail/${encodeURIComponent(id)}" alt="" loading="lazy">`;
}
async function refreshVideos() {
  const result = await request('/api/videos');
  videos = result.videos;
  $('count-all').textContent = videos.length;
  $('count-ready').textContent = videos.filter(v => v.status === 'ready').length;
  $('count-processing').textContent = videos.filter(v => ['queued','indexing'].includes(v.status)).length;
  $('library-count').textContent = `${videos.length} video${videos.length === 1 ? '' : 's'}`;
  $('videos').innerHTML = videos.length ? videos.map(v => `
    <div class="video-row">
      ${thumb(v.id,'video-thumb')}
      <div><div class="video-title">${escapeHtml(v.filename)}</div>
        <div class="video-meta"><span class="status-badge ${escapeHtml(v.status)}">${escapeHtml(v.status)}</span><span>${v.duration_sec ? fmtTime(v.duration_sec) : '—'} duration</span><span>${v.indexed_version === 'multimodal-v1' ? 'Visual · Speech · Screen text' : 'Visual index'}</span></div>
        ${v.status === 'queued' || v.status === 'indexing' ? `<div class="video-progress">${escapeHtml(v.progress || 'Processing')}</div>` : ''}
        ${v.error ? `<div class="video-progress" style="color:#b73f3a">${escapeHtml(v.error)}</div>` : ''}
      </div>
      <div class="row-actions">
        ${v.status === 'ready' ? `<button class="text-button watch-video" data-id="${escapeHtml(v.id)}" type="button">Watch</button>` : ''}
        <button class="text-button danger delete-video" data-id="${escapeHtml(v.id)}" type="button">Delete</button>
      </div>
    </div>`).join('') : '<div class="empty-state"><span>▦</span><strong>Your library is ready</strong><p>Upload your first video to start finding moments.</p><button class="primary" id="empty-upload" type="button">＋ Upload video</button></div>';
  $('videos').querySelectorAll('.watch-video').forEach(button => button.addEventListener('click', () => openVideo(button.dataset.id, 0)));
  $('videos').querySelectorAll('.delete-video').forEach(button => button.addEventListener('click', () => removeVideo(button.dataset.id)));
  $('empty-upload')?.addEventListener('click', () => $('upload-dialog').showModal());
  clearTimeout(refreshTimer);
  if (videos.some(v => ['queued','indexing'].includes(v.status))) refreshTimer = setTimeout(() => refreshVideos().catch(showError), 3000);
}
function showError(error) {
  console.error(error);
  if (!videos.length) {
    $('videos').innerHTML = `<div class="empty-state"><span>!</span><strong>Could not load the library</strong><p>${escapeHtml(error.message)}</p><button class="primary" id="retry-load" type="button">Retry</button></div>`;
    $('retry-load').addEventListener('click', () => refreshVideos().catch(showError));
  }
}
function renderResults() {
  $('results').innerHTML = currentHits.length ? currentHits.map((hit, i) => `
    <button class="result-row ${selectedHit === i ? 'active' : ''}" data-index="${i}" type="button">
      ${thumb(hit.video_id,'result-thumb')}
      <span><span class="result-title">${escapeHtml(hit.filename)}</span><span class="result-meta"><span class="source-tag">${escapeHtml(hit.source)}</span> ${fmtTime(hit.timestamp_sec)} · ${hit.sources.length} signal${hit.sources.length === 1 ? '' : 's'}</span><span class="result-evidence">${escapeHtml(hit.evidence)}</span></span>
    </button>`).join('') : '<div class="empty-state"><span>⌕</span><strong>No matching moment yet</strong><p>Try a visible object, spoken phrase, or words shown on screen.</p></div>';
  $('results').querySelectorAll('.result-row').forEach(button => button.addEventListener('click', () => showHit(Number(button.dataset.index))));
}
function showHit(index) {
  const hit = currentHits[index]; if (!hit) return;
  selectedHit = index; renderResults();
  $('viewer').innerHTML = `<video id="moment-player" controls preload="metadata" src="${hit.video_url}"></video>
    <div class="viewer-info"><p class="eyebrow">MATCHED MOMENT · ${fmtTime(hit.timestamp_sec)}</p><h2>${escapeHtml(hit.filename)}</h2><p><span class="source-tag">${escapeHtml(hit.source)}</span> ${escapeHtml(hit.evidence)}</p>
    <div class="viewer-actions"><button id="share-button" class="subtle-button" type="button">↗ Copy moment link</button><button id="feedback-button" class="subtle-button" type="button">Flag wrong moment</button></div><p id="viewer-message" class="message" role="status"></p></div>`;
  const player = $('moment-player');
  player.addEventListener('loadedmetadata', () => { player.currentTime = hit.timestamp_sec; }, {once:true});
  $('share-button').addEventListener('click', async () => {
    const link = new URL(location.origin + location.pathname);
    link.searchParams.set('video',hit.video_id); link.searchParams.set('t',String(hit.timestamp_sec));
    try { await navigator.clipboard.writeText(link.toString()); $('viewer-message').textContent = 'Moment link copied.'; }
    catch { $('viewer-message').textContent = link.toString(); }
  });
  $('feedback-button').addEventListener('click', async () => {
    try {
      await request('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_id:hit.video_id,query:currentQuery,timestamp_sec:hit.timestamp_sec,value:'wrong_moment'})});
      $('viewer-message').textContent = 'Thanks — this result was flagged for review.';
    } catch(error) { $('viewer-message').textContent = error.message; }
  });
}
function openVideo(id, t) {
  const video = videos.find(v => v.id === id);
  if (!video || video.status !== 'ready') return;
  currentQuery = ''; currentHits = [{video_id:id,filename:video.filename,timestamp_sec:t,source:'library',sources:['library'],evidence:'Selected from your library',video_url:`/api/media/${id}`,thumbnail_url:`/api/thumbnail/${id}`}];
  $('result-heading').textContent = video.filename;
  $('search-status').textContent = 'Playing from your library';
  showView('search'); showHit(0);
}
async function removeVideo(id) {
  const video = videos.find(v => v.id === id);
  if (!confirm(`Delete "${video?.filename || 'video'}" and its search index?`)) return;
  try { await request('/api/videos/'+encodeURIComponent(id),{method:'DELETE'}); await refreshVideos(); }
  catch(error) { alert(error.message); }
}
async function search(query) {
  currentQuery = query;
  showView('search');
  $('result-heading').textContent = `Results for “${query}”`;
  $('search-status').textContent = 'Searching your video library…';
  $('results').innerHTML = '';
  $('viewer').innerHTML = '<div class="viewer-placeholder"><span>⌕</span><strong>Finding moments</strong></div>';
  try {
    const data = await request('/api/search?q='+encodeURIComponent(query));
    currentHits = data.results; selectedHit = null;
    $('search-status').textContent = `${currentHits.length} candidate moment${currentHits.length === 1 ? '' : 's'}`;
    renderResults();
    if (currentHits.length) showHit(0);
  } catch(error) { $('search-status').textContent = error.message; }
}
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
$('search-form').addEventListener('submit', event => { event.preventDefault(); const query=$('query').value.trim(); if(query) search(query); });
$('upload-open').addEventListener('click', () => $('upload-dialog').showModal());
$('upload-close').addEventListener('click', () => $('upload-dialog').close());
$('refresh-button').addEventListener('click', () => refreshVideos().catch(showError));
$('file').addEventListener('change', () => { $('selected-file').textContent = $('file').files[0]?.name || ''; });
$('upload-form').addEventListener('submit', async event => {
  event.preventDefault(); const file=$('file').files[0]; if(!file) return;
  $('upload-status').textContent = 'Uploading and validating video…';
  try {
    const form=new FormData(); form.append('file',file);
    const data=await request('/api/upload',{method:'POST',body:form});
    $('upload-status').textContent = data.status === 'already_uploaded' ? 'This video is already in the library.' : 'Upload complete. Indexing has started.';
    $('upload-form').reset(); $('selected-file').textContent = '';
    await refreshVideos(); setTimeout(() => $('upload-dialog').close(),1000);
  } catch(error) { $('upload-status').textContent = error.message; }
});
function followSharedLink() {
  const params = new URLSearchParams(location.search);
  const id=params.get('video'), t=Number(params.get('t')||0);
  if(id) openVideo(id,Number.isFinite(t)?t:0);
}
if (location.protocol === 'file:') {
  $('app-shell').hidden = true;
  $('launch-screen').hidden = false;
} else {
  refreshVideos().then(followSharedLink).catch(showError);
}
