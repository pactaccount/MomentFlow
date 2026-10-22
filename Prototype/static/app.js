const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const fmtTime = sec => {
  const n = Math.max(0, Math.floor(Number(sec) || 0));
  return n >= 3600
    ? `${Math.floor(n/3600)}:${String(Math.floor(n%3600/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`
    : `${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`;
};
let videos = [], currentQuery = '', currentHits = [], selectedHit = null, refreshTimer = null;

async function request(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
function showView(view) {
  $('library-view').hidden = view !== 'library';
  $('search-view').hidden = view !== 'search';
  $('breadcrumb-current').textContent = view === 'library' ? 'Video library' : 'Moment search';
  document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === view));
  if (view === 'search' && !currentQuery && !currentHits.length) renderResults();
}
function thumbnail(id, className, ready = true) {
  return ready
    ? `<img class="${className}" src="/api/thumbnail/${encodeURIComponent(id)}" alt="" loading="lazy">`
    : `<span class="${className}" aria-hidden="true"></span>`;
}
function statusLabel(status) {
  return ({ready:'Ready',queued:'Queued',indexing:'Indexing',failed:'Failed'})[status] || 'Unknown';
}
function renderLibrary() {
  const needle = $('filter-input').value.trim().toLowerCase();
  const status = $('status-filter').value;
  const shown = videos.filter(video => {
    const nameMatches = video.filename.toLowerCase().includes(needle);
    const statusMatches = status === 'all' || (status === 'processing'
      ? ['queued','indexing'].includes(video.status) : video.status === status);
    return nameMatches && statusMatches;
  });
  const container = $('videos');
  if (!videos.length) {
    container.innerHTML = `<div class="empty-state"><span class="empty-icon" aria-hidden="true">▦</span><strong>Your library is ready</strong><p>Upload a recording or load our original nine-second sample. After it is Ready, search for “red car”.</p><button class="primary" id="empty-upload" type="button">＋ Upload video</button> <button class="subtle-button" id="load-demo" type="button">Load sample video</button><p id="demo-status" class="message" role="status"></p></div>`;
    $('empty-upload').addEventListener('click', () => $('upload-dialog').showModal());
    $('load-demo').addEventListener('click', loadDemoSample);
    return;
  }
  if (!shown.length) {
    container.innerHTML = `<div class="empty-state"><span class="empty-icon" aria-hidden="true">⌕</span><strong>No recordings match</strong><p>Try another filename or processing status.</p><button class="subtle-button" id="clear-filters" type="button">Clear filters</button></div>`;
    $('clear-filters').addEventListener('click', () => {
      $('filter-input').value = '';
      $('status-filter').value = 'all';
      renderLibrary();
    });
    return;
  }
  container.innerHTML = shown.map(video => {
    const busy = ['queued','indexing'].includes(video.status);
    const safeStatus = ['ready','queued','indexing','failed'].includes(video.status) ? video.status : 'unknown';
    return `<div class="video-row">
      <div class="video-main">${thumbnail(video.id,'video-thumb',video.status === 'ready')}
        <div class="video-info"><div class="video-title" title="${escapeHtml(video.filename)}">${escapeHtml(video.filename)}</div>
          <div class="video-meta">${video.created_at ? new Date(video.created_at*1000).toLocaleDateString() : 'Uploaded video'}</div>
          ${busy ? `<div class="video-progress">${escapeHtml(video.progress || 'Preparing for search')}</div>` : ''}
          ${video.error ? `<div class="video-error">${escapeHtml(video.error)}</div>` : ''}
        </div>
      </div>
      <div class="status-cell"><span class="status-badge ${safeStatus}">${statusLabel(video.status)}</span></div>
      <div class="duration-cell">${video.duration_sec ? fmtTime(video.duration_sec) : '—'}</div>
      <div class="index-cell">${video.status === 'ready' ? 'Multimodal index' : busy ? 'Building index' : 'Unavailable'}</div>
      <div class="row-actions">
        ${video.status === 'ready' ? `<button class="text-button watch-video" data-id="${escapeHtml(video.id)}" type="button">Open</button>` : ''}
        <button class="text-button danger delete-video" data-id="${escapeHtml(video.id)}" type="button" ${busy ? 'disabled title="Available after processing"' : ''}>Delete</button>
      </div>
    </div>`;
  }).join('');
  container.querySelectorAll('.watch-video').forEach(button => button.addEventListener('click', () => openVideo(button.dataset.id, 0)));
  container.querySelectorAll('.delete-video').forEach(button => button.addEventListener('click', () => removeVideo(button.dataset.id)));
}
async function loadDemoSample() {
  const button = $('load-demo');
  const status = $('demo-status');
  button.disabled = true;
  status.textContent = 'Loading sample video…';
  try {
    const asset = await fetch('/static/demo-sample.mp4');
    if (!asset.ok) throw new Error('Sample video is unavailable');
    const file = new File([await asset.blob()], 'momentflow-original-demo.mp4', {type:'video/mp4'});
    const form = new FormData();
    form.append('file', file);
    const result = await request('/api/upload', {method:'POST', body:form});
    status.textContent = result.status === 'already_uploaded' ? 'Sample is already in the library.' : 'Sample queued. Indexing may take a minute.';
    await refreshVideos();
  } catch (error) {
    status.textContent = error.message;
    button.disabled = false;
  }
}
async function refreshVideos() {
  const result = await request('/api/videos');
  videos = result.videos;
  $('count-all').textContent = videos.length;
  $('count-ready').textContent = videos.filter(v => v.status === 'ready').length;
  $('count-processing').textContent = videos.filter(v => ['queued','indexing'].includes(v.status)).length;
  $('library-count').textContent = videos.length;
  renderLibrary();
  clearTimeout(refreshTimer);
  if (videos.some(v => ['queued','indexing'].includes(v.status)))
    refreshTimer = setTimeout(() => refreshVideos().catch(showError), 3000);
}
function showError(error) {
  console.error(error);
  if (!videos.length) {
    $('videos').innerHTML = `<div class="empty-state"><span class="empty-icon" aria-hidden="true">!</span><strong>Could not load the library</strong><p>${escapeHtml(error.message)}</p><button class="primary" id="retry-load" type="button">Retry</button></div>`;
    $('retry-load').addEventListener('click', () => refreshVideos().catch(showError));
  }
}
function renderResults() {
  $('results-count').textContent = currentHits.length;
  $('results').innerHTML = currentHits.length ? currentHits.map((hit, index) => `
    <button class="result-row ${selectedHit === index ? 'active' : ''}" data-index="${index}" type="button">
      ${thumbnail(hit.video_id,'result-thumb')}
      <span class="result-copy"><span class="result-title">${escapeHtml(hit.filename)}</span>
        <span class="result-meta"><span class="source-tag">${escapeHtml(hit.source)}</span><span>${fmtTime(hit.timestamp_sec)}</span><span>· ${hit.sources?.length || 1} signal${(hit.sources?.length || 1) === 1 ? '' : 's'}</span></span>
        <span class="result-evidence">${escapeHtml(hit.evidence)}</span>
      </span>
    </button>`).join('') : `<div class="empty-state"><span class="empty-icon" aria-hidden="true">⌕</span><strong>${currentQuery ? 'No matching moment yet' : 'Ready to search'}</strong><p>${currentQuery ? 'Try a different object, action, spoken phrase, or words on screen.' : 'Enter a query above to search indexed videos by scene, speech, and screen text.'}</p></div>`;
  $('results').querySelectorAll('.result-row').forEach(button => button.addEventListener('click', () => showHit(Number(button.dataset.index))));
}
function showHit(index) {
  const hit = currentHits[index];
  if (!hit) return;
  selectedHit = index;
  renderResults();
  $('viewer').innerHTML = `<video id="moment-player" controls preload="metadata" src="${escapeHtml(hit.video_url)}"></video>
    <div class="viewer-info"><p class="eyebrow">MATCHED MOMENT · ${fmtTime(hit.timestamp_sec)}</p><h2>${escapeHtml(hit.filename)}</h2>
      <p><span class="source-tag">${escapeHtml(hit.source)}</span> ${escapeHtml(hit.evidence)}</p>
      <div class="viewer-actions"><button id="share-button" class="subtle-button" type="button">↗ Copy moment link</button><button id="feedback-button" class="subtle-button" type="button">Flag wrong moment</button></div>
      <p id="viewer-message" class="message" role="status"></p></div>`;
  const player = $('moment-player');
  player.addEventListener('loadedmetadata', () => { player.currentTime = hit.timestamp_sec; }, {once:true});
  $('share-button').addEventListener('click', async () => {
    const link = new URL(location.origin + location.pathname);
    link.searchParams.set('video',hit.video_id);
    link.searchParams.set('t',String(hit.timestamp_sec));
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
function openVideo(id, stamp) {
  const video = videos.find(v => v.id === id);
  if (!video || video.status !== 'ready') return;
  currentQuery = '';
  currentHits = [{video_id:id,filename:video.filename,timestamp_sec:stamp,source:'library',sources:['library'],evidence:'Selected from the video library',video_url:`/api/media/${encodeURIComponent(id)}`}];
  $('result-heading').textContent = video.filename;
  $('search-status').textContent = 'Playing from your library';
  showView('search');
  showHit(0);
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
  $('search-status').textContent = 'Searching the video library…';
  currentHits = [];
  selectedHit = null;
  $('results').innerHTML = '<div class="empty-state"><span class="empty-icon" aria-hidden="true">⌕</span><strong>Finding moments</strong><p>Checking visual, speech, and screen-text signals.</p></div>';
  $('viewer').innerHTML = '<div class="viewer-placeholder"><span class="placeholder-play">▶</span><strong>Finding moments</strong></div>';
  try {
    const data = await request('/api/search?q='+encodeURIComponent(query));
    currentHits = data.results;
    $('search-status').textContent = `${currentHits.length} candidate moment${currentHits.length === 1 ? '' : 's'}`;
    renderResults();
    if (currentHits.length) showHit(0);
    else $('viewer').innerHTML = '<div class="viewer-placeholder"><span class="placeholder-play">⌕</span><strong>No moment selected</strong><p>Try another query to find a matching timestamp.</p></div>';
  } catch(error) {
    $('search-status').textContent = error.message;
    renderResults();
    $('viewer').innerHTML = '<div class="viewer-placeholder"><span class="placeholder-play">!</span><strong>Search unavailable</strong><p>Check the model service and try again.</p></div>';
  }
}
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
$('search-form').addEventListener('submit', event => { event.preventDefault(); const query=$('query').value.trim(); if(query) search(query); });
$('upload-open').addEventListener('click', () => $('upload-dialog').showModal());
$('upload-close').addEventListener('click', () => $('upload-dialog').close());
$('refresh-button').addEventListener('click', () => refreshVideos().catch(showError));
$('filter-input').addEventListener('input', renderLibrary);
$('status-filter').addEventListener('change', renderLibrary);
$('file').addEventListener('change', () => { $('selected-file').textContent = $('file').files[0]?.name || ''; });
$('upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  const file = $('file').files[0];
  if (!file) return;
  $('upload-status').textContent = 'Uploading and validating video…';
  try {
    const form = new FormData();
    form.append('file',file);
    const data = await request('/api/upload',{method:'POST',body:form});
    $('upload-status').textContent = data.status === 'already_uploaded' ? 'This video is already in the library.' : 'Upload complete. Indexing has started.';
    $('upload-form').reset();
    $('selected-file').textContent = '';
    await refreshVideos();
    setTimeout(() => $('upload-dialog').close(),1000);
  } catch(error) { $('upload-status').textContent = error.message; }
});
function followSharedLink() {
  const params = new URLSearchParams(location.search);
  const id = params.get('video'), stamp = Number(params.get('t') || 0);
  if (id) openVideo(id, Number.isFinite(stamp) ? stamp : 0);
}
if (location.protocol === 'file:') {
  $('app-shell').hidden = true;
  $('launch-screen').hidden = false;
} else {
  refreshVideos().then(followSharedLink).catch(showError);
}
