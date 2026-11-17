// MomentFlow - Apple Minimal UI Controller
(function() {
  const $ = id => document.getElementById(id);

  // State
  let selectedVideoFile = null;
  let currentVideoUrl = null;
  let activeMoments = [];
  let currentMomentIndex = 0;

  // DOM Elements
  const videoInput = $('video-input');
  const dropZone = $('drop-zone');
  const uploadPrompt = $('upload-prompt');
  const fileChip = $('file-chip');
  const chipFilename = $('chip-filename');
  const chipDetails = $('chip-details');
  const removeFileBtn = $('remove-file-btn');
  const searchForm = $('search-form');
  const queryInput = $('query-input');
  const searchBtn = $('search-btn');
  const loadingState = $('loading-state');
  const resultSection = $('result-section');
  const videoPlayer = $('video-player');
  const momentTimestamp = $('moment-timestamp');
  const momentExplanation = $('moment-explanation');
  const replayBtn = $('replay-btn');
  const otherMoments = $('other-moments');
  const momentsList = $('moments-list');
  const statusText = $('status-text');

  // Format seconds to mm:ss
  function formatTime(sec) {
    const s = Math.max(0, Math.floor(Number(sec) || 0));
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m}:${String(rem).padStart(2, '0')}`;
  }

  // Format file size
  function formatSize(bytes) {
    if (!bytes) return '';
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  }

  // Handle File Selection
  function selectFile(file) {
    if (!file) return;
    selectedVideoFile = file;

    if (currentVideoUrl) {
      URL.revokeObjectURL(currentVideoUrl);
    }
    currentVideoUrl = URL.createObjectURL(file);

    chipFilename.textContent = file.name;
    chipDetails.textContent = formatSize(file.size);
    uploadPrompt.hidden = true;
    fileChip.hidden = false;

    // Preload metadata to get duration
    videoPlayer.src = currentVideoUrl;
    videoPlayer.load();

    queryInput.focus();
  }

  // Clear Selected File
  function clearFile() {
    selectedVideoFile = null;
    if (currentVideoUrl) {
      URL.revokeObjectURL(currentVideoUrl);
      currentVideoUrl = null;
    }
    videoInput.value = '';
    uploadPrompt.hidden = false;
    fileChip.hidden = true;
    resultSection.hidden = true;
  }

  // Drag and Drop Listeners
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      selectFile(files[0]);
    }
  });

  videoInput.addEventListener('change', () => {
    if (videoInput.files.length > 0) {
      selectFile(videoInput.files[0]);
    }
  });

  removeFileBtn.addEventListener('click', e => {
    e.stopPropagation();
    clearFile();
  });

  // Display a specific moment
  function displayMoment(index) {
    if (!activeMoments[index]) return;
    currentMomentIndex = index;
    const moment = activeMoments[index];

    const startSec = Math.max(0, moment.start || 0);
    const endSec = moment.end || (startSec + 4);
    const duration = Math.max(1, Math.round(endSec - startSec));

    momentTimestamp.textContent = `${formatTime(startSec)} – ${formatTime(endSec)} (${duration}s)`;

    // Clean human-friendly description
    let cleanText = moment.explanation || 'Action detected in this scene.';
    // Remove technical artifacts if any
    cleanText = cleanText.replace(/^(Answer:|Scene:|Observation:)\s*/i, '');
    momentExplanation.textContent = cleanText;

    // Jump player and play
    videoPlayer.currentTime = startSec;
    videoPlayer.play().catch(() => {});

    // Highlight active pill if multiple
    const pills = momentsList.querySelectorAll('.moment-pill-btn');
    pills.forEach((p, idx) => p.classList.toggle('active', idx === index));
  }

  // Replay Button
  replayBtn.addEventListener('click', () => {
    if (activeMoments[currentMomentIndex]) {
      videoPlayer.currentTime = activeMoments[currentMomentIndex].start || 0;
      videoPlayer.play().catch(() => {});
    }
  });

  // Search Submission
  searchForm.addEventListener('submit', async e => {
    e.preventDefault();
    const query = queryInput.value.trim();

    if (!selectedVideoFile) {
      alert('Please choose a video first.');
      return;
    }
    if (!query) {
      queryInput.focus();
      return;
    }

    // UI Loading state
    searchBtn.disabled = true;
    loadingState.hidden = false;
    resultSection.hidden = true;

    const form = new FormData();
    form.append('file', selectedVideoFile);
    form.append('query', query);
    form.append('sample_fps', '1.0');
    form.append('top_k', '3');

    try {
      let res = await fetch('/extract_moments', { method: 'POST', body: form });
      if (!res.ok && res.status === 404) {
        res = await fetch('/api/extract_moments', { method: 'POST', body: form });
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Could not find the moment in this video.');
      }

      const data = await res.json();
      activeMoments = data.moments || [];

      if (!activeMoments.length) {
        alert(`No moment found matching "${query}". Try describing it with different words.`);
        return;
      }

      // Populate other moments list if > 1
      momentsList.innerHTML = '';
      if (activeMoments.length > 1) {
        otherMoments.hidden = false;
        activeMoments.forEach((m, idx) => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = `moment-pill-btn ${idx === 0 ? 'active' : ''}`;
          btn.textContent = `Scene ${idx + 1} (${formatTime(m.start)})`;
          btn.addEventListener('click', () => displayMoment(idx));
          momentsList.appendChild(btn);
        });
      } else {
        otherMoments.hidden = true;
      }

      // Show result and play top moment
      resultSection.hidden = false;
      displayMoment(0);

      // Smooth scroll to result
      resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
      alert(err.message || 'Error processing video.');
    } finally {
      searchBtn.disabled = false;
      loadingState.hidden = true;
    }
  });

  // Background health check
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        if (data.kaggle_gpu_status === 'connected') {
          statusText.textContent = 'AI Ready';
        } else if (data.status === 'healthy') {
          statusText.textContent = 'Ready';
        }
      }
    } catch {
      statusText.textContent = 'Ready';
    }
  }

  checkHealth();
})();
