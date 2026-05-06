import { useState, useRef, useEffect, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'https://crimevision-qa-backend.onrender.com'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface VideoInfo {
  video_id: string
  filename?: string
  video_url?: string
  category?: string
  duration_seconds?: number
  frame_count?: number
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamps?: number[]
  sources?: SourceDoc[]
  strategy_used?: string
  processing_time?: number
}

interface SourceDoc {
  frame_file?: string
  timestamp_seconds?: number
  description?: string
  start_time?: number
  end_time?: number
  text?: string
}

type Strategy = 'zero_shot' | 'cot' | 'few_shot' | 'react'

const STRATEGY_LABELS: Record<Strategy, { label: string; desc: string }> = {
  zero_shot: { label: 'Zero-Shot', desc: 'Direct answer without examples' },
  cot: { label: 'Chain-of-Thought', desc: 'Step-by-step reasoning' },
  few_shot: { label: 'Few-Shot', desc: 'Answer using learned examples' },
  react: { label: 'ReAct', desc: 'Reason + Act iteratively (up to 3x)' },
}

const CATEGORY_COLORS: Record<string, string> = {
  Abuse: '#ef4444', Arrest: '#f97316', Arson: '#fb923c',
  Assault: '#dc2626', Burglary: '#7c3aed', Explosion: '#ea580c',
  Fighting: '#b91c1c', RoadAccidents: '#ca8a04', Robbery: '#9333ea',
  Shooting: '#be123c', Shoplifting: '#0284c7', Stealing: '#0369a1',
  Stealing2: '#0891b2', Vandalism: '#16a34a', NormalVideos: '#22c55e',
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function getCategoryColor(category?: string): string {
  if (!category) return '#6366f1'
  return CATEGORY_COLORS[category] || '#6366f1'
}

function renderAnswerWithTimestamps(
  text: string,
  onSeek: (t: number) => void
): React.ReactNode {
  const parts = text.split(/(\d+\.?\d*s|\d+:\d{2})/g)
  return parts.map((part, i) => {
    const mmss = part.match(/^(\d+):(\d{2})$/)
    const secs = part.match(/^(\d+\.?\d*)s$/)
    if (mmss) {
      const t = parseInt(mmss[1]) * 60 + parseInt(mmss[2])
      return (
        <button key={i} className="timestamp-link" onClick={() => onSeek(t)}>
          {part}
        </button>
      )
    }
    if (secs) {
      return (
        <button key={i} className="timestamp-link" onClick={() => onSeek(parseFloat(secs[1]))}>
          {part}
        </button>
      )
    }
    return <span key={i}>{part}</span>
  })
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export default function App() {
  const [videos, setVideos] = useState<VideoInfo[]>([])
  const [selectedVideo, setSelectedVideo] = useState<string>('')
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [strategy, setStrategy] = useState<Strategy>('zero_shot')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [uploadJobId, setUploadJobId] = useState<string | null>(null)
  const [uploadStatus, setUploadStatus] = useState<{
    status: string; progress: number; message: string; video_id: string; error?: string
  } | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Fetch video list on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/videos`)
      .then(r => r.json())
      .then((data: VideoInfo[]) => {
        setVideos(data)
        if (data.length > 0) setSelectedVideo(data[0].video_id)
      })
      .catch(err => console.error('Failed to load videos:', err))
  }, [])

  // Update video URL when selected video changes
  useEffect(() => {
    const v = videos.find(v => v.video_id === selectedVideo)
    setVideoUrl(v?.video_url ? `${API_BASE}${v.video_url}` : null)
  }, [selectedVideo, videos])

  // Reset messages when video changes
  useEffect(() => {
    setMessages([])
  }, [selectedVideo])

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Upload handler
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''  // reset so the same file can be re-selected

    const form = new FormData()
    form.append('file', file)
    form.append('category', 'Unknown')

    try {
      const resp = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || 'Upload failed')
      setUploadJobId(data.job_id)
      setUploadStatus({ status: 'queued', progress: 0, message: 'Queued…', video_id: data.video_id })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed'
      setUploadStatus({ status: 'error', progress: 0, message: msg, video_id: '' })
    }
  }

  // Poll upload job status
  useEffect(() => {
    if (!uploadJobId) return
    const id = setInterval(async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/upload/status/${uploadJobId}`)
        const data = await resp.json()
        setUploadStatus(data)
        if (data.status === 'done') {
          clearInterval(id)
          setUploadJobId(null)
          fetch(`${API_BASE}/api/videos`)
            .then(r => r.json())
            .then((vids: VideoInfo[]) => {
              setVideos(vids)
              setSelectedVideo(data.video_id)
            })
        }
        if (data.status === 'error') clearInterval(id)
      } catch { /* network hiccup — keep polling */ }
    }, 3000)
    return () => clearInterval(id)
  }, [uploadJobId])

  const seekToTimestamp = useCallback((seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds
      videoRef.current.play()
    }
  }, [])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || !selectedVideo || loading) return

    const userMsg: Message = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const resp = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, video_id: selectedVideo, strategy }),
      })

      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || 'Server error')
      }

      const data = await resp.json()
      const assistantMsg: Message = {
        role: 'assistant',
        content: data.answer,
        timestamps: data.timestamps,
        sources: data.sources,
        strategy_used: data.strategy_used,
        processing_time: data.processing_time,
      }
      setMessages(prev => [...prev, assistantMsg])

      // Auto-seek to first timestamp
      if (data.timestamps?.length > 0) {
        seekToTimestamp(data.timestamps[0])
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `⚠️ Error: ${msg}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const selectedInfo = videos.find(v => v.video_id === selectedVideo)
  const catColor = getCategoryColor(selectedInfo?.category)

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(o => !o)}>
            {sidebarOpen ? '⊟' : '⊞'}
          </button>
          <div className="logo">
            <span className="logo-icon">👁</span>
            <span className="logo-text">CrimeVision<span className="logo-qa">QA</span></span>
          </div>
          <span className="tagline">Multimodal Surveillance Video Analysis · DATA 266</span>
        </div>
        <div className="header-right">
          {selectedInfo && (
            <div className="current-video-badge" style={{ borderColor: catColor }}>
              <span className="cat-dot" style={{ background: catColor }} />
              <span>{selectedInfo.category}</span>
              <span className="sep">·</span>
              <span className="vid-id">{selectedVideo}</span>
              {selectedInfo.frame_count && (
                <span className="frame-count">{selectedInfo.frame_count} frames</span>
              )}
            </div>
          )}
        </div>
      </header>

      <div className="workspace">
        {/* Sidebar — video list */}
        <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
          <div className="sidebar-header">
            <span>Processed Videos</span>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <span className="count-badge">{videos.length}</span>
              <button
                className="upload-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={!!uploadJobId}
                title="Upload a video to ingest"
              >
                {uploadJobId ? '⏳' : '+ Upload'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                style={{ display: 'none' }}
                onChange={handleUpload}
              />
            </div>
          </div>

          {uploadStatus && uploadStatus.status !== 'done' && (
            <div className="upload-progress">
              <div className="upload-progress-label">
                <span className="upload-vid-name">{uploadStatus.video_id || 'Uploading…'}</span>
                <span>{uploadStatus.progress}%</span>
              </div>
              <div className="upload-progress-bar">
                <div className="upload-progress-fill" style={{ width: `${uploadStatus.progress}%` }} />
              </div>
              <div className="upload-progress-msg">{uploadStatus.message}</div>
              {uploadStatus.status === 'error' && (
                <div className="upload-error">{uploadStatus.error || 'Pipeline failed — check backend logs'}</div>
              )}
            </div>
          )}

          <div className="video-list">
            {videos.length === 0 ? (
              <div className="empty-list">
                <p>No videos ingested yet.</p>
                <code>python scripts/ingest_dataset.py</code>
              </div>
            ) : (
              videos.map(v => (
                <button
                  key={v.video_id}
                  className={`video-item ${v.video_id === selectedVideo ? 'active' : ''}`}
                  onClick={() => setSelectedVideo(v.video_id)}
                  style={v.video_id === selectedVideo ? { borderLeftColor: getCategoryColor(v.category) } : {}}
                >
                  <span
                    className="cat-pill"
                    style={{ background: getCategoryColor(v.category) + '33', color: getCategoryColor(v.category) }}
                  >
                    {v.category || 'Unknown'}
                  </span>
                  <span className="vid-name">{v.video_id}</span>
                  {v.frame_count && <span className="frame-ct">{v.frame_count}f</span>}
                </button>
              ))
            )}
          </div>
        </aside>

        {/* Main — video player + chat */}
        <main className="main-area">
          {/* Video Player */}
          <section className="video-section">
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                className="video-player"
              />
            ) : (
              <div className="video-placeholder">
                {selectedVideo ? 'Video file not found in videos/ folder' : 'Select a video to begin'}
              </div>
            )}
          </section>

          {/* Chat Panel */}
          <section className="chat-section">
            {/* Strategy selector */}
            <div className="strategy-bar">
              {(Object.keys(STRATEGY_LABELS) as Strategy[]).map(s => (
                <button
                  key={s}
                  className={`strategy-btn ${strategy === s ? 'active' : ''}`}
                  onClick={() => setStrategy(s)}
                  title={STRATEGY_LABELS[s].desc}
                >
                  {STRATEGY_LABELS[s].label}
                </button>
              ))}
            </div>

            {/* Messages */}
            <div className="messages">
              {messages.length === 0 && (
                <div className="empty-chat">
                  <div className="empty-icon">💬</div>
                  <p>Ask a question about <strong>{selectedVideo || 'the selected video'}</strong></p>
                  <div className="suggestions">
                    {[
                      'What happened in this video?',
                      'Describe the people involved',
                      'What events occurred at the start?',
                      'How many people are visible?',
                    ].map(s => (
                      <button
                        key={s}
                        className="suggestion"
                        onClick={() => { setInput(s); }}
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="msg-avatar">
                    {msg.role === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className="msg-body">
                    <div className="msg-content">
                      {msg.role === 'assistant'
                        ? renderAnswerWithTimestamps(msg.content, seekToTimestamp)
                        : msg.content}
                    </div>

                    {/* Timestamp chips */}
                    {msg.role === 'assistant' && msg.timestamps && msg.timestamps.length > 0 && (
                      <div className="timestamp-chips">
                        {msg.timestamps.slice(0, 6).map((t, j) => (
                          <button key={j} className="chip" onClick={() => seekToTimestamp(t)}>
                            ▶ {formatTimestamp(t)}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Sources */}
                    {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                      <details className="sources">
                        <summary>{msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}</summary>
                        <div className="sources-list">
                          {msg.sources.slice(0, 4).map((s, j) => (
                            <div key={j} className="source-item">
                              {s.timestamp_seconds !== undefined && (
                                <button
                                  className="source-ts"
                                  onClick={() => seekToTimestamp(s.timestamp_seconds!)}
                                >
                                  ⏱ {formatTimestamp(s.timestamp_seconds)}
                                </button>
                              )}
                              <p className="source-desc">{s.description || s.text}</p>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {/* Meta */}
                    {msg.role === 'assistant' && msg.processing_time !== undefined && (
                      <div className="msg-meta">
                        <span>{STRATEGY_LABELS[msg.strategy_used as Strategy]?.label || msg.strategy_used}</span>
                        <span>·</span>
                        <span>{msg.processing_time}s</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div className="message assistant">
                  <div className="msg-avatar">🤖</div>
                  <div className="msg-body">
                    <div className="typing-indicator">
                      <span /><span /><span />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="input-row">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={`Ask about ${selectedVideo || 'the video'} (Enter to send, Shift+Enter for newline)`}
                rows={2}
                disabled={loading || !selectedVideo}
              />
              <button
                id="send-btn"
                className="send-btn"
                onClick={sendMessage}
                disabled={loading || !input.trim() || !selectedVideo}
              >
                {loading ? (
                  <span className="spin">⟳</span>
                ) : (
                  '→'
                )}
              </button>
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}
