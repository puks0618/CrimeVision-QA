import { useState, useRef, useEffect, useCallback } from 'react'

const API_BASE = 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface VideoInfo {
  video_id: string
  filename?: string
  category?: string
  duration_seconds?: number
  frame_count?: number
}

interface FrameInfo {
  frame_file: string
  frame_number: number
  timestamp_seconds: number
  description: string
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
// FrameViewer Component
// ---------------------------------------------------------------------------
function FrameViewer({
  videoId,
  frames,
  activeTimestamp,
}: {
  videoId: string
  frames: FrameInfo[]
  activeTimestamp?: number
}) {
  const [activeIdx, setActiveIdx] = useState(0)
  const stripRef = useRef<HTMLDivElement>(null)

  // Jump to closest frame when a timestamp is seeked
  useEffect(() => {
    if (activeTimestamp === undefined || frames.length === 0) return
    let best = 0
    let bestDiff = Infinity
    frames.forEach((f, i) => {
      const diff = Math.abs(f.timestamp_seconds - activeTimestamp)
      if (diff < bestDiff) { bestDiff = diff; best = i }
    })
    setActiveIdx(best)
    // Scroll the thumbnail strip
    const el = stripRef.current?.children[best] as HTMLElement
    el?.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
  }, [activeTimestamp, frames])

  if (frames.length === 0) {
    return (
      <div className="frame-viewer empty">
        <div className="frame-placeholder">
          <span className="frame-icon">🎬</span>
          <p>No frames loaded yet</p>
          <p className="hint">Select a video to browse its frames</p>
        </div>
      </div>
    )
  }

  const active = frames[activeIdx]
  const frameUrl = `${API_BASE}/frames/${videoId}/${active.frame_file}`

  return (
    <div className="frame-viewer">
      {/* Main frame */}
      <div className="frame-main">
        <img
          src={frameUrl}
          alt={`Frame at ${formatTimestamp(active.timestamp_seconds)}`}
          className="frame-image"
          onError={(e) => { (e.target as HTMLImageElement).src = '' }}
        />
        <div className="frame-overlay">
          <span className="frame-ts">⏱ {formatTimestamp(active.timestamp_seconds)}</span>
          <span className="frame-num">Frame #{active.frame_number}</span>
        </div>
        {/* Navigation arrows */}
        <button
          className="nav-arrow left"
          onClick={() => setActiveIdx(i => Math.max(0, i - 1))}
          disabled={activeIdx === 0}
        >‹</button>
        <button
          className="nav-arrow right"
          onClick={() => setActiveIdx(i => Math.min(frames.length - 1, i + 1))}
          disabled={activeIdx === frames.length - 1}
        >›</button>
      </div>

      {/* Description */}
      {active.description && (
        <div className="frame-desc">
          <span className="desc-icon">🔍</span>
          <p>{active.description}</p>
        </div>
      )}

      {/* Thumbnail strip */}
      <div className="frame-strip" ref={stripRef}>
        {frames.map((f, i) => (
          <button
            key={f.frame_file}
            className={`thumb ${i === activeIdx ? 'active' : ''}`}
            onClick={() => setActiveIdx(i)}
            title={`${formatTimestamp(f.timestamp_seconds)}`}
          >
            <img
              src={`${API_BASE}/frames/${videoId}/${f.frame_file}`}
              alt=""
              loading="lazy"
            />
            <span>{formatTimestamp(f.timestamp_seconds)}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export default function App() {
  const [videos, setVideos] = useState<VideoInfo[]>([])
  const [selectedVideo, setSelectedVideo] = useState<string>('')
  const [frames, setFrames] = useState<FrameInfo[]>([])
  const [strategy, setStrategy] = useState<Strategy>('zero_shot')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [seekTs, setSeekTs] = useState<number | undefined>()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

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

  // Fetch frames when video changes
  useEffect(() => {
    if (!selectedVideo) return
    setFrames([])
    fetch(`${API_BASE}/api/videos/${selectedVideo}/frames`)
      .then(r => r.ok ? r.json() : [])
      .then((data: FrameInfo[]) => setFrames(data))
      .catch(() => setFrames([]))
  }, [selectedVideo])

  // Reset messages when video changes
  useEffect(() => {
    setMessages([])
  }, [selectedVideo])

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const seekToTimestamp = useCallback((seconds: number) => {
    setSeekTs(seconds)
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
        setSeekTs(data.timestamps[0])
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
            <span className="count-badge">{videos.length}</span>
          </div>
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

        {/* Main — chat + frame viewer */}
        <main className="main-area">
          {/* Frame Viewer */}
          <section className="frame-section">
            <FrameViewer
              videoId={selectedVideo}
              frames={frames}
              activeTimestamp={seekTs}
            />
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
