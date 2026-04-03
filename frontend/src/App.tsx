import { useState, useRef, useEffect } from 'react'

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

const STRATEGY_LABELS: Record<Strategy, string> = {
  zero_shot: 'Zero-Shot',
  cot: 'Chain-of-Thought',
  few_shot: 'Few-Shot',
  react: 'ReAct',
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

/** Render answer text, turning timestamp mentions into clickable spans */
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
  const [strategy, setStrategy] = useState<Strategy>('zero_shot')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
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

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const seekVideo = (seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds
      videoRef.current.play()
    }
  }

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
    } catch (err: any) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}` },
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

  const videoUrl = selectedVideo
    ? `${API_BASE}/videos/${videos.find(v => v.video_id === selectedVideo)?.filename || selectedVideo + '.mp4'}`
    : ''

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>CrimeVision-QA</h1>
        <span className="subtitle">Multimodal Surveillance Video Analysis</span>
      </header>

      <div className="main-content">
        {/* Left Panel — Chat */}
        <div className="chat-panel">
          {/* Controls */}
          <div className="controls">
            <div className="control-group">
              <label>Video</label>
              <select value={selectedVideo} onChange={e => setSelectedVideo(e.target.value)}>
                {videos.length === 0 && <option value="">No videos processed</option>}
                {videos.map(v => (
                  <option key={v.video_id} value={v.video_id}>
                    {v.video_id} {v.category ? `(${v.category})` : ''}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label>Strategy</label>
              <select value={strategy} onChange={e => setStrategy(e.target.value as Strategy)}>
                {(Object.keys(STRATEGY_LABELS) as Strategy[]).map(s => (
                  <option key={s} value={s}>{STRATEGY_LABELS[s]}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Messages */}
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <p>Ask a question about the selected surveillance video.</p>
                <p className="examples">
                  Try: "What happened in this video?" · "Describe the people involved" ·
                  "What events occurred between 0:30 and 1:00?"
                </p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <div className="message-content">
                  {msg.role === 'assistant'
                    ? renderAnswerWithTimestamps(msg.content, seekVideo)
                    : msg.content}
                </div>
                {msg.role === 'assistant' && msg.timestamps && msg.timestamps.length > 0 && (
                  <div className="timestamp-chips">
                    {msg.timestamps.slice(0, 8).map((t, j) => (
                      <button key={j} className="chip" onClick={() => seekVideo(t)}>
                        ▶ {formatTimestamp(t)}
                      </button>
                    ))}
                  </div>
                )}
                {msg.role === 'assistant' && msg.processing_time !== undefined && (
                  <div className="meta">
                    {STRATEGY_LABELS[msg.strategy_used as Strategy] || msg.strategy_used}
                    {' · '}{msg.processing_time}s
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="message assistant">
                <div className="typing-indicator">
                  <span /><span /><span />
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
              placeholder="Ask about the video… (Enter to send)"
              rows={2}
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={loading || !input.trim() || !selectedVideo}>
              {loading ? '…' : 'Send'}
            </button>
          </div>
        </div>

        {/* Right Panel — Video Player */}
        <div className="video-panel">
          {selectedVideo ? (
            <>
              <video
                ref={videoRef}
                className="video-player"
                src={videoUrl}
                controls
                key={selectedVideo}
              />
              <div className="video-info">
                <strong>{selectedVideo}</strong>
                {videos.find(v => v.video_id === selectedVideo)?.category && (
                  <span className="category-badge">
                    {videos.find(v => v.video_id === selectedVideo)?.category}
                  </span>
                )}
              </div>
            </>
          ) : (
            <div className="no-video">No video selected</div>
          )}
        </div>
      </div>
    </div>
  )
}
