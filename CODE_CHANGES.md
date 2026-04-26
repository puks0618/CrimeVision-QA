# Code Changes Summary — Line-by-Line Reference

## File: `llm/gen_frame_desc.py`

### Change 1: Vision Prompt (Lines 20-56)

**OLD (Generic):**
```python
_PROMPT = (
    "Describe this surveillance video frame for law enforcement analysis. "
    "Include: people (appearance, clothing, actions, positions), vehicles "
    "(type, color, partial plates if visible), objects, setting, lighting "
    "conditions, and any visible text or signage. Be concise and factual."
)
```

**NEW (Detailed Surveillance-Specific):**
```python
_PROMPT = (
    "Analyze this surveillance frame for criminal activity detection with MAXIMUM detail. "
    "MANDATORY - Report with precision:\n\n"
    "PEOPLE (if visible):\n"
    "- Count, approximate age range, skin tone, ethnicity\n"
    "- Exact clothing: colors, types, visible logos/text, accessories\n"
    "- PRECISE ACTIONS: running/walking/standing/fighting/stealing/concealing/pointing/attacking/fleeing\n"
    "- Hand positions: empty, carrying items, using weapons, raised, in pockets\n"
    "- Visible face details: mask/hoodie obscuring face, visible features, jewelry\n"
    "- Any visible weapons, tools, or stolen items\n"
    "- Injuries, blood, or suspicious marks\n"
    "- Direction of movement with compass direction if possible\n\n"
    "VEHICLES (if visible):\n"
    "- Make, model, color, condition\n"
    "- License plate: any visible digits/characters\n"
    "- Windows tinted/normal, occupants visible\n"
    "- Damage, modifications, unique features\n"
    "- Direction/position relative to scene\n\n"
    "OBJECTS/PROPERTY:\n"
    "- Items on ground, in hands, or being transported\n"
    "- Signs of theft: open doors, broken windows, scattered items\n"
    "- Weapons visible: guns, knives, bats, explosives\n"
    "- Packages, boxes, bags - color and contents if identifiable\n\n"
    "SETTING:\n"
    "- Location type: street/parking lot/store/home/warehouse/alley\n"
    "- Entry/exit points visible\n"
    "- Time of day indicators: sunlight/darkness/street lights\n"
    "- Weather: dry/wet/snowing\n"
    "- Surrounding buildings, signs, landmarks\n\n"
    "CRITICAL OBSERVATIONS:\n"
    "- Is there suspicious behavior? Be specific.\n"
    "- Are multiple people coordinating movements?\n"
    "- Anyone acting as lookout or sentinel?\n"
    "- Signs of organized crime vs opportunistic theft?\n"
    "- Any violence, weapons, or imminent danger indicators?\n\n"
    "IMPORTANT: Be EXTREMELY SPECIFIC and FACTUAL. Avoid generic descriptions like "
    "'appears to show' or 'seems to be'. If you cannot clearly see a detail, write "
    "'[NOT VISIBLE]' instead of guessing. Prioritize details that distinguish this "
    "frame from normal activity."
)
```

### Change 2: API Parameters (Lines 80-85)

**OLD:**
```python
"max_tokens": 300,
"temperature": 0.2,
```

**NEW:**
```python
"max_tokens": 1000,
"temperature": 0.1,
```

### Change 3: Add Quality Validation Function (Lines 139-178 - NEW)

```python
def _validate_description_quality(description: str) -> bool:
    """Check if description is sufficiently detailed (not generic/hallucinated).
    
    Returns True if quality is acceptable, False if suspiciously generic.
    """
    # Reject very short descriptions (likely hallucinated)
    if len(description) < 100:
        return False
    
    # Reject descriptions with excessive uncertainty markers
    if description.count("[NOT VISIBLE]") > 5:
        return False
    
    # Check for minimum specificity — must mention at least some concrete details
    specific_keywords = [
        "color", "clothing", "person", "vehicle", "action", 
        "standing", "running", "walking", "holding", "wearing",
        "number", "street", "store", "parking", "building"
    ]
    
    description_lower = description.lower()
    keyword_count = sum(1 for kw in specific_keywords if kw in description_lower)
    
    # Must have at least 4 specific keywords to avoid generic descriptions
    if keyword_count < 4:
        return False
    
    # Reject descriptions that sound generic
    generic_phrases = [
        "appears to show a scene",
        "shows some people",
        "could be",
        "might be",
        "possibly",
        "unclear what",
        "hard to make out",
    ]
    
    if any(phrase in description_lower for phrase in generic_phrases):
        return False
    
    return True
```

---

## File: `llm/process_frames.py`

### Change 1: Import Quality Validation (Line 25)

**OLD:**
```python
from llm.gen_frame_desc import describe_frame
```

**NEW:**
```python
from llm.gen_frame_desc import describe_frame, _validate_description_quality
```

### Change 2: Add Quality Check During Ingestion (Lines 138-139)

**OLD:**
```python
if desc in ("[DESCRIPTION UNAVAILABLE]", "[INVALID FRAME]"):
    errors += 1
    continue  # Don't store frames with no description
if emb == _ZERO_VECTOR:
    errors += 1
    continue  # Don't pollute index with zero-vector frames

doc = {
```

**NEW:**
```python
if desc in ("[DESCRIPTION UNAVAILABLE]", "[INVALID FRAME]"):
    errors += 1
    continue  # Don't store frames with no description
if emb == _ZERO_VECTOR:
    errors += 1
    continue  # Don't pollute index with zero-vector frames

# Warn about low-quality descriptions but don't skip them
if not _validate_description_quality(desc):
    print(f"⚠️  Low-quality description for {m['frame_file']}: {desc[:80]}...")

doc = {
```

---

## New Files Created

### File: `cleanup_mongodb.py`

**Purpose:** Remove zero-vectors and duplicates  
**Key Functions:**
- `cleanup_zero_vectors()` - Removes all [0.0]*1024 embeddings
- `cleanup_duplicates()` - Keeps most recent frame per (video_id, frame_file)
- `cleanup_transcript_duplicates()` - Keeps most recent transcript per (video_id, segment_index)
- `get_collection_stats()` - Prints before/after stats

**Usage:**
```bash
python cleanup_mongodb.py
```

### File: `debug_embeddings.py`

**Purpose:** Diagnostic tool for embedding quality  
**Stages:**
1. Vision Model Accuracy - checks if descriptions are specific
2. Embedding Dimension - validates 1024-dim vectors
3. MongoDB Indexes - verifies vs_frames_index exists and is ACTIVE
4. Vector Search Quality - tests retrieval with sample query
5. Description Quality - flags suspicious patterns

**Usage:**
```bash
python debug_embeddings.py --video-id VideoName --query "test query" --output report.json
```

### File: `EMBEDDING_FIX_GUIDE.md`

Complete guide including:
- Problem summary
- Root causes (priority order)
- Step-by-step fixes
- Validation checklist
- Troubleshooting

### File: `EXECUTION_GUIDE.md`

Quick execution reference:
- 4-step execution process
- Expected outputs
- Expected improvements table
- Troubleshooting quick fixes

### File: `IMPLEMENTATION_SUMMARY.md`

Overview document with:
- Files modified
- What each change fixes
- Implementation checklist
- Expected results
- Technical details
- Rollback instructions

---

## Summary of Changes

| Component | Lines | Change Type | Impact |
|-----------|-------|-------------|--------|
| Vision Prompt | 20-56 | Replaced | High - Much more detailed |
| max_tokens | 85 | 300 → 1000 | High - Allows detailed output |
| temperature | 85 | 0.2 → 0.1 | Medium - More precision |
| Quality Validation | +139-178 | New function | Medium - Flags hallucinations |
| Process Pipeline | 139-140 | Warning added | Low - Informational |
| Imports | 25 | Updated | Low - Enables quality check |

---

## Testing the Changes

### Quick Test (5 minutes)
```bash
# Test vision model with one frame
python -c "
from llm.gen_frame_desc import describe_frame
desc = describe_frame('frames/VideoName/frame_0001.png')
print('Description length:', len(desc))
print('First 200 chars:', desc[:200])
print('Is quality acceptable:', len(desc) > 100 and desc.count('[NOT VISIBLE]') < 5)
"
```

### Full Test (30 minutes)
```bash
# 1. Clean data
python cleanup_mongodb.py

# 2. Ingest one video
python llm/process_frames.py --frames-dir frames/VideoName/ --video-id VideoName --category Category

# 3. Check quality
python debug_embeddings.py --video-id VideoName

# 4. Test retrieval
python -c "
from llm.inference import semantic_search_frames
results = semantic_search_frames('person running', video_id='VideoName', k=5)
for r in results:
    print(f\"Score: {r['score']:.4f} - {r['description'][:80]}\")
"
```

---

## Verification Checklist

- [ ] `gen_frame_desc.py` has new detailed _PROMPT
- [ ] max_tokens increased to 1000
- [ ] temperature reduced to 0.1
- [ ] `_validate_description_quality()` function exists
- [ ] `process_frames.py` imports quality validation
- [ ] Process pipeline has quality warning (line 139)
- [ ] `cleanup_mongodb.py` exists and has cleanup functions
- [ ] `debug_embeddings.py` exists with 5 diagnostic stages
- [ ] All documentation files created and readable
- [ ] Execution steps clear and documented

