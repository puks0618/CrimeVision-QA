# 🚀 CrimeVision Embedding Fix — Execution Checklist

**Start Date**: ________________  
**Completion Date**: ________________  

---

## PRE-EXECUTION VERIFICATION

### Environment Setup
- [ ] MongoDB connection string in `.env` (MONGODB_URI)
- [ ] Fireworks API key in `.env` (FIREWORKS_API_KEY)
- [ ] Voyage API key in `.env` (VOYAGE_API_KEY)
- [ ] Python environment activated (conda/venv)
- [ ] Required packages installed (`pip install -r requirements.txt`)

### Database Preparation
- [ ] Connected to MongoDB Atlas
- [ ] Database `video_intelligence` exists
- [ ] Collection `video_intelligence` exists
- [ ] Collection `video_intelligence_transcripts` exists
- [ ] Vector index `vs_frames_index` exists and is ACTIVE
- [ ] Vector index `vs_transcripts_index` exists and is ACTIVE

### Video Preparation
- [ ] Video frames extracted to `frames/VideoName/`
- [ ] Frames are JPG or PNG format
- [ ] At least 10 frames in directory (for testing)
- [ ] Frame naming convention verified (frame_XXXX_tYYs.jpg or VideoName_XXXX.png)

---

## STEP 1: Clean Corrupted Data

**Objective**: Remove zero-vectors and duplicates from previous ingestions

```bash
python cleanup_mongodb.py
```

### Before Cleanup Stats
- [ ] Record frames count: __________
- [ ] Record transcripts count: __________

### Cleanup Output Verification
```
CLEANING ZERO-VECTORS FROM FRAMES
✅ Deleted X zero-vector frame documents
    Expected: X should be 0 (already fixed) or small number

CLEANING ZERO-VECTORS FROM TRANSCRIPTS  
✅ Deleted X zero-vector transcript documents
    Expected: X should be 0 (already fixed) or small number

CLEANING DUPLICATES FROM FRAMES
✅ Total duplicate documents deleted: X
    Expected: X should be 0 (already fixed) or small number
```

### After Cleanup Stats
- [ ] Record frames count: __________
- [ ] Record transcripts count: __________
- [ ] Verify counts are reasonable (not zero unless starting fresh)

### ✅ Step 1 Complete When:
- [ ] Cleanup script ran successfully
- [ ] No errors in output
- [ ] Statistics show reasonable numbers

---

## STEP 2: Re-Ingest Video with Improved Prompt

**Objective**: Generate detailed descriptions with new prompt

```bash
VIDEO_ID="Abuse001_x264"  # Change this
FRAMES_DIR="frames/Abuse001_x264/"  # Change this
CATEGORY="Abuse"  # Change this

python llm/process_frames.py \
  --frames-dir "$FRAMES_DIR" \
  --video-id "$VIDEO_ID" \
  --category "$CATEGORY"
```

### Execution Parameters
- [ ] VIDEO_ID: __________
- [ ] FRAMES_DIR: __________
- [ ] CATEGORY: __________

### Expected Output Verification
```
[Process] Processing NNN frames for video 'VideoName'
[Process] Describing batch 1... [████] 10 frames
[Vision] ...descriptions being generated...
⚠️ Low-quality description for frame_XXXX (if any)
✅ Stored NNN frames with embeddings
```

### Quality Warnings
- [ ] Count ⚠️ warnings: __________ (should be < 5% of total)
- [ ] Review flagged descriptions (if any)
- [ ] Decide: Accept or re-run with different model?

### Post-Ingestion Stats
```bash
python -c "
from llm.config import frames_col
count = frames_col.count_documents({'video_id': 'VideoID'})
print(f'Frames ingested: {count}')

doc = frames_col.find_one({'video_id': 'VideoID'})
if doc:
    print(f'Description length: {len(doc[\"description\"])}')
    print(f'Embedding dim: {len(doc[\"embedding\"])}')
    print(f'Sample: {doc[\"description\"][:100]}...')
"
```

- [ ] Frames ingested: __________
- [ ] Description length: __________ (should be 300+)
- [ ] Embedding dimension: __________ (should be 1024)

### ✅ Step 2 Complete When:
- [ ] All frames processed without errors
- [ ] No more than 5% low-quality warnings
- [ ] Sample descriptions are detailed (300+ chars)
- [ ] Embedding dimension is 1024

---

## STEP 3: Verify Quality with Diagnostic Tool

**Objective**: Verify descriptions and embeddings are correct

```bash
python debug_embeddings.py \
  --video-id "$VIDEO_ID" \
  --query "person running" \
  --output diagnostic_report.json
```

### Stage 1: Vision Model Accuracy
- [ ] Found frame documents: ✅
- [ ] Sample descriptions are detailed: ✅
- [ ] No hallucinated descriptions: ✅

### Stage 2: Embedding Dimension
- [ ] Actual dimension: __________
- [ ] Expected dimension: __________ (should be 1024)
- [ ] Dimensions match: ✅

### Stage 3: MongoDB Vector Indexes
- [ ] vs_frames_index found: ✅
- [ ] vs_frames_index status: __________ (should be ACTIVE)
- [ ] vs_transcripts_index found: ✅
- [ ] vs_transcripts_index status: __________ (should be ACTIVE)

### Stage 4: Vector Search Quality
- [ ] Search returned results: ✅
- [ ] Number of results: __________
- [ ] Top score: __________ (should be > 0.7)
- [ ] Results are relevant: ✅

### Stage 5: Description Quality
- [ ] Manual inspection completed: ⚪ (Not automated, optional)
- [ ] Descriptions match actual video content: ✅

### Diagnostic Report JSON
- [ ] Report saved to: diagnostic_report.json
- [ ] Review JSON for any "status": "error"

### ✅ Step 3 Complete When:
- [ ] All 5 stages pass verification
- [ ] Vector search returns relevant results
- [ ] No errors in diagnostic report

---

## STEP 4: Test Retrieval Accuracy (Optional)

**Objective**: Evaluate retrieval quality with test queries

```bash
python evaluation/eval_retrieval.py \
  --video-id "$VIDEO_ID" \
  --output eval_results.json
```

### Evaluation Metrics
- [ ] Total queries executed: __________
- [ ] Top-5 accuracy: _________% (aim for 70%+)
- [ ] Mean reciprocal rank: __________ (aim for 0.7+)
- [ ] Precision@1: _________% (aim for 80%+)

### Error Analysis
- [ ] False positives: __________% (aim for < 10%)
- [ ] No results returned: __________% (aim for < 5%)
- [ ] Duplicate results: __________% (aim for 0%)

### Sample Results Review
- [ ] Top-3 results for "person running": Relevant? ✅/❌
- [ ] Top-3 results for "vehicle": Relevant? ✅/❌
- [ ] Top-3 results for "weapon": Relevant? ✅/❌

### Comparison with Before Fix
- [ ] Before accuracy: __________%
- [ ] After accuracy: __________%
- [ ] Improvement: +_________% ✅

### ✅ Step 4 Complete When:
- [ ] Evaluation ran without errors
- [ ] Top-5 accuracy > 70%
- [ ] Improvement is measurable

---

## POST-EXECUTION VERIFICATION

### Data Integrity Check
```bash
python -c "
from llm.config import frames_col
# Check for zero vectors
zero = frames_col.count_documents({'embedding': [0.0]*1024})
print(f'Zero vectors: {zero}')  # Should be 0

# Check embedding dimension
doc = frames_col.find_one()
dim = len(doc['embedding']) if doc else 0
print(f'Embedding dim: {dim}')  # Should be 1024

# Check for valid descriptions
no_desc = frames_col.count_documents({'description': '[DESCRIPTION UNAVAILABLE]'})
print(f'Missing descriptions: {no_desc}')  # Should be 0
"
```

- [ ] Zero vectors: __________ (should be 0)
- [ ] Embedding dimension: __________ (should be 1024)
- [ ] Missing descriptions: __________ (should be 0)

### Vector Search Sanity Check
```bash
python -c "
from llm.inference import semantic_search_frames
results = semantic_search_frames('person fighting', video_id='$VIDEO_ID', k=5)
print(f'Results found: {len(results)}')  # Should be > 0
if results:
    print(f'Top score: {results[0][\"score\"]}')  # Should be > 0.6
"
```

- [ ] Results found: __________
- [ ] Top score: __________ (should be > 0.6)

### MongoDB Index Status
```bash
# MongoDB Atlas console → Database → video_intelligence → Search Indexes
# Verify both indexes exist and show "ACTIVE"
```

- [ ] vs_frames_index status: ACTIVE ✅
- [ ] vs_transcripts_index status: ACTIVE ✅

---

## TROUBLESHOOTING

### If Descriptions Are Still Generic

**Check 1**: Verify new prompt is being used
```bash
grep "PRECISE ACTIONS" llm/gen_frame_desc.py
# Should find the new detailed prompt
```
- [ ] New prompt detected: ✅

**Check 2**: Verify model wasn't downgraded
```bash
grep "FIREWORKS_VISION_MODEL" llm/config.py
# Should show kimi-k2p5 or similar VLM
```
- [ ] Vision model is kimi-k2p5: ✅

**Check 3**: Try alternative model
```python
# llm/config.py line 114 - try:
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/qwen2-vl-72b-instruct"
```
- [ ] Alternative model selected: ⚪

### If Vector Search Returns Wrong Results

**Check 1**: MongoDB index is ACTIVE
- [ ] vs_frames_index status in Atlas: ACTIVE ✅

**Check 2**: Embedding dimensions match
```bash
python -c "
from llm.config import frames_col
doc = frames_col.find_one()
print(len(doc['embedding']))  # Should be 1024
"
```
- [ ] Dimension is 1024: ✅

**Check 3**: Documents were inserted
```bash
python -c "
from llm.config import frames_col
print(frames_col.count_documents({'video_id': 'YOUR_VIDEO_ID'}))
"
```
- [ ] Count > 0: ✅

### If Ingestion Is Slow

**Option 1**: Increase batch size
```bash
# In process_frames.py, change batch_size parameter:
process_video_frames(..., batch_size=20)  # from 10
```

**Option 2**: Reduce inter-request delay
```bash
# When calling process_frames.py:
python llm/process_frames.py ... --inter-request-delay 0.2
```

**Option 3**: Run in background
```bash
nohup python llm/process_frames.py ... > ingestion.log 2>&1 &
# Check progress: tail -f ingestion.log
```

- [ ] Performance issue resolved: ⚪

---

## FINAL SIGN-OFF

### Implementation Complete
- [ ] All 4 steps executed successfully
- [ ] No critical errors encountered
- [ ] Vector search quality improved
- [ ] Documentation reviewed

### Deployment Ready
- [ ] Changes committed to version control
- [ ] Updated documentation in place
- [ ] Team notified of changes
- [ ] Monitoring/alerts configured

### Long-Term Monitoring
- [ ] Vector search accuracy tracked
- [ ] Embedding quality monitored
- [ ] Performance metrics recorded
- [ ] Issues logged for future improvements

---

## Notes & Comments

**Date of Execution**: ______________

**Issues Encountered**: 
```
_____________________________________________________________________________

_____________________________________________________________________________
```

**Improvements Observed**:
```
_____________________________________________________________________________

_____________________________________________________________________________
```

**Next Steps**:
```
_____________________________________________________________________________

_____________________________________________________________________________
```

**Sign-Off**: ________________  
**Verified By**: ________________  
**Date**: ________________

---

**✅ READY TO EXECUTE!**

Review this checklist as you execute each step. Complete all checkboxes before moving to next step.

