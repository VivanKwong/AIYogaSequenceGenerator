# AI Yoga Sequence Generator

An intelligent yoga sequence generator that uses semantic embeddings to match poses to user intentions, energy levels, and injuries.

## Features

✨ **Semantic Understanding**: Uses AI embeddings to understand natural language queries
🧘‍♀️ **100+ Yoga Poses**: Comprehensive pose library with metadata
🎯 **Intelligent Matching**: Finds poses based on meaning, not just keywords
⚕️ **Injury-Aware**: Respects contraindications and safety
📊 **Rich Metadata**: Intensity, energy, body parts, teaching cues

## Input Parameters

- **Intention** (e.g. "grounding," "heart-opening")
- **Time** (20 / 45 / 60 min)
- **Level** (beginner / intermediate)
- **Injuries** (e.g. knee, wrist)
- **Energy** (low / steady / fiery)

## Output

A yoga sequence with:
- Pose names (English + Sanskrit)
- Approximate durations
- Teaching cues
- Peak pose + transitions
- Semantic similarity scores

## Getting Started

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Pose Library with Embeddings

```bash
python generate_yoga_poses.py
```

This will:
1. Generate 100 yoga poses with rich metadata
2. Create semantic embeddings using sentence-transformers
3. Save to `yoga_poses.json`

### Test Semantic Search

```bash
python demo_search.py
```

Try natural language queries like:
- "grounding and calming poses for meditation"
- "energizing fiery poses to build heat"
- "hip flexibility and release"
- "gentle stretches for lower back pain"

### Run Full Demo Suite

```bash
python test_semantic_search.py
```

## How It Works

1. **Embeddings**: Each pose is converted to a 384-dimensional vector that represents its semantic meaning
2. **Similarity Search**: User queries are embedded and compared to pose embeddings using cosine similarity
3. **Intelligent Matching**: Finds poses that match the *meaning* of the query, not just keywords
4. **Filtering**: Applies constraints like injuries, intensity, duration

## Technology Stack

- **sentence-transformers**: For generating semantic embeddings
- **all-MiniLM-L6-v2**: Lightweight, fast embedding model
- **scikit-learn**: For cosine similarity calculations
- **numpy**: For efficient vector operations

## Next Steps

- [ ] Build sequence generation logic (warm-up → peak → cool-down)
- [ ] Add transition recommendations between poses
- [ ] Integrate duration planning based on time constraint
- [ ] Create API endpoint for sequence generation
- [ ] Add Spotify playlist integration

## Project Structure

```
.
├── generate_yoga_poses.py    # Pose generation + embedding creation
├── test_semantic_search.py   # Full demo with multiple search modes
├── demo_search.py             # Simple search demo
├── yoga_poses.json            # 100 poses with embeddings
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## License

MIT
