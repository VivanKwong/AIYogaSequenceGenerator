"""
Simple demo to showcase semantic search working with yoga poses.
Run this to see the embeddings in action!
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_poses_with_embeddings(filepath="yoga_poses.json"):
    """Load poses with their embeddings from JSON."""
    with open(filepath, 'r') as f:
        poses = json.load(f)
    
    for pose in poses:
        pose['embedding'] = np.array(pose['embedding'])
    
    return poses


def semantic_search(query, poses, model, top_k=5):
    """Find poses most similar to a natural language query."""
    query_embedding = model.encode(query)
    
    results = []
    for pose in poses:
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            pose['embedding'].reshape(1, -1)
        )[0][0]
        results.append((pose, float(similarity)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def display_results(query, results):
    """Pretty print search results."""
    print(f"\n{'='*70}")
    print(f"🔍 Query: '{query}'")
    print(f"{'='*70}\n")
    
    for i, (pose, score) in enumerate(results, 1):
        print(f"{i}. {pose['name']} ({pose['sanskrit_name']})")
        print(f"   📊 Similarity: {score:.3f}")
        print(f"   🏷️  Category: {pose['category']} | Intensity: {pose['intensity']}")
        print(f"   ⚡ Energy: {', '.join(pose['energy'])}")
        print(f"   🎯 Targets: {', '.join(pose['target_body_parts'])}")
        if pose['contraindications']:
            print(f"   ⚠️  Avoid with: {', '.join(pose['contraindications'])}")
        print()


def main():
    print("\n" + "="*70)
    print("🧘‍♀️ AI YOGA SEQUENCE GENERATOR - Semantic Search Demo")
    print("="*70)
    
    print("\nLoading poses with embeddings...")
    poses = load_poses_with_embeddings()
    print(f"✓ Loaded {len(poses)} poses")
    
    print("Loading semantic search model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model ready\n")
    
    # Demo searches
    queries = [
        "grounding and calming poses for meditation",
        "energizing fiery poses to build heat",
        "hip flexibility and release",
        "challenging advanced arm balances",
        "gentle stretches for lower back pain",
        "heart opening chest expansion"
    ]
    
    for query in queries:
        results = semantic_search(query, poses, model, top_k=5)
        display_results(query, results)
        input("Press Enter to continue to next query...")
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    print("\n💡 What you just saw:")
    print("  • Natural language queries matched to yoga poses")
    print("  • Semantic understanding (not just keyword matching)")
    print("  • Similarity scoring based on meaning")
    print("\n🚀 Next Steps:")
    print("  • Build sequence generation logic")
    print("  • Add filtering by injuries/contraindications")
    print("  • Create progression arcs (warm-up → peak → cool-down)")
    print("  • Integrate user preferences & intentions")
    print()


if __name__ == "__main__":
    main()
