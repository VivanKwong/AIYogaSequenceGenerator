"""
Demo script to test semantic search with yoga poses.
This shows how embeddings enable intelligent pose matching.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_poses_with_embeddings(filepath="yoga_poses.json"):
    """Load poses with their embeddings from JSON."""
    with open(filepath, 'r') as f:
        poses = json.load(f)
    
    # Convert embedding lists back to numpy arrays for faster computation
    for pose in poses:
        pose['embedding'] = np.array(pose['embedding'])
    
    return poses


def semantic_search(query, poses, model, top_k=10, filters=None):
    """
    Find poses most similar to a natural language query.
    
    Args:
        query: Natural language search query
        poses: List of pose dictionaries with embeddings
        model: SentenceTransformer model
        top_k: Number of top results to return
        filters: Dict of filters to apply (e.g., {'category': 'standing'})
    
    Returns:
        List of (pose, similarity_score) tuples
    """
    # Generate embedding for the query
    query_embedding = model.encode(query)
    
    # Apply filters if provided
    filtered_poses = poses
    if filters:
        for key, value in filters.items():
            if isinstance(value, list):
                # For list filters, check if pose value is in the list
                filtered_poses = [p for p in filtered_poses if p.get(key) in value]
            else:
                # For single value filters
                filtered_poses = [p for p in filtered_poses if p.get(key) == value]
    
    # Calculate similarity scores
    results = []
    for pose in filtered_poses:
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            pose['embedding'].reshape(1, -1)
        )[0][0]
        results.append((pose, float(similarity)))
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def filter_by_injury(poses, injury):
    """
    Filter out poses that are contraindicated for a specific injury.
    
    Args:
        poses: List of pose dictionaries
        injury: Injury type (e.g., "knee injury", "wrist injury")
    
    Returns:
        Filtered list of safe poses
    """
    return [pose for pose in poses if injury not in pose.get('contraindications', [])]


def display_results(query, results, show_details=True):
    """Pretty print search results."""
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'='*70}\n")
    
    for i, (pose, score) in enumerate(results, 1):
        print(f"{i}. {pose['name']} ({pose['sanskrit_name']})")
        print(f"   Similarity: {score:.3f} | Category: {pose['category']} | Intensity: {pose['intensity']}")
        
        if show_details:
            print(f"   Energy: {', '.join(pose['energy'])}")
            print(f"   Targets: {', '.join(pose['target_body_parts'])}")
            if pose['contraindications']:
                print(f"   ⚠️  Contraindications: {', '.join(pose['contraindications'])}")
        print()


def demo_searches(poses, model):
    """Run several demo searches to showcase semantic understanding."""
    
    print("\n" + "="*70)
    print("DEMO: Semantic Search for Intelligent Yoga Sequences")
    print("="*70)
    
    # Demo 1: Basic semantic search
    print("\n--- Demo 1: Finding Grounding Poses ---")
    results = semantic_search(
        "grounding and calming poses for meditation",
        poses,
        model,
        top_k=5
    )
    display_results("grounding and calming poses for meditation", results, show_details=False)
    
    # Demo 2: Energy-based search
    print("\n--- Demo 2: Finding Energizing Poses ---")
    results = semantic_search(
        "energizing fiery poses to build heat",
        poses,
        model,
        top_k=5
    )
    display_results("energizing fiery poses to build heat", results, show_details=False)
    
    # Demo 3: Body part targeting
    print("\n--- Demo 3: Hip Opening Poses ---")
    results = semantic_search(
        "hip flexibility and release",
        poses,
        model,
        top_k=5
    )
    display_results("hip flexibility and release", results, show_details=False)
    
    # Demo 4: Injury-aware search
    print("\n--- Demo 4: Injury-Aware Search (Knee Injury) ---")
    print("First, find hip openers...")
    results = semantic_search(
        "hip opening and flexibility",
        poses,
        model,
        top_k=10
    )
    print("Then, filter out poses contraindicated for knee injury...")
    safe_poses = [(p, s) for p, s in results if 'knee injury' not in p.get('contraindications', [])]
    display_results("hip opening (safe for knee injury)", safe_poses[:5], show_details=True)
    
    # Demo 5: Challenge level search
    print("\n--- Demo 5: Advanced Challenge Poses ---")
    results = semantic_search(
        "challenging advanced arm balances and inversions",
        poses,
        model,
        top_k=5
    )
    display_results("challenging advanced arm balances", results, show_details=False)
    
    # Demo 6: Intention-based search
    print("\n--- Demo 6: Heart-Opening Flow ---")
    results = semantic_search(
        "heart opening chest expansion uplifting",
        poses,
        model,
        top_k=5
    )
    display_results("heart opening flow", results, show_details=False)


def interactive_search(poses, model):
    """Interactive search mode."""
    print("\n" + "="*70)
    print("INTERACTIVE SEARCH MODE")
    print("="*70)
    print("Enter a natural language query to find matching poses.")
    print("Examples:")
    print("  - 'gentle stretches for lower back'")
    print("  - 'standing poses for strength'")
    print("  - 'calming restorative poses'")
    print("\nType 'quit' to exit.\n")
    
    while True:
        query = input("Search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 🧘‍♀️")
            break
        
        if not query:
            continue
        
        results = semantic_search(query, poses, model, top_k=5)
        display_results(query, results, show_details=True)


def main():
    print("Loading poses with embeddings...")
    poses = load_poses_with_embeddings()
    print(f"✓ Loaded {len(poses)} poses")
    
    print("Loading semantic search model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model ready")
    
    # Run demo searches
    demo_searches(poses, model)
    
    # Optional: Interactive mode
    print("\n" + "="*70)
    response = input("\nWould you like to try interactive search? (y/n): ").strip().lower()
    if response == 'y':
        interactive_search(poses, model)
    else:
        print("\nDemo complete! 🧘‍♀️")
        print("\nNext steps:")
        print("  1. Use these embeddings to build sequence logic")
        print("  2. Combine semantic search with rule-based filters")
        print("  3. Create sequence arcs (warm-up → peak → cool-down)")


if __name__ == "__main__":
    main()
