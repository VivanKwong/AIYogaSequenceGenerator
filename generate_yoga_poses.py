import json
from itertools import cycle
from sentence_transformers import SentenceTransformer
import time

# ---------- CATEGORY TEMPLATES ----------
CATEGORY_CONFIG = {
    "centering": {
        "intensity": 1,
        "energy": ["grounding", "calm"],
        "body_parts": ["spine", "hips"],
        "duration": 3
    },
    "warm-up": {
        "intensity": 1,
        "energy": ["fluid"],
        "body_parts": ["spine", "shoulders"],
        "duration": 2
    },
    "standing": {
        "intensity": 3,
        "energy": ["fiery", "steady"],
        "body_parts": ["legs", "core"],
        "duration": 2
    },
    "balance": {
        "intensity": 3,
        "energy": ["focused"],
        "body_parts": ["legs", "core"],
        "duration": 2
    },
    "core": {
        "intensity": 4,
        "energy": ["fiery"],
        "body_parts": ["core"],
        "duration": 1
    },
    "backbend": {
        "intensity": 3,
        "energy": ["energizing"],
        "body_parts": ["spine", "chest"],
        "duration": 1
    },
    "hip-opener": {
        "intensity": 3,
        "energy": ["releasing"],
        "body_parts": ["hips"],
        "duration": 3
    },
    "twist": {
        "intensity": 2,
        "energy": ["cleansing"],
        "body_parts": ["spine"],
        "duration": 2
    },
    "inversion": {
        "intensity": 4,
        "energy": ["steady"],
        "body_parts": ["core", "neck"],
        "duration": 2
    },
    "arm-balance": {
        "intensity": 5,
        "energy": ["fiery"],
        "body_parts": ["arms", "core", "wrists"],
        "duration": 1
    },
    "peak": {
        "intensity": 5,
        "energy": ["fiery"],
        "body_parts": ["full body"],
        "duration": 1
    },
    "cool-down": {
        "intensity": 1,
        "energy": ["grounding"],
        "body_parts": ["spine"],
        "duration": 3
    },
    "restorative": {
        "intensity": 1,
        "energy": ["calm"],
        "body_parts": ["full body"],
        "duration": 5
    }
}

# ---------- POSE LIBRARY (expandable) ----------
POSES = [
    ("Easy Seat", "Sukhasana", "centering"),
    ("Mountain Pose", "Tadasana", "centering"),
    ("Neck Rolls", "Greeva Sanchalana", "warm-up"),
    ("Cat Pose", "Marjaryasana", "warm-up"),
    ("Cow Pose", "Bitilasana", "warm-up"),
    ("Sun Salutation A", "Surya Namaskar A", "warm-up"),
    ("Chair Pose", "Utkatasana", "standing"),
    ("Warrior I", "Virabhadrasana I", "standing"),
    ("Warrior II", "Virabhadrasana II", "standing"),
    ("Triangle Pose", "Trikonasana", "standing"),
    ("Extended Side Angle", "Utthita Parsvakonasana", "standing"),
    ("Tree Pose", "Vrksasana", "balance"),
    ("Eagle Pose", "Garudasana", "balance"),
    ("Half Moon Pose", "Ardha Chandrasana", "balance"),
    ("Boat Pose", "Navasana", "core"),
    ("Forearm Plank", "Makara Adho Mukha Svanasana", "core"),
    ("Cobra Pose", "Bhujangasana", "backbend"),
    ("Locust Pose", "Salabhasana", "backbend"),
    ("Camel Pose", "Ustrasana", "backbend"),
    ("Bridge Pose", "Setu Bandhasana", "backbend"),
    ("Pigeon Pose", "Eka Pada Rajakapotasana", "hip-opener"),
    ("Goddess Pose", "Utkata Konasana", "hip-opener"),
    ("Seated Forward Fold", "Paschimottanasana", "hip-opener"),
    ("Seated Twist", "Ardha Matsyendrasana", "twist"),
    ("Revolved Lunge", "Parivrtta Anjaneyasana", "twist"),
    ("Shoulder Stand", "Salamba Sarvangasana", "inversion"),
    ("Headstand", "Sirsasana", "inversion"),
    ("Crow Pose", "Bakasana", "arm-balance"),
    ("Side Crow", "Parsva Bakasana", "arm-balance"),
    ("Firefly Pose", "Tittibhasana", "peak"),
    ("Wheel Pose", "Urdhva Dhanurasana", "peak"),
    ("Supine Twist", "Supta Matsyendrasana", "cool-down"),
    ("Happy Baby", "Ananda Balasana", "cool-down"),
    ("Legs Up the Wall", "Viparita Karani", "restorative"),
    ("Corpse Pose", "Savasana", "restorative")
]

# ---------- DEFAULT CUES ----------
CATEGORY_CUES = {
    "centering": [
        "lengthen your spine upward to create space for breath",
        "ground your sit bones down to feel stable",
        "soften your shoulders away from your ears"
    ],

    "warm-up": [
        "move your spine through its full range of motion",
        "circle your shoulders to release tension",
        "coordinate your breath with your movement"
    ],

    "standing": [
        "press your feet firmly into the mat to feel grounded",
        "engage your legs to support your posture",
        "lift your chest slightly to stay upright"
    ],

    "balance": [
        "fix your gaze forward to support balance",
        "engage your core to stabilize your body",
        "root your standing foot down for steadiness"
    ],

    "core": [
        "engage your core muscles to support your spine",
        "draw your navel in toward your spine for stability",
        "lift your chest while keeping your core active"
    ],

    "backbend": [
        "lift your chest forward and up to open the front body",
        "engage your glutes gently to support your lower back",
        "lengthen your spine before bending to avoid compression"
    ],

    "hip-opener": [
        "relax your hips downward to encourage release",
        "keep your hips level to protect your joints",
        "soften your jaw and face to reduce holding"
    ],

    "twist": [
        "lengthen your spine upward before rotating",
        "rotate your torso from the ribcage rather than the shoulders",
        "keep your hips steady to focus the twist"
    ],

    "inversion": [
        "engage your core to support your spine",
        "stack your shoulders over your elbows or wrists for stability",
        "lengthen your neck by lifting away from the shoulders"
    ],

    "arm-balance": [
        "press your hands firmly into the mat to activate your arms",
        "engage your core to lift your body",
        "shift your weight forward gradually to find balance"
    ],

    "peak": [
        "focus your breath to stay present in the pose",
        "engage your whole body to support the shape",
        "maintain steady effort without forcing"
    ],

    "cool-down": [
        "slow your breath to signal the body to relax",
        "release your muscles gradually to unwind effort",
        "support your spine with the mat or props"
    ],

    "restorative": [
        "allow your body to fully relax into support",
        "release your muscles completely with each exhale",
        "rest your weight downward to encourage recovery"
    ]
}


# ---------- CONTRAINDICATION MAP ----------
CONTRAINDICATIONS = {
    "balance": ["ankle injury"],
    "backbend": ["low back injury"],
    "arm-balance": ["wrist injury"],
    "inversion": ["neck injury"],
    "hip-opener": ["knee injury"]
}

# ---------- GENERATION ----------
def generate_poses(target_count=100):
    poses = []
    id_counter = 1
    pose_cycle = cycle(POSES)

    while len(poses) < target_count:
        name, sanskrit, category = next(pose_cycle)
        config = CATEGORY_CONFIG[category]

        pose = {
            "id": f"p{id_counter:03d}",
            "name": name,
            "sanskrit_name": sanskrit,
            "category": category,
            "intensity": config["intensity"],
            "energy": config["energy"],
            "target_body_parts": config["body_parts"],
            "contraindications": CONTRAINDICATIONS.get(category, []),
            "base_duration_min": config["duration"],
            "cues": CATEGORY_CUES.get(category, []),
            "embedding_text": (
                f"{name} {sanskrit}: {category} pose targeting "
                f"{', '.join(config['body_parts'])}. "
                f"Intensity {config['intensity']} with "
                f"{', '.join(config['energy'])} energy."
            )
        }

        poses.append(pose)
        id_counter += 1

    return poses


def generate_embeddings(poses, model_name='all-MiniLM-L6-v2'):
    """
    Generate semantic embeddings for each pose using sentence-transformers.
    
    Args:
        poses: List of pose dictionaries
        model_name: Name of the sentence-transformers model to use
    
    Returns:
        poses with 'embedding' field added
    """
    print(f"Loading embedding model: {model_name}...")
    print("(This may take a moment on first run as the model downloads)")
    
    start_time = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f} seconds")
    
    print(f"\nGenerating embeddings for {len(poses)} poses...")
    start_time = time.time()
    
    # Extract all embedding texts
    embedding_texts = [pose['embedding_text'] for pose in poses]
    
    # Generate embeddings in batch (much faster than one at a time)
    embeddings = model.encode(embedding_texts, show_progress_bar=True)
    
    # Add embeddings to poses
    for pose, embedding in zip(poses, embeddings):
        pose['embedding'] = embedding.tolist()  # Convert numpy array to list for JSON
    
    generation_time = time.time() - start_time
    print(f"✓ Generated {len(poses)} embeddings in {generation_time:.2f} seconds")
    print(f"  ({len(poses)/generation_time:.1f} poses/second)")
    print(f"  Embedding dimensions: {len(embeddings[0])}")
    
    return poses


if __name__ == "__main__":
    print("=" * 60)
    print("AI Yoga Sequence Generator - Pose & Embedding Generation")
    print("=" * 60)
    print()
    
    # Generate poses
    print("Step 1: Generating pose library...")
    yoga_poses = generate_poses(100)
    print(f"✓ Generated {len(yoga_poses)} poses from {len(POSES)} base poses")
    print()
    
    # Generate embeddings
    print("Step 2: Generating semantic embeddings...")
    yoga_poses = generate_embeddings(yoga_poses)
    print()
    
    # Save to JSON
    print("Step 3: Saving to yoga_poses.json...")
    with open("yoga_poses.json", "w") as f:
        json.dump(yoga_poses, f, indent=2)
    
    print("✓ Saved yoga_poses.json")
    print()
    
    # Show sample
    print("=" * 60)
    print("Sample Pose with Embedding:")
    print("=" * 60)
    sample = yoga_poses[0]
    print(f"Name: {sample['name']}")
    print(f"Category: {sample['category']}")
    print(f"Embedding text: {sample['embedding_text']}")
    print(f"Embedding (first 10 values): {sample['embedding'][:10]}...")
    print(f"Embedding (last 10 values): ...{sample['embedding'][-10:]}")
    print()
    print("🧘‍♀️ Ready for intelligent sequence generation!")
