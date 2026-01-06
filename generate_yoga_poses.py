import json
from itertools import cycle

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
DEFAULT_CUES = [
    "Breathe steadily.",
    "Engage with awareness.",
    "Move with control."
]

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
            "cues": DEFAULT_CUES,
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


if __name__ == "__main__":
    yoga_poses = generate_poses(100)

    with open("yoga_poses.json", "w") as f:
        json.dump(yoga_poses, f, indent=2)

    print("Generated yoga_poses.json with 100 poses.")
