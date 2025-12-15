import pygame
import sys
import os
from typing import Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pygame
import sys
import os
import json
import math
import random
import uuid
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2
import glob
from torch.utils.data import Dataset
import time  # Add this import
import tkinter as tk
from tkinter import filedialog, messagebox
import shutil
# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("AI-Enhanced Drawing and Animation App")  # Fixed method name

# Color and drawing area constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
COLORS = [BLACK, RED, GREEN, BLUE]
SCREEN_RECT = pygame.Rect(0, 0, 1200, 800)  # Add this line
DRAWING_AREA = pygame.Rect(50, 0, 800, 600)  # Updated y-coordinate for 10% higher positioning
BORDER_COLOR = BLACK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ========================
# ROCA Symbolic-Only Capsule System (Production-Ready)
# ========================
# Core Specification:
# - Fully deterministic symbolic pose vectors (UUID5 + seeded RNG)
# - Merge logic purely symbolic (averaged vectors, no neural ops)
# - Coactivations and hierarchy preserved in JSON persistence
# - Compatible with existing CapsuleOverlay / GUI animation layer
# - Visual organization via KIND-BASED BANDS (concentric rings)
#
# Key Invariants:
# 1. All pose_vector values originate from _stable_pose_vector(seed)
# 2. Merge produces averaged pose: [(x+y)/2 for x,y in zip(...)]
# 3. Similarity threshold (SIM_THRESHOLD = 0.88) is cosine-based
# 4. Coactivation counts trigger hierarchy promotion (atomic→sequence→cycle)
# 5. JSON state preserves all capsule data across sessions
# 6. Zero neural/learned components; purely deterministic heuristics
# 7. Shadow tracking: all source capsules preserved for unpacking merges
# 8. Radius bands: each kind occupies a specific radial ring for visual clustering
#
# ========================
# ANIMATOR USE: SYMBOLIC POSE VECTORS FOR ANIMATION CONSISTENCY
# ========================
#
# The symbolic pose vector system ensures REPRODUCIBLE ANIMATION COORDINATES
# across sessions without retraining or learned embeddings.
#
# How it works:
# - Each capsule's name + kind seed produces the same 8D pose vector always
# - Example: create_symbolic_capsule("ArmRaise", "Character") always generates
#   the exact same pose coordinates, deterministically
#
# KEYFRAME ANCHORING:
# 1. Create a capsule for your reference pose:
#    capsule = create_symbolic_capsule("JumpApex", "Character")
#
# 2. Anchor your keyframe to the capsule:
#    capsule_overlay.anchor_keyframe("JumpApex", frame_number)
#
# 3. Later, retrieve the capsule and snap to its pose:
#    pose = capsule_overlay.get_symbolic_pose("JumpApex")
#    # pose is always the same 8D vector, guaranteeing consistency
#
# REPEATED SEQUENCES:
# For animations that repeat (walk cycles, idle poses, attacks):
# - Define capsules once: "WalkCycleStart", "WalkCycleMid", "WalkCycleEnd"
# - Anchor keyframes in your timeline
# - In future sessions, call the same capsule names to snap back to exact poses
# - No neural retraining required; purely symbolic heuristics
#
# ========================
# MERGING AND HIERARCHICAL ANIMATION
# ========================
#
# Combine atomic capsules into composite sequences while preserving traceability.
#
# COMPOSITION WORKFLOW:
# 1. Create atomic capsules (single movements):
#    walk = create_symbolic_capsule("WalkCycle", "Skill")
#    wave = create_symbolic_capsule("WaveHand", "Skill")
#    smile = create_symbolic_capsule("Smile", "Character")
#
# 2. Merge atomics into complex sequences:
#    walk_wave = system.merge_capsules(walk.id, wave.id)
#    # Creates capsule named "WalkCycle+WaveHand"
#
# 3. Merge composites recursively:
#    walk_wave_smile = system.merge_capsules(walk_wave.id, smile.id)
#    # Creates capsule named "WalkCycle+WaveHand+Smile"
#    # walk_wave_smile.shadows tracks all atomic sources
#
# SHADOW TRACKING:
# - Every capsule stores source IDs in its 'shadows' attribute
# - Enables unpacking: walk_wave_smile.shadows → [walk.id, wave.id, smile.id]
# - Preserved in JSON state for full traceability across sessions
#
# POSE COMPUTATION:
# - Merged pose = averaged components: [(x+y)/2 for x,y in zip(pose_a, pose_b)]
# - Commutative: merge(A,B) = merge(B,A) semantically
# - Deterministic: same sources always produce same result
#
# ANIMATOR BENEFITS:
# ✅ Build complex movements incrementally without redefining atomics
# ✅ Trace any composite back to its atomic components
# ✅ Create variations by merging base movements with different moods
# ✅ Generate procedural animations from atomic combinations
# ✅ Team collaboration: share atomic library, compose freely
#
# ========================
# VISUAL ORGANIZATION: RADIUS AND BANDS
# ========================
#
# Capsules organize into concentric rings based on their kind:
#
# KIND_BANDS:
# - Core (0-8%):       Central core personality
# - Character (18-26%): Emotions and character traits
# - Style (26-32%):    Visual and behavioral styles
# - Skill (32-38%):    Movements and actions
# - Topic (38-44%):    Topics and concepts
# - Memory (44-50%):   Learned patterns
# - Experimental (50-58%): Experimental capsules
#
# Animator Use:
# - Quickly identify where different kinds of capsules appear
# - Drag-and-drop to arrange and compose sequences
# - Visual clustering makes organization intuitive
# - Each ring color-coded for easy recognition
#
# ========================
# DETERMINISTIC YET FLEXIBLE: EMERGENT SUGGESTIONS
# ========================
#
# Paradox Resolution:
# - Deterministic: Same capsule name = same pose vector (always)
# - Flexible: Coactivation weights emerge from usage patterns
#
# How It Works:
# - Coactivations track which capsules are frequently used together
# - System learns patterns from animator interactions
# - Suggests likely pairings based on history
# - No hardcoded rules; purely emergent from usage
#
# Animator Use:
# - System suggests related capsules when you select one
# - Find similar capsules by pose (semantic alternatives)
# - Get recommended sequences based on what works well together
# - Highlight successful combinations in UI
#
# Methods for Suggestions:
# - get_related_capsules(id): Capsules frequently paired with this one
# - find_similar_capsules(id): Capsules with similar pose vectors
# - suggest_next_capsule(id): Best single pairing
# - recommend_sequence(id, depth): Full suggested sequence
# - get_suggested_pairings(id, top_n): Top suggestions with weights
#
# ========================
# BENEFITS (ALL SYSTEMS):
# ✅ Session-independent animation rig coordinates (Keyframe Anchoring)
# ✅ Reproducible pose layouts across team/time (Symbolic ROCA)
# ✅ No learned model drift or data loss
# ✅ Perfect for animation libraries and asset reuse
# ✅ Deterministic for procedural animation generation
# ✅ Full traceability for composition and debugging (Merging)
# ✅ Intuitive visual organization for animator workflows (Bands)
# ✅ Drag-and-drop interface for building sequences
# ✅ Intelligent suggestions based on usage patterns (Coactivations)
# ✅ Flexible yet predictable system behavior
# ========================

# -------------------------
# ROCA capsule system (standalone)
# -------------------------
KIND_BANDS: dict[str, tuple[float, float]] = {
    "Core": (0.00, 0.08),
    "Character": (0.18, 0.26),
    "Style": (0.26, 0.32),
    "Skill": (0.32, 0.38),
    "Topic": (0.38, 0.44),
    "Memory": (0.44, 0.50),
    "Experimental": (0.50, 0.58),
}

RING_COLORS = (
    (70, 90, 120),
    (80, 105, 140),
    (95, 125, 165),
    (110, 145, 190),
    (125, 165, 210),
)

STATE_PATH = Path(__file__).with_name("ringpose_state.json")
RINGPOSE_STATE_PATH = STATE_PATH
POSE_DIM = 8
SIM_THRESHOLD = 0.88


def _load_ringpose_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_ringpose_state(path: Path, payload: dict) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _band_for_kind(kind: str) -> tuple[float, float]:
    k = kind.strip() if kind else "Memory"
    return KIND_BANDS.get(k, KIND_BANDS.get("Memory", (0.44, 0.50)))


def _color_for_kind(kind: str) -> tuple[int, int, int]:
    k = kind.lower()
    if k == "character":
        return (220, 190, 120)
    if k in {"skill", "core"}:
        return (140, 210, 180)
    return (120, 180, 255)


def _stable_pose_vector(seed: str, dim: int = POSE_DIM) -> list[float]:
    # Deterministic vector based on seed
    h = uuid.uuid5(uuid.NAMESPACE_URL, seed)
    rnd = random.Random(h.int)  # Use int for consistency
    vec = [rnd.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def snap_to_symbolic(frame: int, snap_interval: int = 5) -> int:
    """Snap frame number to nearest symbolic frame boundary.
    
    Used for timeline synchronization with capsule keyframes.
    Ensures frames align to symbolic boundaries (e.g., every 5 frames).
    
    Args:
        frame: The frame number to snap
        snap_interval: Interval for snapping (default 5 frames)
    
    Returns:
        int: The snapped frame number
    
    Example:
        snap_to_symbolic(12, snap_interval=5)  # Returns 10
        snap_to_symbolic(14, snap_interval=5)  # Returns 15
        snap_to_symbolic(20, snap_interval=5)  # Returns 20 (already aligned)
    """
    return (frame // snap_interval) * snap_interval


def create_symbolic_capsule(name: str, kind: str, system: "ROCASystem") -> "Capsule":
    """Convenience helper for creating capsules with deterministic symbolic poses.
    
    Ensures all capsule creation goes through symbolic heuristics, never learned vectors.
    
    ANIMATOR WORKFLOW EXAMPLE:
    ──────────────────────────
    # 1. Create a reference pose capsule for your animation keyframe
    jump_apex = create_symbolic_capsule("JumpApex", "Character", capsule_system)
    # → Always generates the same 8D pose vector for "JumpApex"
    
    # 2. In CapsuleOverlay, anchor your keyframe to this capsule
    capsule_overlay.anchor_keyframe("JumpApex", frame_number=5)
    # → Stored persistently in ringpose_state.json
    
    # 3. Later (even in a new session), retrieve the pose
    pose = capsule_overlay.get_symbolic_pose("JumpApex")
    # → Guaranteed to be the exact same 8D vector, reproducible forever
    
    # 4. For walk cycles, create multiple capsules:
    create_symbolic_capsule("WalkStart", "Character", capsule_system)
    create_symbolic_capsule("WalkMid", "Character", capsule_system)
    create_symbolic_capsule("WalkEnd", "Character", capsule_system)
    # → Each has its own stable pose; anchor keyframes to each
    
    BENEFITS:
    • Session-independent: Pose vectors never change across sessions
    • No neural drift: Purely symbolic, deterministic heuristics
    • Team-shareable: Same capsule name = same pose for all animators
    • Asset library: Build animation libraries keyed to capsule names
    • Reproducible: Perfect for procedural animation generation
    """
    vec = _stable_pose_vector(name + kind)
    return system.add_capsule(name, kind, vec)


class OrbitRing:
    def __init__(self, radius: int, color: tuple[int, int, int]):
        self.radius = int(radius)
        self.color = color

    def draw(self, screen: pygame.Surface, center: tuple[int, int]) -> None:
        pygame.draw.circle(screen, self.color, center, int(self.radius), 1)


class PersonalityCore:
    def __init__(self, pos: tuple[int, int]):
        self.x, self.y = pos
        self.base_radius = 34
        self.pulse_amp = 4
        self.pulse_speed = 0.002
        self.time = 0
        self.kick = 0.0

    def set_pos(self, pos: tuple[int, int]) -> None:
        self.x, self.y = pos

    def update(self, dt_ms: int) -> None:
        self.time += dt_ms
        self.kick = max(0.0, self.kick - dt_ms * 0.02)

    def draw(self, screen: pygame.Surface) -> None:
        pulse = math.sin(self.time * self.pulse_speed) * self.pulse_amp
        r = int(self.base_radius + pulse + self.kick)

        for i in range(7, 0, -1):
            glow_r = r + i * 6
            alpha = max(10, 50 - i * 6)
            surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 200, 120, alpha), (glow_r, glow_r), glow_r)
            screen.blit(surf, (self.x - glow_r, self.y - glow_r))

        pygame.draw.circle(screen, (255, 248, 220), (self.x, self.y), r)


class Capsule:
    def __init__(
        self,
        name: str,
        kind: str,
        pose_vector: list[float],
        *,
        shadows: Optional[list[str]] = None,
        level: str = "atomic",
        start_frame: int = 0,
        end_frame: int = 0,
        color: tuple = (200, 200, 200),
        tags: Optional[list[str]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.kind = kind
        self.pose_vector = pose_vector
        self.shadows = list(shadows or [])
        self.level = level
        self.parent_id: Optional[str] = None
        
        # Frame-based timeline information
        self.start_frame = start_frame  # First frame this capsule appears on timeline
        self.end_frame = end_frame      # Last frame this capsule appears on timeline
        self.color = color              # UI color for visual identification (R, G, B)
        self.tags = list(tags or [])    # Semantic tags: ["walk_cycle", "signature_pose", "acting_beat"]
        
        # Orbital mechanics for visual display
        self.satellite_radius = 28.0
        self.local_theta = random.random() * math.tau
        self.local_speed = 0.001 + random.random() * 0.0005
        self.theta = random.random() * math.tau
        self.radius = 0.0
        self.target_radius = 0.0
        self.orbit_speed = 0.0004 + random.random() * 0.0003
        self.pulse = 0.0
        
        # Confidence tracking: safety rails and affordances
        self.usage_count = 0           # How many times used
        self.correction_count = 0      # How many times corrected/refined
        
        # Scope locking: prevent accidental overlearning
        self.scope = "universal"       # "universal", "character-specific", "one-off"
        self.scope_context: Optional[str] = None  # e.g., "elf", "human", "monster"
        
        # Inspection mode: debuggability and transparency
        self.last_modified = 0.0       # Timestamp of last modification (seconds since epoch)
        
        # Capsule freezing: trust feature for professional animators
        self.is_frozen = False         # Once perfect, freeze it
        self.freeze_reason: Optional[str] = None  # Why it's frozen: "walk_cycle", "signature_pose", "acting_beat"
        
        # In-between tracking: for live interpolation updates
        self.ref_start_id: Optional[str] = None  # Reference to start pose for in-betweens
        self.ref_end_id: Optional[str] = None    # Reference to end pose for in-betweens
        self.interpolation_ratio: float = 0.5    # How far along the interpolation (0.0 to 1.0)
        self.is_inbetween = False                # True if this is auto-generated in-between

    def update(self, dt_ms: int) -> None:
        seconds = dt_ms / 1000.0
        smooth = 1.0 - math.exp(-2.4 * seconds)
        self.radius += (self.target_radius - self.radius) * smooth
        self.theta = (self.theta + self.orbit_speed * dt_ms) % math.tau
        self.local_theta = (self.local_theta + self.local_speed * dt_ms) % math.tau
        self.pulse = max(0.0, self.pulse - dt_ms * 0.01)
    
    def get_confidence(self) -> float:
        """Compute confidence level (0.0 to 1.0) based on usage and corrections.
        
        Returns:
            float: Confidence from 0.0 (experimental) to 1.0 (highly trusted)
                   - 0 usage: 0.1 (experimental)
                   - 1+ usage: grows with usage count
                   - Corrections reduce confidence (1 correction = -0.2)
        """
        if self.usage_count == 0:
            return 0.1  # Experimental: never used
        
        # Confidence grows with usage (logarithmic, approaches 1.0)
        base_confidence = min(0.95, 0.5 + math.log10(self.usage_count + 1) * 0.15)
        
        # Corrections reduce confidence
        correction_penalty = self.correction_count * 0.2
        
        return max(0.1, min(1.0, base_confidence - correction_penalty))
    
    def increment_usage(self) -> None:
        """Called when capsule is used in animation."""
        self.usage_count += 1
    
    def increment_correction(self) -> None:
        """Called when capsule is corrected or refined by animator."""
        self.correction_count += 1
    
    def set_scope(self, scope: str, context: Optional[str] = None) -> None:
        """Set the scope of this capsule to control generalization.
        
        Args:
            scope: "universal" (applies everywhere), "character-specific" (tied to context),
                   or "one-off" (don't generalize)
            context: Required for "character-specific", e.g., "elf", "human", "monster"
        """
        if scope not in ("universal", "character-specific", "one-off"):
            raise ValueError(f"Invalid scope: {scope}")
        
        self.scope = scope
        if scope == "character-specific" and context:
            self.scope_context = context
        elif scope == "one-off":
            self.scope_context = None
    
    def applies_to_context(self, context: Optional[str] = None) -> bool:
        """Check if this capsule applies in the given context.
        
        Args:
            context: Character type or context name, e.g., "elf", "human"
        
        Returns:
            True if capsule should be used in this context
        """
        if self.scope == "universal":
            return True
        if self.scope == "one-off":
            return False  # One-off doesn't generalize anywhere
        if self.scope == "character-specific":
            return context == self.scope_context  # Only in matching context
        return False
    
    def get_scope_info(self) -> str:
        """Get human-readable scope information."""
        if self.scope == "universal":
            return "Universal (applies everywhere)"
        elif self.scope == "character-specific":
            return f"Character-specific ({self.scope_context})"
        else:
            return "One-off (experimental only)"
    
    def touch_modification(self) -> None:
        """Update last_modified timestamp to current time."""
        if not self.is_frozen:
            self.last_modified = time.time()
    
    def get_last_modified_str(self) -> str:
        """Get human-readable last modification time."""
        if self.last_modified == 0.0:
            return "Never"
        from datetime import datetime
        dt = datetime.fromtimestamp(self.last_modified)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def freeze(self, reason: Optional[str] = None) -> None:
        """Freeze this capsule to prevent modifications.
        
        Perfect walk cycles, signature poses, and acting beats can be frozen
        to ensure they don't get accidentally changed.
        
        Args:
            reason: Why it's frozen: "walk_cycle", "signature_pose", "acting_beat", etc.
        """
        self.is_frozen = True
        self.freeze_reason = reason
    
    def unfreeze(self) -> None:
        """Unfreeze this capsule to allow modifications again."""
        self.is_frozen = False
        self.freeze_reason = None
    
    def get_freeze_status(self) -> str:
        """Get human-readable freeze status."""
        if not self.is_frozen:
            return "Active (can be modified)"
        if self.freeze_reason:
            return f"Frozen ({self.freeze_reason})"
        return "Frozen"

    def pos(self, center: tuple[int, int], parent_pos: Optional[tuple[int, int]] = None) -> tuple[int, int]:
        if parent_pos is not None:
            px, py = parent_pos
            r = max(12.0, self.satellite_radius)
            return (int(px + math.cos(self.local_theta) * r), int(py + math.sin(self.local_theta) * r))
        cx, cy = center
        return (int(cx + math.cos(self.theta) * self.radius), int(cy + math.sin(self.theta) * self.radius))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "pose_vector": self.pose_vector,
            "shadows": self.shadows,
            "level": self.level,
            "parent_id": self.parent_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "color": self.color,
            "tags": self.tags,
            "satellite_radius": self.satellite_radius,
            "local_theta": self.local_theta,
            "theta": self.theta,
            "radius": self.radius,
            "target_radius": self.target_radius,
            "usage_count": self.usage_count,
            "correction_count": self.correction_count,
            "scope": self.scope,
            "scope_context": self.scope_context,
            "last_modified": self.last_modified,
            "is_frozen": self.is_frozen,
            "freeze_reason": self.freeze_reason,
            "ref_start_id": self.ref_start_id,
            "ref_end_id": self.ref_end_id,
            "interpolation_ratio": self.interpolation_ratio,
            "is_inbetween": self.is_inbetween,
        }

    @staticmethod
    def from_dict(data: dict) -> "Capsule":
        cap = Capsule(
            name=data.get("name", "capsule"),
            kind=data.get("kind", "Memory"),
            pose_vector=list(data.get("pose_vector", [])),
            shadows=list(data.get("shadows", [])),
            level=data.get("level", "atomic"),
            start_frame=int(data.get("start_frame", 0)),
            end_frame=int(data.get("end_frame", 0)),
            color=tuple(data.get("color", (200, 200, 200))),
            tags=list(data.get("tags", [])),
        )
        cap.id = data.get("id", cap.id)
        cap.parent_id = data.get("parent_id")
        cap.satellite_radius = float(data.get("satellite_radius", 28.0))
        cap.local_theta = float(data.get("local_theta", cap.local_theta))
        cap.theta = float(data.get("theta", cap.theta))
        cap.radius = float(data.get("radius", 0.0))
        cap.target_radius = float(data.get("target_radius", 0.0))
        cap.usage_count = int(data.get("usage_count", 0))
        cap.correction_count = int(data.get("correction_count", 0))
        cap.scope = data.get("scope", "universal")
        cap.scope_context = data.get("scope_context")
        cap.last_modified = float(data.get("last_modified", 0.0))
        cap.is_frozen = bool(data.get("is_frozen", False))
        cap.freeze_reason = data.get("freeze_reason")
        cap.ref_start_id = data.get("ref_start_id")
        cap.ref_end_id = data.get("ref_end_id")
        cap.interpolation_ratio = float(data.get("interpolation_ratio", 0.5))
        cap.is_inbetween = bool(data.get("is_inbetween", False))
        return cap


class ROCASystem:
    def __init__(self, center: tuple[int, int], radius_scale: float):
        self.center = center
        self.radius_scale = radius_scale
        self.capsules: dict[str, Capsule] = {}
        self.coactivations: dict[tuple[str, str], float] = {}
        self.atomic_to_sequence = 3
        self.sequence_to_cycle = 6

    def _assign_radius(self, cap: Capsule) -> None:
        inner_frac, outer_frac = _band_for_kind(cap.kind)
        inner = self.radius_scale * inner_frac
        outer = self.radius_scale * outer_frac
        jitter = random.uniform(-8.0, 8.0)
        cap.target_radius = max(0.0, random.uniform(inner, outer) + jitter)
        if cap.radius <= 0:
            cap.radius = cap.target_radius

    def add_capsule(self, name: str, kind: str, pose_vector: Optional[list[float]] = None, *, shadows: Optional[list[str]] = None, auto_merge: bool = True) -> Capsule:
        if pose_vector is None:
            pose_vector = _stable_pose_vector(name + kind)

        new_cap = Capsule(name, kind, pose_vector, shadows=shadows)

        # similarity check
        best_sim = -1.0
        best_cap: Optional[Capsule] = None
        for cap in self.capsules.values():
            try:
                dot = sum(a * b for a, b in zip(pose_vector, cap.pose_vector))
                na = math.sqrt(sum(a * a for a in pose_vector)) or 1e-9
                nb = math.sqrt(sum(b * b for b in cap.pose_vector)) or 1e-9
                sim = dot / (na * nb)
            except Exception:
                sim = -1.0
            if sim > best_sim:
                best_sim = sim
                best_cap = cap

        if best_cap is not None and best_sim >= SIM_THRESHOLD:
            if auto_merge:
                self._assign_radius(new_cap)
                self.capsules[new_cap.id] = new_cap
                merged = self.merge_capsules(best_cap.id, new_cap.id, delete_sources=True)
                if merged:
                    merged.pulse = 10.0
                    return merged
            self.reinforce_coactivation(best_cap.id, best_cap.id, weight=0.2)
            best_cap.pulse = 8.0
            return best_cap

        self._assign_radius(new_cap)
        self.capsules[new_cap.id] = new_cap
        return new_cap

    def merge_capsules(self, a_id: str, b_id: str, *, delete_sources: bool = False) -> Optional[Capsule]:
        a = self.capsules.get(a_id)
        b = self.capsules.get(b_id)
        if not a or not b:
            return None
        merged_pose = [(x + y) / 2.0 for x, y in zip(a.pose_vector, b.pose_vector)] if a.pose_vector and b.pose_vector else a.pose_vector or b.pose_vector
        shadows = list(dict.fromkeys((a.shadows or []) + [a.id] + (b.shadows or []) + [b.id]))
        merged = self.add_capsule(f"{a.name}+{b.name}", a.kind, merged_pose, shadows=shadows, auto_merge=False)
        a.parent_id = merged.id
        b.parent_id = merged.id
        if delete_sources:
            self.capsules.pop(a_id, None)
            self.capsules.pop(b_id, None)
        return merged

    def reinforce_coactivation(self, id1: str, id2: str, *, weight: float = 1.0) -> None:
        key = (id1, id2) if id1 == id2 else tuple(sorted((id1, id2)))
        self.coactivations[key] = self.coactivations.get(key, 0.0) + weight
        self._maybe_promote_pair(key)

    def _maybe_promote_pair(self, key: tuple[str, str]) -> None:
        if len(key) != 2:
            return
        a = self.capsules.get(key[0])
        b = self.capsules.get(key[1])
        if not a or not b:
            return
        count = self.coactivations.get(key, 0.0)
        target_level = None
        if (a.level == "atomic" or b.level == "atomic") and count >= self.atomic_to_sequence:
            target_level = "sequence"
        if (a.level == "sequence" or b.level == "sequence") and count >= self.sequence_to_cycle:
            target_level = "cycle"
        if not target_level:
            return
        merged_pose = [(x + y) / 2.0 for x, y in zip(a.pose_vector, b.pose_vector)] if a.pose_vector and b.pose_vector else a.pose_vector or b.pose_vector
        name = f"{a.name}×{b.name} ({target_level})"
        merged = self.add_capsule(name, a.kind, merged_pose, shadows=[a.id, b.id], auto_merge=False)
        merged.level = target_level
        merged.pulse = 10.0
        a.parent_id = merged.id
        b.parent_id = merged.id

    def to_dict(self) -> dict:
        return {
            "capsules": [c.to_dict() for c in self.capsules.values()],
            "coactivations": [[k[0], k[1], w] for k, w in self.coactivations.items()],
            "thresholds": {
                "atomic_to_sequence": self.atomic_to_sequence,
                "sequence_to_cycle": self.sequence_to_cycle,
            },
        }
    
    def unpack_capsule(self, capsule_id: str) -> list["Capsule"]:
        """Recursively unpack a composite capsule to its atomic components.
        
        Args:
            capsule_id: ID of capsule to unpack
        
        Returns:
            List of atomic capsules that compose this capsule
        """
        capsule = self.capsules.get(capsule_id)
        if not capsule:
            return []
        
        # Atomic capsule (no shadows)
        if not capsule.shadows:
            return [capsule]
        
        # Composite capsule: recursively unpack all shadows
        result = []
        for shadow_id in capsule.shadows:
            result.extend(self.unpack_capsule(shadow_id))
        
        return result
    
    def find_composites_containing(self, capsule_id: str) -> list["Capsule"]:
        """Find all composite capsules that include a given capsule as a component.
        
        Args:
            capsule_id: ID of capsule to search for
        
        Returns:
            List of composite capsules containing this capsule
        """
        results = []
        for cap in self.capsules.values():
            if capsule_id in cap.shadows:
                results.append(cap)
        return results
    
    def get_component_names(self, capsule_id: str) -> list[str]:
        """Get the names of all atomic components in a composite.
        
        Args:
            capsule_id: ID of capsule
        
        Returns:
            List of atomic capsule names
        """
        components = self.unpack_capsule(capsule_id)
        return [c.name for c in components]
    
    def merge_and_anchor(self, capsule_a_id: str, capsule_b_id: str, 
                        frame_number: int, overlay: "CapsuleOverlay") -> bool:
        """Convenience method: merge two capsules and anchor to a keyframe.
        
        Args:
            capsule_a_id: ID of first capsule
            capsule_b_id: ID of second capsule
            frame_number: Frame to anchor the merged capsule
            overlay: CapsuleOverlay for anchoring
        
        Returns:
            True if successful, False otherwise
        """
        merged = self.merge_capsules(capsule_a_id, capsule_b_id)
        if not merged:
            return False
        
        overlay.anchor_keyframe(merged.name, frame_number)
        return True
    
    def get_related_capsules(self, capsule_id: str, threshold: float = 0.1) -> list["Capsule"]:
        """Find capsules frequently used with a given capsule (by coactivation).
        
        Emergent behavior: system learns what pairs work well together.
        
        Args:
            capsule_id: ID of reference capsule
            threshold: Minimum coactivation weight to include (default 0.1)
        
        Returns:
            List of capsules sorted by coactivation strength (highest first)
        """
        related = {}
        for (id1, id2), weight in self.coactivations.items():
            if id1 == capsule_id and weight >= threshold:
                cap = self.capsules.get(id2)
                if cap:
                    related[id2] = weight
            elif id2 == capsule_id and weight >= threshold:
                cap = self.capsules.get(id1)
                if cap:
                    related[id1] = weight
        
        # Sort by weight descending
        sorted_ids = sorted(related.keys(), key=lambda k: related[k], reverse=True)
        return [self.capsules[id] for id in sorted_ids if id in self.capsules]
    
    def find_similar_capsules(self, capsule_id: str, threshold: float = 0.75) -> list["Capsule"]:
        """Find capsules with similar pose vectors (by cosine similarity).
        
        Helps animator find semantic alternatives in the pose space.
        
        Args:
            capsule_id: ID of reference capsule
            threshold: Minimum cosine similarity 0-1 (default 0.75)
        
        Returns:
            List of similar capsules sorted by similarity (highest first)
        """
        ref = self.capsules.get(capsule_id)
        if not ref or not ref.pose_vector:
            return []
        
        similarities = {}
        for cap in self.capsules.values():
            if cap.id == capsule_id or not cap.pose_vector:
                continue
            
            # Cosine similarity
            try:
                dot = sum(a * b for a, b in zip(ref.pose_vector, cap.pose_vector))
                norm_ref = math.sqrt(sum(a*a for a in ref.pose_vector)) or 1e-9
                norm_cap = math.sqrt(sum(a*a for a in cap.pose_vector)) or 1e-9
                sim = dot / (norm_ref * norm_cap)
                
                if sim >= threshold:
                    similarities[cap.id] = sim
            except Exception:
                continue
        
        # Sort by similarity descending
        sorted_ids = sorted(similarities.keys(), key=lambda k: similarities[k], reverse=True)
        return [self.capsules[id] for id in sorted_ids if id in self.capsules]
    
    def get_coactivation_strength(self, id1: str, id2: str) -> float:
        """Get the coactivation weight between two capsules.
        
        Higher values indicate more frequent pairing/usage.
        
        Args:
            id1: First capsule ID
            id2: Second capsule ID
        
        Returns:
            Coactivation weight (0.0 if never paired)
        """
        key = (id1, id2) if id1 == id2 else tuple(sorted((id1, id2)))
        return self.coactivations.get(key, 0.0)
    
    def suggest_next_capsule(self, current_id: str) -> Optional["Capsule"]:
        """Suggest the best capsule to combine with the current one.
        
        Based on coactivation history and usage patterns.
        
        Args:
            current_id: ID of current capsule
        
        Returns:
            Most frequently paired capsule, or None if no suggestions
        """
        related = self.get_related_capsules(current_id, threshold=0.0)
        return related[0] if related else None
    
    def recommend_sequence(self, capsule_id: str, depth: int = 2) -> list["Capsule"]:
        """Recommend a complete sequence starting from a capsule.
        
        Builds a sequence by greedily selecting the most frequent pairing.
        
        Args:
            capsule_id: Starting capsule ID
            depth: How many capsules to suggest (default 2)
        
        Returns:
            List of capsules forming a suggested sequence
        """
        if capsule_id not in self.capsules:
            return []
        
        sequence = [self.capsules[capsule_id]]
        current_id = capsule_id
        
        for _ in range(depth - 1):
            next_cap = self.suggest_next_capsule(current_id)
            if not next_cap or next_cap.id in [c.id for c in sequence]:
                break  # Avoid cycles
            sequence.append(next_cap)
            current_id = next_cap.id
        
        return sequence
    
    def get_suggested_pairings(self, capsule_id: str, top_n: int = 5) -> list[tuple["Capsule", float]]:
        """Get the top N suggested pairings for a capsule.
        
        Useful for UI suggestion panels.
        
        Args:
            capsule_id: ID of capsule
            top_n: How many suggestions to return (default 5)
        
        Returns:
            List of (capsule, coactivation_weight) tuples sorted by weight
        """
        related = self.get_related_capsules(capsule_id, threshold=0.0)
        results = []
        for cap in related[:top_n]:
            weight = self.get_coactivation_strength(capsule_id, cap.id)
            results.append((cap, weight))
        return results
    
    def get_capsule_confidence(self, capsule_id: str) -> float:
        """Get confidence level (0.0-1.0) for a capsule.
        
        Args:
            capsule_id: ID of capsule
        
        Returns:
            Confidence level, or 0.0 if capsule not found
        """
        cap = self.capsules.get(capsule_id)
        return cap.get_confidence() if cap else 0.0
    
    def record_capsule_usage(self, capsule_id: str) -> None:
        """Record that a capsule was used in animation.
        
        Increments usage_count, which affects confidence level.
        
        Args:
            capsule_id: ID of capsule to record usage for
        """
        cap = self.capsules.get(capsule_id)
        if cap:
            cap.increment_usage()
    
    def record_capsule_correction(self, capsule_id: str) -> None:
        """Record that a capsule was corrected or refined.
        
        Increments correction_count, which reduces confidence.
        Useful for tracking when animator manually adjusts a capsule.
        
        Args:
            capsule_id: ID of capsule to record correction for
        """
        cap = self.capsules.get(capsule_id)
        if cap:
            cap.increment_correction()
    
    def get_confidence_report(self) -> dict[str, dict]:
        """Get a report of all capsules with their confidence levels.
        
        Returns:
            Dict mapping capsule name to confidence info:
            {
                "Walk": {
                    "confidence": 0.85,
                    "usage_count": 12,
                    "correction_count": 1,
                    "status": "trusted"
                },
                ...
            }
        """
        report = {}
        for cap in self.capsules.values():
            confidence = cap.get_confidence()
            
            # Classify status
            if confidence >= 0.8:
                status = "trusted"
            elif confidence >= 0.5:
                status = "stable"
            elif confidence >= 0.2:
                status = "experimental"
            else:
                status = "untested"
            
            report[cap.name] = {
                "confidence": round(confidence, 2),
                "usage_count": cap.usage_count,
                "correction_count": cap.correction_count,
                "status": status,
            }
        
        return report
    
    def set_capsule_scope(self, capsule_id: str, scope: str, context: Optional[str] = None) -> bool:
        """Set the scope of a capsule to control generalization.
        
        Args:
            capsule_id: ID of capsule
            scope: "universal", "character-specific", or "one-off"
            context: Required for "character-specific", e.g., "elf", "human"
        
        Returns:
            True if successful, False if capsule not found
        """
        cap = self.capsules.get(capsule_id)
        if not cap:
            return False
        try:
            cap.set_scope(scope, context)
            return True
        except ValueError:
            return False
    
    def get_applicable_capsules(self, context: Optional[str] = None) -> list["Capsule"]:
        """Get capsules that apply in the given context.
        
        Filters out one-off capsules and character-specific capsules that don't match.
        
        Args:
            context: Character type or None for universal only
        
        Returns:
            List of applicable capsules
        """
        return [cap for cap in self.capsules.values() if cap.applies_to_context(context)]
    
    def get_scope_summary(self) -> dict[str, list[str]]:
        """Get a summary of capsule scopes.
        
        Returns:
            {
                "universal": ["Walk", "Run", ...],
                "character-specific": {"elf": ["elf_walk"], "human": ["human_run"]},
                "one-off": ["experimental_pose_1", ...]
            }
        """
        universal = []
        character_specific = {}
        one_off = []
        
        for cap in self.capsules.values():
            if cap.scope == "universal":
                universal.append(cap.name)
            elif cap.scope == "character-specific":
                ctx = cap.scope_context or "unknown"
                if ctx not in character_specific:
                    character_specific[ctx] = []
                character_specific[ctx].append(cap.name)
            elif cap.scope == "one-off":
                one_off.append(cap.name)
        
        return {
            "universal": universal,
            "character-specific": character_specific,
            "one-off": one_off,
        }
    
    def get_capsule_inspection(self, capsule_id: str, keyframe_anchors: Optional[dict] = None) -> Optional[dict]:
        """Get complete inspection data for a capsule.
        
        Shows which frames reference it, which transitions use it, and when modified.
        Turns ROCA into a debuggable system, not a black box.
        
        Args:
            capsule_id: ID of capsule to inspect
            keyframe_anchors: Optional dict of frame → capsule_name mappings (from CapsuleOverlay)
        
        Returns:
            Dict with frames, transitions, and metadata, or None if capsule not found
        """
        cap = self.capsules.get(capsule_id)
        if not cap:
            return None
        
        # Find frames that reference this capsule
        frames = []
        if keyframe_anchors:
            for frame, capsule_name in keyframe_anchors.items():
                if capsule_name == cap.name:
                    frames.append(int(frame))
        frames.sort()
        
        # Find transitions (coactivations) that use this capsule
        transitions = []
        for (id1, id2), weight in self.coactivations.items():
            if id1 == capsule_id or id2 == capsule_id:
                other_id = id2 if id1 == capsule_id else id1
                other_cap = self.capsules.get(other_id)
                if other_cap:
                    transitions.append({
                        "partner": other_cap.name,
                        "weight": round(weight, 2),
                    })
        
        # Sort transitions by weight
        transitions.sort(key=lambda t: t["weight"], reverse=True)
        
        return {
            "capsule_name": cap.name,
            "capsule_kind": cap.kind,
            "capsule_id": capsule_id,
            "level": cap.level,
            "confidence": round(cap.get_confidence(), 2),
            "usage_count": cap.usage_count,
            "correction_count": cap.correction_count,
            "scope": cap.scope,
            "scope_context": cap.scope_context,
            "last_modified": cap.get_last_modified_str(),
            "frames_referencing": frames,
            "frame_count": len(frames),
            "transitions": transitions,
            "transition_count": len(transitions),
            "is_composite": len(cap.shadows) > 0,
            "components": cap.shadows[:5] if cap.shadows else [],  # Show first 5
        }
    
    def find_capsule_by_name(self, name: str) -> Optional[str]:
        """Find a capsule ID by name.
        
        Args:
            name: Capsule name to search for
        
        Returns:
            Capsule ID if found, None otherwise
        """
        for cap in self.capsules.values():
            if cap.name == name:
                return cap.id
        return None
    
    def get_inspection_summary(self, keyframe_anchors: Optional[dict] = None) -> dict:
        """Get inspection summary for all capsules.
        
        Useful for debugging and understanding system state.
        
        Args:
            keyframe_anchors: Optional dict of frame → capsule_name mappings
        
        Returns:
            Dict with summary of all capsules by status and usage
        """
        summary = {
            "total_capsules": len(self.capsules),
            "by_usage": {
                "never_used": [],
                "rarely_used": [],  # 1-5 uses
                "frequently_used": [],  # 6+
            },
            "by_status": {
                "trusted": [],
                "stable": [],
                "experimental": [],
                "untested": [],
            },
            "by_type": {},
            "most_referenced_frames": {},
            "most_active_transitions": [],
        }
        
        # Categorize by usage
        for cap in self.capsules.values():
            confidence = cap.get_confidence()
            
            # By usage
            if cap.usage_count == 0:
                summary["by_usage"]["never_used"].append(cap.name)
            elif cap.usage_count <= 5:
                summary["by_usage"]["rarely_used"].append(cap.name)
            else:
                summary["by_usage"]["frequently_used"].append(cap.name)
            
            # By status
            if confidence >= 0.8:
                summary["by_status"]["trusted"].append(cap.name)
            elif confidence >= 0.5:
                summary["by_status"]["stable"].append(cap.name)
            elif confidence >= 0.2:
                summary["by_status"]["experimental"].append(cap.name)
            else:
                summary["by_status"]["untested"].append(cap.name)
            
            # By type
            if cap.kind not in summary["by_type"]:
                summary["by_type"][cap.kind] = []
            summary["by_type"][cap.kind].append(cap.name)
        
        # Count frame references
        if keyframe_anchors:
            for frame, capsule_name in keyframe_anchors.items():
                if capsule_name not in summary["most_referenced_frames"]:
                    summary["most_referenced_frames"][capsule_name] = 0
                summary["most_referenced_frames"][capsule_name] += 1
        
        # Get top transitions
        top_transitions = sorted(
            self.coactivations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for (id1, id2), weight in top_transitions:
            cap1 = self.capsules.get(id1)
            cap2 = self.capsules.get(id2)
            if cap1 and cap2:
                summary["most_active_transitions"].append({
                    "pair": f"{cap1.name} ↔ {cap2.name}",
                    "weight": round(weight, 2),
                })
        
        return summary
    
    def freeze_capsule(self, capsule_id: str, reason: Optional[str] = None) -> bool:
        """Freeze a capsule to lock it as perfect.
        
        Once frozen, ROCA stops touching it unless explicitly told to.
        Perfect for walk cycles, signature poses, and acting beats.
        
        Args:
            capsule_id: ID of capsule to freeze
            reason: Why it's frozen: "walk_cycle", "signature_pose", "acting_beat", etc.
        
        Returns:
            True if successful, False if capsule not found
        """
        cap = self.capsules.get(capsule_id)
        if not cap:
            return False
        cap.freeze(reason)
        return True
    
    def unfreeze_capsule(self, capsule_id: str) -> bool:
        """Unfreeze a capsule to allow modifications again.
        
        Args:
            capsule_id: ID of capsule to unfreeze
        
        Returns:
            True if successful, False if capsule not found
        """
        cap = self.capsules.get(capsule_id)
        if not cap:
            return False
        cap.unfreeze()
        return True
    
    def get_frozen_capsules(self) -> list[dict]:
        """Get all frozen capsules with their freeze reasons.
        
        Returns:
            List of dicts with capsule info and freeze reason
        """
        frozen = []
        for cap in self.capsules.values():
            if cap.is_frozen:
                frozen.append({
                    "name": cap.name,
                    "kind": cap.kind,
                    "id": cap.id,
                    "reason": cap.freeze_reason or "locked",
                })
        return sorted(frozen, key=lambda x: x["name"])
    
    def can_modify_capsule(self, capsule_id: str) -> bool:
        """Check if a capsule can be modified.
        
        Returns False if capsule is frozen.
        
        Args:
            capsule_id: ID of capsule to check
        
        Returns:
            True if can be modified, False if frozen or not found
        """
        cap = self.capsules.get(capsule_id)
        if not cap:
            return False
        return not cap.is_frozen

    # ========================
    # Auto-Generation: In-Betweens
    # ========================
    
    def generate_inbetweens(
        self,
        capsule_ids: list[str],
        num_inbetweens: int = 1,
        auto_add: bool = True,
    ) -> list[Capsule]:
        """Auto-generate in-between capsules between poses/transitions.
        
        Generates intermediate capsules filling the gap between two adjacent capsules
        on the timeline. Uses symbolic pose interpolation to smoothly transition.
        
        Args:
            capsule_ids: List of capsule IDs in sequence order
            num_inbetweens: How many in-betweens to generate between each pair (default 1)
            auto_add: If True, add generated capsules to system (default True)
        
        Returns:
            List of newly created in-between capsules
        
        Example:
            # Create a walk cycle with auto-filled in-betweens
            sequence = [start_id, mid_id, end_id]
            inbetweens = system.generate_inbetweens(sequence, num_inbetweens=2)
            # Automatically creates smooth transition between each pair
        """
        new_capsules = []
        
        for i in range(len(capsule_ids) - 1):
            cap_a = self.capsules.get(capsule_ids[i])
            cap_b = self.capsules.get(capsule_ids[i + 1])
            
            if not cap_a or not cap_b:
                continue
            
            # Generate in-betweens between frame end_a+1 and frame start_b-1
            gap_start = cap_a.end_frame + 1
            gap_end = cap_b.start_frame - 1
            gap_size = gap_end - gap_start + 1
            
            if gap_size <= 0:
                continue  # No gap to fill
            
            # Clamp number of inbetweens to available gap
            actual_inbetweens = min(num_inbetweens, max(1, gap_size))
            
            # Generate inbetween poses via interpolation
            for j in range(1, actual_inbetweens + 1):
                # Interpolation factor (0.0 = cap_a, 1.0 = cap_b)
                t = j / (actual_inbetweens + 1)
                
                # Interpolated pose vector
                pose_a = cap_a.pose_vector or []
                pose_b = cap_b.pose_vector or []
                
                if pose_a and pose_b:
                    interpolated_pose = [
                        (1.0 - t) * pa + t * pb
                        for pa, pb in zip(pose_a, pose_b)
                    ]
                else:
                    interpolated_pose = pose_a or pose_b
                
                # Frame number for this inbetween
                frame_offset = int((j / (actual_inbetweens + 1)) * gap_size)
                inb_frame = gap_start + frame_offset
                
                # Create inbetween capsule
                inb_name = f"in_between_{cap_a.name}_to_{cap_b.name}_#{j}"
                inb = Capsule(
                    name=inb_name,
                    kind="Memory",  # In-betweens are generated, not keyframes
                    pose_vector=interpolated_pose,
                    shadows=[cap_a.id, cap_b.id],  # Track source capsules
                    level="atomic",
                    start_frame=inb_frame,
                    end_frame=inb_frame,
                    color=(180, 180, 180),  # Gray for generated in-betweens
                    tags=["auto_inbetween"],
                )
                
                if auto_add:
                    self.capsules[inb.id] = inb
                    # Record as derived from both capsules
                    self.reinforce_coactivation(cap_a.id, inb.id, weight=0.5)
                    self.reinforce_coactivation(inb.id, cap_b.id, weight=0.5)
                
                new_capsules.append(inb)
        
        return new_capsules
    
    def generate_inbetweens_for_gap(
        self,
        capsule_before_id: str,
        capsule_after_id: str,
        num_inbetweens: int = 2,
        auto_add: bool = True,
    ) -> list[Capsule]:
        """Generate in-betweens to fill gap between two specific capsules.
        
        Simpler version for filling a single gap.
        
        Args:
            capsule_before_id: ID of capsule before the gap
            capsule_after_id: ID of capsule after the gap
            num_inbetweens: How many in-betweens to generate (default 2)
            auto_add: If True, add to system (default True)
        
        Returns:
            List of generated in-between capsules
        """
        return self.generate_inbetweens(
            [capsule_before_id, capsule_after_id],
            num_inbetweens=num_inbetweens,
            auto_add=auto_add
        )
    
    def auto_fill_timeline(
        self,
        total_frames: int,
        inbetweens_per_gap: int = 2,
    ) -> list[Capsule]:
        """Auto-fill entire timeline with in-betweens where gaps exist.
        
        Scans all capsules on timeline and generates in-betweens to smooth transitions.
        
        Args:
            total_frames: Total frames in animation
            inbetweens_per_gap: How many in-betweens per gap (default 2)
        
        Returns:
            List of all generated in-between capsules
        
        Example:
            # Create 3 keyframes, auto-fill the gaps
            keyframes = [start_id, mid_id, end_id]
            inbetweens = system.auto_fill_timeline(total_frames=100, inbetweens_per_gap=3)
        """
        # Sort capsules by start_frame
        sorted_caps = sorted(self.capsules.values(), key=lambda c: c.start_frame)
        
        # Build list of capsule IDs in order
        capsule_ids = [c.id for c in sorted_caps]
        
        if len(capsule_ids) < 2:
            return []
        
        return self.generate_inbetweens(
            capsule_ids,
            num_inbetweens=inbetweens_per_gap,
            auto_add=True
        )
    
    def get_inbetween_capsules(self) -> list[Capsule]:
        """Get all auto-generated in-between capsules.
        
        Returns:
            List of capsules with "auto_inbetween" tag
        """
        return [c for c in self.capsules.values() if "auto_inbetween" in c.tags]
    
    def clear_inbetweens(self) -> int:
        """Remove all auto-generated in-between capsules.
        
        Useful for re-generating with different parameters.
        
        Returns:
            Number of capsules removed
        """
        inb_ids = [c.id for c in self.get_inbetween_capsules()]
        for iid in inb_ids:
            self.capsules.pop(iid, None)
        return len(inb_ids)

    @staticmethod
    def from_dict(data: dict, center: tuple[int, int], radius_scale: float) -> "ROCASystem":
        sys_obj = ROCASystem(center, radius_scale)
        for cdata in data.get("capsules", []):
            cap = Capsule.from_dict(cdata)
            sys_obj._assign_radius(cap)
            sys_obj.capsules[cap.id] = cap
        for entry in data.get("coactivations", []):
            if not isinstance(entry, list) or len(entry) != 3:
                continue
            key = (str(entry[0]), str(entry[1]))
            sys_obj.coactivations[key] = float(entry[2])
        th = data.get("thresholds", {})
        try:
            sys_obj.atomic_to_sequence = int(th.get("atomic_to_sequence", sys_obj.atomic_to_sequence))
            sys_obj.sequence_to_cycle = int(th.get("sequence_to_cycle", sys_obj.sequence_to_cycle))
        except Exception:
            pass
        return sys_obj


def _build_rings_for_rect(rect: pygame.Rect) -> list[OrbitRing]:
    min_dim = max(1, min(rect.width, rect.height))
    rings: list[OrbitRing] = []
    for i, band in enumerate(sorted(KIND_BANDS.values())):
        inner, outer = band
        r = int((min_dim * 0.5) * float(outer))
        col = RING_COLORS[min(i, len(RING_COLORS) - 1)]
        rings.append(OrbitRing(r, col))
    return rings

# U-Net Model Definition
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(6, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(512 + 256, 256)
        self.dec3 = DoubleConv(256 + 128, 128)  # Skip connection from enc3
        self.dec2 = DoubleConv(128 + 64, 64)    # Skip connection from enc2
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder with skip connections
        d4 = self.upsample(e4)
        d3 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.upsample(d3)
        d2 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.upsample(d2)
        d1 = self.dec2(torch.cat([d2, e1], dim=1))

        return torch.sigmoid(self.final(d1))

# Modified Animation Dataset
class AnimationDataset(Dataset):
    def __init__(self, frame_pairs, transform=None):
        self.frame_pairs = frame_pairs  # List of (start_frame, end_frame) tuples
        self.transform = transform
        
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        start_frame, end_frame = self.frame_pairs[idx]
        
        # Convert pygame surfaces to PIL Images
        start_array = pygame.surfarray.array3d(start_frame).transpose(1, 0, 2)
        start_pil = Image.fromarray(start_array)
        
        end_array = pygame.surfarray.array3d(end_frame).transpose(1, 0, 2)
        end_pil = Image.fromarray(end_array)
        
        if self.transform:
            start_tensor = self.transform(start_pil)
            end_tensor = self.transform(end_pil)
            
        # Stack as input-target pair (6 channels input, 3 channels target)
        return torch.cat([start_tensor, end_tensor], dim=0), end_tensor

# Modified Drawing State with training controls
class DrawingState:
    def __init__(self):
        self.drawing = False
        self.last_pos = None
        self.brush_color = BLACK
        self.brush_size = 5
        self.layers = [pygame.Surface((800, 600), pygame.SRCALPHA)]
        self.current_layer = 0
        self.drawing_history = []
        self.animation_frames = [pygame.Surface((800, 600), pygame.SRCALPHA)]  # Initial empty frame
        self.current_frame = 0  # Track current animation frame index
        self.playback = False
        self.playback_frame = 0
        self.playback_speed = 30
        self.keyframes = []
        self.onion_skinning = False
        self.ai_assist = False
        self.undo_stack = []
        self.redo_stack = []
        self.training_progress = 0.0
        self.is_training = False
        self.imported_images = []
        self.last_save_time = 0  # For save cooldown
        self.last_frame_time = 0  # For playback timing
        self.svg_objects = []  # For SVG drawing
        self.video_frames = []  # For video frames
        self.previous_frame = None  # Track the previous frame for timeline
        self.frame_changed = False  # Track if frame needs saving
        self.mode = "draw"  # draw | capsule

    def save_current_frame(self):
        """Save the current layers to animation frames"""
        if self.current_frame < len(self.animation_frames):
            # Combine all layers into one surface
            combined = pygame.Surface((800, 600), pygame.SRCALPHA)
            for layer in self.layers:
                combined.blit(layer, (0, 0))
            self.animation_frames[self.current_frame] = combined.copy()
            self.frame_changed = False
            
    def switch_to_frame(self, frame_index):
        """Switch to a different frame, saving current work"""
        if self.frame_changed:
            self.save_current_frame()
        
        self.current_frame = frame_index
        self.layers = [self.animation_frames[frame_index].copy()]
        self.frame_changed = False


# ========================
# Live In-Between Updates
# ========================
def find_previous_pose(capsules: list[Capsule], target_capsule: Capsule) -> Optional[Capsule]:
    """Find the closest pose capsule before target in timeline."""
    candidates = [
        c for c in capsules 
        if not c.is_inbetween and c.end_frame < target_capsule.start_frame
    ]
    return max(candidates, key=lambda c: c.end_frame) if candidates else None


def find_next_pose(capsules: list[Capsule], target_capsule: Capsule) -> Optional[Capsule]:
    """Find the closest pose capsule after target in timeline."""
    candidates = [
        c for c in capsules 
        if not c.is_inbetween and c.start_frame > target_capsule.end_frame
    ]
    return min(candidates, key=lambda c: c.start_frame) if candidates else None


def update_inbetween_from_tweak(capsules: list[Capsule], edited_capsule: Capsule) -> None:
    """When a pose capsule is modified, update all related in-betweens.
    
    ROCA keeps symbolic heuristics and learns from the tweak:
    - Updates reference pose IDs for affected in-betweens
    - Regenerates interpolation ratio based on new frame positions
    - Preserves in-between tags and scope context
    
    Args:
        capsules: List of all capsules in timeline
        edited_capsule: The pose capsule that was just edited
    """
    if edited_capsule.is_inbetween:
        return  # Only update from pose edits
    
    for inbetween in capsules:
        if not inbetween.is_inbetween:
            continue
        
        # Check if inbetween is between the edited capsule and neighbors
        prev_pose = find_previous_pose(capsules, inbetween)
        next_pose = find_next_pose(capsules, inbetween)
        
        needs_update = False
        
        # If edited capsule is the previous pose, update start reference
        if prev_pose and prev_pose.id == edited_capsule.id:
            inbetween.ref_start_id = edited_capsule.id
            needs_update = True
        
        # If edited capsule is the next pose, update end reference
        if next_pose and next_pose.id == edited_capsule.id:
            inbetween.ref_end_id = edited_capsule.id
            needs_update = True
        
        if needs_update and inbetween.ref_start_id and inbetween.ref_end_id:
            # Recalculate interpolation ratio based on new frame positions
            start_pose = next((c for c in capsules if c.id == inbetween.ref_start_id), None)
            end_pose = next((c for c in capsules if c.id == inbetween.ref_end_id), None)
            
            if start_pose and end_pose:
                # Ratio: where inbetween is positioned between start and end
                gap = end_pose.start_frame - start_pose.end_frame
                if gap > 0:
                    offset = inbetween.start_frame - start_pose.end_frame
                    inbetween.interpolation_ratio = min(1.0, max(0.0, offset / gap))
                
                # Regenerate pose via interpolation (symbolic blend)
                interpolate_pose(
                    start_pose.pose_vector,
                    end_pose.pose_vector,
                    inbetween.pose_vector,
                    inbetween.interpolation_ratio
                )
                
                # Update modification time
                inbetween.touch_modification()


def interpolate_pose(
    start_vector: list[float],
    end_vector: list[float],
    result_vector: list[float],
    t: float
) -> None:
    """Linear interpolation between two pose vectors.
    
    Args:
        start_vector: Starting pose values
        end_vector: Ending pose values
        result_vector: Vector to populate with interpolated values (modified in-place)
        t: Interpolation parameter (0.0 = start, 1.0 = end)
    """
    # Ensure result vector is same length
    while len(result_vector) < len(start_vector):
        result_vector.append(0.0)
    
    for i in range(min(len(start_vector), len(end_vector))):
        result_vector[i] = start_vector[i] * (1.0 - t) + end_vector[i] * t


def c_progress_ratio(capsule: Capsule, prev_capsule: Optional[Capsule] = None, next_capsule: Optional[Capsule] = None) -> float:
    """Calculate normalized progress ratio (0-1) of an in-between between its neighboring poses.
    
    This determines where the in-between sits interpolation-wise between the start and end poses.
    
    Args:
        capsule: The in-between capsule
        prev_capsule: Optional previous pose (if None, uses ref_start_id)
        next_capsule: Optional next pose (if None, uses ref_end_id)
    
    Returns:
        float: Progress ratio from 0.0 (at start pose) to 1.0 (at end pose)
    """
    if prev_capsule and next_capsule:
        gap = next_capsule.start_frame - prev_capsule.end_frame
        if gap > 0:
            offset = capsule.start_frame - prev_capsule.end_frame
            return min(1.0, max(0.0, offset / gap))
    
    return capsule.interpolation_ratio


def generate_capsule_image(
    start_capsule: Capsule,
    end_capsule: Capsule,
    ratio: float
) -> tuple[int, int, int]:
    """Generate visual representation of in-between via symbolic interpolation.
    
    ROCA symbolic interpolation: Blends capsule properties (color, pose) based on ratio.
    In a full implementation, this would interpolate pose vectors and generate animation frames.
    For now, we blend colors for visual feedback.
    
    Args:
        start_capsule: Starting pose capsule
        end_capsule: Ending pose capsule
        ratio: Interpolation ratio (0.0 to 1.0)
    
    Returns:
        tuple: Blended RGB color
    """
    start_color = start_capsule.color
    end_color = end_capsule.color
    
    # Blend colors linearly
    blended = tuple(
        int(start_color[i] * (1.0 - ratio) + end_color[i] * ratio)
        for i in range(min(3, len(start_color), len(end_color)))
    )
    
    return blended  # type: ignore


# Animator workflow demo removed for experimental build.





# ========================
# Timeline Track Renderer
# ========================
class TimelineRenderer:
    """Renders capsules as tracks on an interactive timeline.
    
    Features:
    - Frame-based positioning with pixel scaling
    - Color-coded capsules (RGB from capsule.color)
    - Confidence-based coloring (optional override)
    - Frozen capsule visual indicators
    - Interactive hover and click detection
    - Snap-to-symbolic frame snapping
    - Frame ruler and timing markers
    """
    
    # UI Constants
    TRACK_HEIGHT = 40              # Height of each capsule track
    TRACK_PADDING = 10             # Vertical space between tracks
    CAPSULE_HEIGHT = 30            # Height of capsule rectangle
    RULER_HEIGHT = 30              # Height of timeline ruler at top
    RULER_PADDING = 5              # Space between ruler and tracks
    FRAME_LABEL_INTERVAL = 10      # Draw frame numbers every N frames
    MIN_CAPSULE_WIDTH = 20         # Minimum visual width for a capsule
    
    # Colors
    COLOR_RULER_BG = (245, 245, 245)
    COLOR_RULER_TEXT = (50, 50, 50)
    COLOR_RULER_LINE = (200, 200, 200)
    COLOR_PLAYHEAD = (255, 0, 0)
    COLOR_BORDER = (100, 100, 100)
    COLOR_FROZEN_OVERLAY = (100, 100, 255, 80)  # Semi-transparent blue
    COLOR_HOVER_HIGHLIGHT = (255, 255, 200)
    COLOR_CONFIDENCE_LOW = (255, 100, 100)      # Red (experimental)
    COLOR_CONFIDENCE_MID = (255, 200, 100)      # Orange (stable)
    COLOR_CONFIDENCE_HIGH = (100, 255, 100)     # Green (trusted)
    
    def __init__(self, screen_rect: pygame.Rect, pixels_per_frame: float = 10.0):
        """Initialize timeline renderer.
        
        Args:
            screen_rect: Pygame rect defining the timeline area
            pixels_per_frame: Pixels per frame (controls zoom level)
        """
        self.screen_rect = screen_rect
        self.pixels_per_frame = pixels_per_frame
        self.hovered_capsule_id: Optional[str] = None
        self.selected_capsule_id: Optional[str] = None
        self.scroll_offset = 0  # Horizontal scroll for long timelines
        self.current_frame = 0
        
        # Font for labels and text
        self.font_small = pygame.font.SysFont('Arial', 10)
        self.font_normal = pygame.font.SysFont('Arial', 12)
        self.font_bold = pygame.font.SysFont('Arial', 12, bold=True)
    
    def set_zoom(self, pixels_per_frame: float) -> None:
        """Change zoom level.
        
        Args:
            pixels_per_frame: New zoom factor (default 10.0)
        """
        self.pixels_per_frame = max(1.0, min(50.0, pixels_per_frame))
    
    def set_scroll(self, offset: int) -> None:
        """Horizontal scroll for timelines longer than screen."""
        self.scroll_offset = offset
    
    def set_playhead(self, frame: int) -> None:
        """Set current playhead position."""
        self.current_frame = frame
    
    def draw_ruler(self, screen: pygame.Surface, total_frames: int) -> None:
        """Draw timeline ruler with frame numbers and markers.
        
        Args:
            screen: Pygame surface to draw on
            total_frames: Total number of frames in animation
        """
        ruler_rect = pygame.Rect(
            self.screen_rect.x,
            self.screen_rect.y,
            self.screen_rect.width,
            self.RULER_HEIGHT
        )
        
        # Background
        pygame.draw.rect(screen, self.COLOR_RULER_BG, ruler_rect)
        pygame.draw.line(
            screen,
            self.COLOR_RULER_LINE,
            (ruler_rect.x, ruler_rect.bottom),
            (ruler_rect.right, ruler_rect.bottom),
            1
        )
        
        # Draw frame markers
        for frame in range(0, total_frames, self.FRAME_LABEL_INTERVAL):
            x = self.screen_rect.x + frame * self.pixels_per_frame - self.scroll_offset
            
            if ruler_rect.x <= x <= ruler_rect.right:
                # Small tick mark
                pygame.draw.line(
                    screen,
                    self.COLOR_RULER_LINE,
                    (x, ruler_rect.bottom - 5),
                    (x, ruler_rect.bottom),
                    1
                )
                
                # Frame number label
                text = self.font_small.render(str(frame), True, self.COLOR_RULER_TEXT)
                text_rect = text.get_rect(center=(x, ruler_rect.centery))
                screen.blit(text, text_rect)
        
        # Draw playhead
        playhead_x = self.screen_rect.x + self.current_frame * self.pixels_per_frame - self.scroll_offset
        if ruler_rect.x <= playhead_x <= ruler_rect.right:
            pygame.draw.line(
                screen,
                self.COLOR_PLAYHEAD,
                (playhead_x, ruler_rect.y),
                (playhead_x, ruler_rect.bottom),
                2
            )
    
    def draw_tracks(
        self,
        screen: pygame.Surface,
        capsules: list[Capsule],
        use_confidence_coloring: bool = False,
    ) -> None:
        """Draw capsule tracks.
        
        Args:
            screen: Pygame surface to draw on
            capsules: List of Capsule objects to render
            use_confidence_coloring: If True, color capsules by confidence instead of capsule.color
        """
        tracks_rect = pygame.Rect(
            self.screen_rect.x,
            self.screen_rect.y + self.RULER_HEIGHT + self.RULER_PADDING,
            self.screen_rect.width,
            self.screen_rect.height - self.RULER_HEIGHT - self.RULER_PADDING,
        )
        
        # Background
        pygame.draw.rect(screen, (255, 255, 255), tracks_rect)
        
        for idx, capsule in enumerate(capsules):
            track_y = tracks_rect.y + idx * (self.TRACK_HEIGHT + self.TRACK_PADDING)
            
            if track_y + self.TRACK_HEIGHT > tracks_rect.bottom:
                break  # Off-screen
            
            # Capsule position and size
            x_start = self.screen_rect.x + capsule.start_frame * self.pixels_per_frame - self.scroll_offset
            x_end = self.screen_rect.x + capsule.end_frame * self.pixels_per_frame - self.scroll_offset
            width = max(self.MIN_CAPSULE_WIDTH, x_end - x_start)
            
            # Determine color
            if use_confidence_coloring:
                confidence = capsule.get_confidence()
                if confidence < 0.3:
                    color = self.COLOR_CONFIDENCE_LOW
                elif confidence < 0.7:
                    color = self.COLOR_CONFIDENCE_MID
                else:
                    color = self.COLOR_CONFIDENCE_HIGH
            else:
                # For in-betweens, use dynamically generated interpolated color
                # For poses, use default color
                color = capsule.color
            
            # Hover highlight
            if capsule.id == self.hovered_capsule_id:
                color = self.COLOR_HOVER_HIGHLIGHT
            
            # Draw capsule rectangle
            capsule_rect = pygame.Rect(x_start, track_y, width, self.CAPSULE_HEIGHT)
            pygame.draw.rect(screen, color, capsule_rect)
            
            # Border (thicker for selected/frozen)
            border_width = 3 if capsule.id == self.selected_capsule_id else 2 if capsule.is_frozen else 1
            border_color = (255, 0, 0) if capsule.is_frozen else self.COLOR_BORDER
            pygame.draw.rect(screen, border_color, capsule_rect, border_width)
            
            # Frozen indicator
            if capsule.is_frozen:
                lock_text = self.font_small.render("🔒", True, (100, 100, 255))
                screen.blit(lock_text, (x_start + 3, track_y + 3))
            
            # In-between indicator
            if capsule.is_inbetween:
                inbetween_text = self.font_small.render("⚡", True, (255, 200, 0))
                screen.blit(inbetween_text, (x_start + 3, track_y + 3))
            
            # Capsule name label
            text_surf = self.font_bold.render(capsule.name, True, (0, 0, 0))
            text_rect = text_surf.get_rect(
                midleft=(x_start + 8, track_y + self.CAPSULE_HEIGHT // 2)
            )
            if text_rect.right < capsule_rect.right:
                screen.blit(text_surf, text_rect)
            
            # Interpolation ratio for in-betweens (live feedback)
            if capsule.is_inbetween:
                ratio_text = self.font_small.render(
                    f"{capsule.interpolation_ratio:.0%}",
                    True,
                    (50, 50, 50)
                )
                screen.blit(ratio_text, (x_end - 30, track_y + 5))
            
            # Confidence indicator (small badge)
            if use_confidence_coloring:
                confidence = capsule.get_confidence()
                conf_text = self.font_small.render(f"{confidence:.1f}", True, (255, 255, 255))
                conf_badge = pygame.Rect(x_end - 25, track_y + 2, 22, 15)
                pygame.draw.rect(screen, (50, 50, 50), conf_badge)
                conf_text_rect = conf_text.get_rect(center=conf_badge.center)
                screen.blit(conf_text, conf_text_rect)
            
            # Scope indicator
            if capsule.scope != "universal":
                scope_text = f"[{capsule.scope_context or capsule.scope}]"
                scope_surf = self.font_small.render(scope_text, True, (100, 100, 100))
                screen.blit(scope_surf, (x_start + 5, track_y + self.CAPSULE_HEIGHT - 12))
    
    def draw_playhead(self, screen: pygame.Surface) -> None:
        """Draw playhead line at current frame."""
        playhead_x = self.screen_rect.x + self.current_frame * self.pixels_per_frame - self.scroll_offset
        
        if self.screen_rect.x <= playhead_x <= self.screen_rect.right:
            pygame.draw.line(
                screen,
                self.COLOR_PLAYHEAD,
                (playhead_x, self.screen_rect.y + self.RULER_HEIGHT),
                (playhead_x, self.screen_rect.bottom),
                3
            )
    
    def get_capsule_at_position(
        self,
        capsules: list[Capsule],
        pos: tuple[int, int],
    ) -> Optional[Capsule]:
        """Get capsule at screen position (for click detection).
        
        Args:
            capsules: List of capsules to check
            pos: (x, y) screen coordinates
        
        Returns:
            Capsule at position, or None
        """
        x, y = pos
        tracks_y = self.screen_rect.y + self.RULER_HEIGHT + self.RULER_PADDING
        
        # Which track row?
        if y < tracks_y:
            return None
        
        track_index = (y - tracks_y) // (self.TRACK_HEIGHT + self.TRACK_PADDING)
        
        if track_index < 0 or track_index >= len(capsules):
            return None
        
        capsule = capsules[track_index]
        track_y = tracks_y + track_index * (self.TRACK_HEIGHT + self.TRACK_PADDING)
        
        # Which capsule in the row?
        x_start = self.screen_rect.x + capsule.start_frame * self.pixels_per_frame - self.scroll_offset
        x_end = self.screen_rect.x + capsule.end_frame * self.pixels_per_frame - self.scroll_offset
        
        if x_start <= x <= x_end and track_y <= y <= track_y + self.CAPSULE_HEIGHT:
            return capsule
        
        return None
    
    def get_frame_at_position(self, x: int) -> int:
        """Convert screen X coordinate to frame number.
        
        Args:
            x: Screen X coordinate
        
        Returns:
            Frame number (snapped to symbolic boundary)
        """
        if self.pixels_per_frame <= 0:
            return 0
        
        frame = int((x - self.screen_rect.x + self.scroll_offset) / self.pixels_per_frame)
        return snap_to_symbolic(frame, snap_interval=5)
    
    def draw(
        self,
        screen: pygame.Surface,
        capsules: list[Capsule],
        total_frames: int,
        use_confidence_coloring: bool = False,
    ) -> None:
        """Draw complete timeline (ruler + tracks + playhead).
        
        Args:
            screen: Pygame surface to draw on
            capsules: List of capsules to render
            total_frames: Total frames in animation
            use_confidence_coloring: If True, color by confidence instead of capsule.color
        """
        self.draw_ruler(screen, total_frames)
        self.draw_tracks(screen, capsules, use_confidence_coloring)
        self.draw_playhead(screen)
    
    def on_hover(self, capsules: list[Capsule], pos: tuple[int, int]) -> None:
        """Update hover state.
        
        Args:
            capsules: List of capsules
            pos: Current mouse position
        """
        self.hovered_capsule_id = self.get_capsule_at_position(capsules, pos).id \
            if self.get_capsule_at_position(capsules, pos) else None
    
    def on_click(self, capsules: list[Capsule], pos: tuple[int, int]) -> Optional[Capsule]:
        """Handle click on timeline.
        
        Args:
            capsules: List of capsules
            pos: Click position
        
        Returns:
            Selected capsule, or None
        """
        capsule = self.get_capsule_at_position(capsules, pos)
        if capsule:
            self.selected_capsule_id = capsule.id
            return capsule
        return None


# ========================
# Timeline Interaction Handler
# ========================
class TimelineInteractionHandler:
    """Handles mouse interactions for timeline capsule manipulation.
    
    Features:
    - Click detection and selection
    - Drag capsules to new timeline positions
    - Resize capsule boundaries (start/end frames)
    - Frame snapping during drag operations
    - Frozen capsule protection (can't drag if frozen)
    - Undo/redo support for drag operations
    - Live in-between updates on pose edits
    """
    
    # Resize zone width (pixels from edge to trigger resize mode)
    RESIZE_HANDLE_WIDTH = 8
    
    def __init__(self, timeline_renderer: TimelineRenderer, capsules: Optional[list[Capsule]] = None):
        """Initialize interaction handler.
        
        Args:
            timeline_renderer: TimelineRenderer instance for coordinate conversion
            capsules: List of all capsules for live in-between updates
        """
        self.renderer = timeline_renderer
        self.capsules = capsules or []  # For live in-between updates
        
        # Current drag state
        self.is_dragging = False
        self.drag_mode = None  # "move", "resize_start", "resize_end"
        self.dragged_capsule: Optional[Capsule] = None
        self.drag_start_pos = (0, 0)
        self.drag_offset_frame = 0
        
        # History for undo/redo
        self.drag_history: list[dict] = []
        self.max_history_size = 50
        
        # Cursor hint
        self.cursor_mode = "normal"  # "normal", "move", "resize"
    
    def get_drag_mode_at_position(
        self,
        capsule: Capsule,
        pos: tuple[int, int],
    ) -> Optional[str]:
        """Determine if clicking near capsule edge (for resize) or interior (for move).
        
        Args:
            capsule: The capsule to check
            pos: Screen position (x, y)
        
        Returns:
            "move" (center), "resize_start" (left edge), "resize_end" (right edge), or None
        """
        x, y = pos
        
        # Get capsule screen bounds
        x_start = self.renderer.screen_rect.x + capsule.start_frame * self.renderer.pixels_per_frame - self.renderer.scroll_offset
        x_end = self.renderer.screen_rect.x + capsule.end_frame * self.renderer.pixels_per_frame - self.renderer.scroll_offset
        
        # Check left edge (resize start)
        if x_start <= x <= x_start + self.RESIZE_HANDLE_WIDTH:
            return "resize_start"
        
        # Check right edge (resize end)
        if x_end - self.RESIZE_HANDLE_WIDTH <= x <= x_end:
            return "resize_end"
        
        # Check center (move)
        if x_start < x < x_end:
            return "move"
        
        return None
    
    def start_drag(
        self,
        capsule: Capsule,
        pos: tuple[int, int],
        mode: str,
    ) -> bool:
        """Begin a drag operation.
        
        Args:
            capsule: Capsule being dragged
            pos: Initial mouse position
            mode: "move", "resize_start", or "resize_end"
        
        Returns:
            True if drag started, False if blocked (e.g., frozen capsule)
        """
        if capsule.is_frozen:
            return False  # Can't drag frozen capsules
        
        self.is_dragging = True
        self.drag_mode = mode
        self.dragged_capsule = capsule
        self.drag_start_pos = pos
        
        # Save state for undo
        self.drag_history.append({
            "capsule_id": capsule.id,
            "start_frame": capsule.start_frame,
            "end_frame": capsule.end_frame,
            "mode": mode,
        })
        
        if len(self.drag_history) > self.max_history_size:
            self.drag_history.pop(0)
        
        return True
    
    def update_drag(self, pos: tuple[int, int]) -> None:
        """Update dragging capsule position/size.
        
        Args:
            pos: Current mouse position
        """
        if not self.is_dragging or not self.dragged_capsule:
            return
        
        current_x, current_y = pos
        start_x, start_y = self.drag_start_pos
        pixel_delta = current_x - start_x
        frame_delta = int(pixel_delta / self.renderer.pixels_per_frame)
        
        capsule = self.dragged_capsule
        
        if self.drag_mode == "move":
            # Move entire capsule
            new_start = capsule.start_frame + frame_delta
            new_end = capsule.end_frame + frame_delta
            
            # Snap to symbolic frames
            capsule.start_frame = snap_to_symbolic(new_start, snap_interval=5)
            capsule.end_frame = snap_to_symbolic(new_end, snap_interval=5)
            
            # Update drag offset for next frame
            self.drag_offset_frame = frame_delta
        
        elif self.drag_mode == "resize_start":
            # Resize left edge
            new_start = capsule.start_frame + frame_delta
            new_start = snap_to_symbolic(new_start, snap_interval=5)
            
            # Don't let start exceed end
            if new_start < capsule.end_frame:
                capsule.start_frame = new_start
        
        elif self.drag_mode == "resize_end":
            # Resize right edge
            new_end = capsule.end_frame + frame_delta
            new_end = snap_to_symbolic(new_end, snap_interval=5)
            
            # Don't let end go before start
            if new_end > capsule.start_frame:
                capsule.end_frame = new_end
        
        # Update modification timestamp
        capsule.touch_modification()
    
    def end_drag(self) -> Optional[Capsule]:
        """End drag operation and return the dragged capsule.
        
        Triggers live in-between updates if capsule is a pose.
        
        Returns:
            The dragged capsule, or None if no drag was active
        """
        if not self.is_dragging:
            return None
        
        capsule = self.dragged_capsule
        
        self.is_dragging = False
        self.drag_mode = None
        self.dragged_capsule = None
        self.drag_offset_frame = 0
        
        # Update in-betweens if dragged capsule is a pose
        if capsule and self.capsules and not capsule.is_inbetween:
            update_inbetween_from_tweak(self.capsules, capsule)
        
        return capsule
    
    def undo_last_drag(self) -> bool:
        """Undo the last drag operation.
        
        Returns:
            True if undo was performed, False if no history
        """
        if not self.drag_history:
            return False
        
        state = self.drag_history.pop()
        
        # Would need access to ROCASystem to find capsule by ID
        # For now, just remove from history
        return True
    
    def handle_mouse_move(
        self,
        capsules: list[Capsule],
        pos: tuple[int, int],
    ) -> str:
        """Update cursor mode based on mouse position over timeline.
        
        Args:
            capsules: List of capsules
            pos: Current mouse position
        
        Returns:
            Cursor mode: "normal", "move", "resize"
        """
        if self.is_dragging:
            self.update_drag(pos)
            return self.cursor_mode
        
        # Check what's under the cursor
        capsule = self.renderer.get_capsule_at_position(capsules, pos)
        
        if capsule:
            mode = self.get_drag_mode_at_position(capsule, pos)
            if mode == "resize_start" or mode == "resize_end":
                self.cursor_mode = "resize"
            else:
                self.cursor_mode = "move"
        else:
            self.cursor_mode = "normal"
        
        return self.cursor_mode
    
    def handle_mouse_down(
        self,
        capsules: list[Capsule],
        pos: tuple[int, int],
    ) -> Optional[Capsule]:
        """Handle mouse button down on timeline.
        
        Args:
            capsules: List of capsules
            pos: Click position
        
        Returns:
            Capsule being dragged, or None
        """
        capsule = self.renderer.get_capsule_at_position(capsules, pos)
        
        if not capsule:
            return None
        
        # Determine drag mode
        mode = self.get_drag_mode_at_position(capsule, pos)
        
        if not mode:
            return None
        
        # Start drag if not frozen
        if self.start_drag(capsule, pos, mode):
            return capsule
        
        return None
    
    def handle_mouse_up(self) -> Optional[Capsule]:
        """Handle mouse button up (end drag).
        
        Returns:
            Capsule that was dragged, or None
        """
        return self.end_drag()
    
    def handle_playhead_click(self, pos: tuple[int, int]) -> int:
        """Handle click on timeline to move playhead.
        
        Args:
            pos: Click position
        
        Returns:
            New frame number
        """
        frame = self.renderer.get_frame_at_position(pos[0])
        self.renderer.set_playhead(frame)
        return frame
    
    def get_cursor_pygame(self) -> int:
        """Get Pygame cursor constant based on current mode.
        
        Returns:
            pygame.SYSTEM_CURSOR_* constant
        """
        if self.cursor_mode == "resize":
            return pygame.SYSTEM_CURSOR_SIZEWE  # Left-right resize
        elif self.cursor_mode == "move":
            return pygame.SYSTEM_CURSOR_SIZEALL  # Move all directions
        else:
            return pygame.SYSTEM_CURSOR_ARROW


class CapsuleInputDialog:
    """Dialog for creating new capsules with name and kind selection."""
    
    def __init__(self):
        self.active = False
        self.name_input = ""
        self.kind_selected = "Memory"  # Default kind
        self.kinds = ["Core", "Character", "Style", "Skill", "Topic", "Memory", "Experimental"]
        self.rect = pygame.Rect(400, 200, 400, 300)
        self.name_input_rect = pygame.Rect(self.rect.x + 20, self.rect.y + 80, 360, 35)
        self.kind_dropdown_rect = pygame.Rect(self.rect.x + 20, self.rect.y + 140, 360, 35)
        self.ok_button_rect = pygame.Rect(self.rect.x + 80, self.rect.y + 240, 80, 40)
        self.cancel_button_rect = pygame.Rect(self.rect.x + 240, self.rect.y + 240, 80, 40)
        self.font = pygame.font.SysFont("Arial", 14)
        self.title_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.result = None
    
    def open(self):
        """Open the dialog for capsule creation."""
        self.active = True
        self.name_input = ""
        self.kind_selected = "Memory"
        self.result = None
    
    def close(self):
        """Close the dialog."""
        self.active = False
    
    def handle_input(self, event: pygame.event.Event) -> None:
        """Handle keyboard and mouse input."""
        if not self.active:
            return
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Name input field
            if self.name_input_rect.collidepoint(event.pos):
                pass  # Name field focused
            # Kind dropdown
            elif self.kind_dropdown_rect.collidepoint(event.pos):
                # Simple cycling through kinds
                current_idx = self.kinds.index(self.kind_selected)
                self.kind_selected = self.kinds[(current_idx + 1) % len(self.kinds)]
            # OK button
            elif self.ok_button_rect.collidepoint(event.pos):
                if self.name_input.strip():
                    self.result = (self.name_input.strip(), self.kind_selected)
                    self.active = False
            # Cancel button
            elif self.cancel_button_rect.collidepoint(event.pos):
                self.active = False
        
        elif event.type == pygame.KEYDOWN:
            if self.name_input_rect.collidepoint(pygame.mouse.get_pos()):
                if event.key == pygame.K_RETURN:
                    if self.name_input.strip():
                        self.result = (self.name_input.strip(), self.kind_selected)
                        self.active = False
                elif event.key == pygame.K_BACKSPACE:
                    self.name_input = self.name_input[:-1]
        
        elif event.type == pygame.TEXTINPUT:
            if self.name_input_rect.collidepoint(pygame.mouse.get_pos()):
                if len(self.name_input) < 30:  # Max 30 chars
                    self.name_input += event.text
    
    def draw(self, screen: pygame.Surface) -> None:
        """Render the dialog."""
        if not self.active:
            return
        
        # Semi-transparent overlay
        overlay = pygame.Surface((screen.get_width(), screen.get_height()))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Dialog box
        pygame.draw.rect(screen, (40, 50, 70), self.rect)
        pygame.draw.rect(screen, (100, 150, 200), self.rect, 3)
        
        # Title
        title = self.title_font.render("Create New Capsule", True, (100, 200, 255))
        screen.blit(title, (self.rect.x + 20, self.rect.y + 10))
        
        # Help text
        help_text = self.font.render("Atomic: single concept | Composite: merged concepts", True, (150, 180, 220))
        screen.blit(help_text, (self.rect.x + 20, self.rect.y + 40))
        
        # Name label and input
        name_label = self.font.render("Name (e.g., 'ArmRaise', 'CartoonEyes'):", True, (200, 200, 200))
        screen.blit(name_label, (self.rect.x + 20, self.rect.y + 65))
        
        pygame.draw.rect(screen, (60, 70, 90), self.name_input_rect)
        pygame.draw.rect(screen, (100, 150, 200), self.name_input_rect, 2)
        name_surf = self.font.render(self.name_input, True, (220, 220, 220))
        screen.blit(name_surf, (self.name_input_rect.x + 8, self.name_input_rect.y + 8))
        
        # Kind label and dropdown
        kind_label = self.font.render("Kind (click to cycle):", True, (200, 200, 200))
        screen.blit(kind_label, (self.rect.x + 20, self.rect.y + 125))
        
        pygame.draw.rect(screen, (60, 70, 90), self.kind_dropdown_rect)
        pygame.draw.rect(screen, (100, 150, 200), self.kind_dropdown_rect, 2)
        kind_surf = self.font.render(self.kind_selected, True, (220, 220, 220))
        screen.blit(kind_surf, (self.kind_dropdown_rect.x + 8, self.kind_dropdown_rect.y + 8))
        
        # Buttons
        pygame.draw.rect(screen, (80, 150, 80), self.ok_button_rect)
        pygame.draw.rect(screen, (100, 180, 100), self.ok_button_rect, 2)
        ok_text = self.font.render("Create", True, (255, 255, 255))
        ok_rect = ok_text.get_rect(center=self.ok_button_rect.center)
        screen.blit(ok_text, ok_rect)
        
        pygame.draw.rect(screen, (150, 100, 100), self.cancel_button_rect)
        pygame.draw.rect(screen, (180, 120, 120), self.cancel_button_rect, 2)
        cancel_text = self.font.render("Cancel", True, (255, 255, 255))
        cancel_rect = cancel_text.get_rect(center=self.cancel_button_rect.center)
        screen.blit(cancel_text, cancel_rect)


class ChatBot:
    """Interactive chatbot interface for capsule network communication."""
    
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.messages: list[tuple[str, str]] = []  # (speaker, text) tuples
        self.input_text = ""
        self.cursor_visible = True
        self.cursor_blink_time = 0
        self.input_rect = pygame.Rect(rect.x + 10, rect.bottom - 60, rect.width - 20, 45)
        self.dialogue_rect = pygame.Rect(rect.x + 10, rect.y + 120, rect.width - 20, rect.bottom - 185)
        self.avatar_rect = pygame.Rect(rect.x + rect.width // 2 - 25, rect.y + 10, 50, 50)
        self.font_small = pygame.font.SysFont("Arial", 12)
        self.font_input = pygame.font.SysFont("Arial", 14)
        
        # Question handling
        self.pending_question = None
        self.answer_received = False
        self.last_answer = None
        
        # Add initial greeting
        self.add_message("CharRing", "Hello! I'm your capsule network. Click here and type to chat.")
    
    def add_message(self, speaker: str, text: str) -> None:
        """Add a message to conversation history."""
        self.messages.append((speaker, text))
    
    def ask_question(self, question: str) -> None:
        """Ask the user a question and wait for response."""
        self.pending_question = question
        self.answer_received = False
        self.last_answer = None
        self.add_message("CharRing", question)
    
    def get_answer(self) -> Optional[str]:
        """Get the answer if available, otherwise None."""
        if self.answer_received:
            answer = self.last_answer
            self.pending_question = None
            self.answer_received = False
            self.last_answer = None
            return answer
        return None
    
    def has_pending_question(self) -> bool:
        """Check if there's a pending question."""
        return self.pending_question is not None
    
    def set_input(self, text: str) -> None:
        """Set input field text."""
        self.input_text = text
    
    def handle_input(self, event: pygame.event.Event) -> None:
        """Handle keyboard and paste events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.input_text.strip():
                    self.add_message("You", self.input_text)
                    # If there's a pending question, record the answer
                    if self.pending_question:
                        self.last_answer = self.input_text.strip()
                        self.answer_received = True
                    self.input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.key == pygame.K_v and pygame.key.get_mods() & pygame.KMOD_CTRL:
                try:
                    import subprocess
                    if sys.platform == "win32":
                        clipboard = subprocess.check_output(['powershell', '-Command', 'Get-Clipboard'], text=True)
                    else:
                        clipboard = subprocess.check_output(['xclip', '-selection', 'clipboard', '-o'], text=True)
                    self.input_text += clipboard.strip()
                except Exception:
                    pass
        elif event.type == pygame.TEXTINPUT:
            self.input_text += event.text
    
    def draw(self, screen: pygame.Surface) -> None:
        """Render chatbot interface."""
        # Background panel
        pygame.draw.rect(screen, (30, 35, 45), self.rect)
        pygame.draw.rect(screen, (60, 70, 90), self.rect, 2)
        
        # Avatar with glasses
        self._draw_avatar(screen)
        
        # Dialogue history
        self._draw_dialogue(screen)
        
        # Input box
        self._draw_input(screen)
    
    def _draw_avatar(self, screen: pygame.Surface) -> None:
        """Draw avatar face with glasses."""
        center_x = self.avatar_rect.centerx
        center_y = self.avatar_rect.centery
        
        # Head circle
        pygame.draw.circle(screen, (220, 190, 120), (center_x, center_y), 20)
        
        # Eyes
        pygame.draw.circle(screen, (40, 40, 40), (center_x - 8, center_y - 5), 4)
        pygame.draw.circle(screen, (40, 40, 40), (center_x + 8, center_y - 5), 4)
        
        # Glasses frames
        pygame.draw.circle(screen, (100, 100, 100), (center_x - 8, center_y - 5), 7, 2)
        pygame.draw.circle(screen, (100, 100, 100), (center_x + 8, center_y - 5), 7, 2)
        pygame.draw.line(screen, (100, 100, 100), (center_x - 1, center_y - 5), (center_x + 1, center_y - 5), 2)
        
        # Smile
        pygame.draw.arc(screen, (40, 40, 40), (center_x - 10, center_y + 2, 20, 12), 0, 3.14, 2)
    
    def _draw_dialogue(self, screen: pygame.Surface) -> None:
        """Draw message history in scrollable box."""
        pygame.draw.rect(screen, (20, 25, 35), self.dialogue_rect)
        pygame.draw.rect(screen, (100, 120, 150), self.dialogue_rect, 1)
        
        y = self.dialogue_rect.y + 8
        for speaker, text in self.messages[-8:]:  # Show last 8 messages
            color = (150, 200, 255) if speaker == "CharRing" else (200, 255, 150)
            label = self.font_small.render(f"{speaker}:", True, color)
            screen.blit(label, (self.dialogue_rect.x + 5, y))
            y += 18
            
            # Wrap text
            words = text.split()
            line = ""
            for word in words:
                test_line = (line + " " + word).strip()
                if self.font_small.size(test_line)[0] > self.dialogue_rect.width - 15:
                    msg = self.font_small.render(line, True, (200, 200, 200))
                    screen.blit(msg, (self.dialogue_rect.x + 8, y))
                    y += 14
                    line = word
                else:
                    line = test_line
            if line:
                msg = self.font_small.render(line, True, (200, 200, 200))
                screen.blit(msg, (self.dialogue_rect.x + 8, y))
                y += 14
    
    def _draw_input(self, screen: pygame.Surface) -> None:
        """Draw input field."""
        pygame.draw.rect(screen, (40, 50, 65), self.input_rect)
        pygame.draw.rect(screen, (100, 150, 200), self.input_rect, 2)
        
        # Blink cursor
        self.cursor_blink_time += 1
        show_cursor = (self.cursor_blink_time // 10) % 2 == 0
        
        # Input text
        text_surf = self.font_input.render(self.input_text, True, (220, 220, 220))
        screen.blit(text_surf, (self.input_rect.x + 8, self.input_rect.y + 8))
        
        # Cursor
        if show_cursor:
            cursor_x = self.input_rect.x + 8 + text_surf.get_width()
            pygame.draw.line(screen, (100, 255, 100), (cursor_x, self.input_rect.y + 5), (cursor_x, self.input_rect.bottom - 5), 2)
        
        # Placeholder
        if not self.input_text:
            placeholder = self.font_input.render("Type here... (Ctrl+V to paste)", True, (150, 150, 150))
            screen.blit(placeholder, (self.input_rect.x + 8, self.input_rect.y + 8))


class CapsuleOverlay:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.center = (rect.width // 2, rect.height // 2)
        self.radius_scale = min(rect.width, rect.height) * 0.5
        state_data = _load_ringpose_state(RINGPOSE_STATE_PATH)
        self.system = ROCASystem.from_dict(state_data, self.center, self.radius_scale) if state_data else ROCASystem(self.center, self.radius_scale)
        self.core = PersonalityCore(self.center)
        self.rings = _build_rings_for_rect(rect)
        self.font = pygame.font.SysFont("Arial", 16)
        self.chatbot: Optional[ChatBot] = None  # Reference to chatbot for querying
        
        # Keyframe anchoring: maps capsule name -> list of anchored frame numbers
        # Restored from persistent state to preserve animation references across sessions
        self.keyframe_anchors: dict[str, list[int]] = state_data.get("keyframe_anchors", {}) if state_data else {}

    def set_chatbot(self, chatbot: "ChatBot") -> None:
        """Set reference to chatbot for user interaction."""
        self.chatbot = chatbot
    
    def ask_user(self, question: str) -> Optional[str]:
        """Ask the user a question through the chatbot and return their answer."""
        if not self.chatbot:
            return None
        self.chatbot.ask_question(question)
        return None  # Answer will be available next frame via get_user_answer()
    
    def get_user_answer(self) -> Optional[str]:
        """Get the answer to the last question if available."""
        if not self.chatbot:
            return None
        return self.chatbot.get_answer()
    
    def anchor_keyframe(self, capsule_name: str, frame_number: int) -> bool:
        """Anchor a keyframe to a capsule for animation consistency.
        
        Args:
            capsule_name: Name of the capsule to anchor (e.g., "JumpApex")
            frame_number: Frame index to anchor
        
        Returns:
            True if anchor was set, False if capsule doesn't exist
        """
        # Verify capsule exists
        capsule = None
        for cap in self.system.capsules.values():
            if cap.name == capsule_name:
                capsule = cap
                break
        
        if not capsule:
            if self.chatbot:
                self.chatbot.add_message("CharRing", f"Can't anchor: capsule '{capsule_name}' not found.")
            return False
        
        # Record the anchor
        if capsule_name not in self.keyframe_anchors:
            self.keyframe_anchors[capsule_name] = []
        
        if frame_number not in self.keyframe_anchors[capsule_name]:
            self.keyframe_anchors[capsule_name].append(frame_number)
            self.keyframe_anchors[capsule_name].sort()
        
        if self.chatbot:
            self.chatbot.add_message("CharRing", f"Anchored keyframe {frame_number} to '{capsule_name}'")
        
        return True
    
    def get_symbolic_pose(self, capsule_name: str) -> Optional[list[float]]:
        """Get the symbolic pose vector for a capsule by name.
        
        This is guaranteed to be the same across sessions for the same capsule name.
        
        Args:
            capsule_name: Name of the capsule
        
        Returns:
            8D pose vector if capsule exists, None otherwise
        """
        for cap in self.system.capsules.values():
            if cap.name == capsule_name:
                return cap.pose_vector.copy()
        return None
    
    def get_anchored_frames(self, capsule_name: str) -> list[int]:
        """Get all frames anchored to a capsule.
        
        Args:
            capsule_name: Name of the capsule
        
        Returns:
            List of frame numbers anchored to this capsule
        """
        return self.keyframe_anchors.get(capsule_name, [])
    
    def get_capsule_for_frame(self, frame_number: int) -> Optional[str]:
        """Get the capsule anchored to a specific frame.
        
        Args:
            frame_number: Frame index
        
        Returns:
            Capsule name if anchored, None otherwise
        """
        for capsule_name, frames in self.keyframe_anchors.items():
            if frame_number in frames:
                return capsule_name
        return None
    
    def list_anchors(self) -> str:
        """Get a formatted list of all keyframe anchors.
        
        Returns:
            String describing all anchored keyframes
        """
        if not self.keyframe_anchors:
            return "No keyframes anchored yet."
        
        lines = []
        for capsule_name in sorted(self.keyframe_anchors.keys()):
            frames = self.keyframe_anchors[capsule_name]
            lines.append(f"{capsule_name}: frames {frames}")
        return "\n".join(lines)

    def resize(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.center = (rect.width // 2, rect.height // 2)
        self.radius_scale = min(rect.width, rect.height) * 0.5
        self.rings = _build_rings_for_rect(rect)
        # Re-anchor targets
        for cap in self.system.capsules.values():
            inner_frac, outer_frac = _band_for_kind(cap.kind)
            cap.target_radius = random.uniform(inner_frac, outer_frac) * self.radius_scale
        self.core.set_pos(self.center)

    def update(self, dt_ms: int) -> None:
        self.core.update(dt_ms)
        for cap in self.system.capsules.values():
            cap.update(dt_ms)

    def draw(self, screen: pygame.Surface) -> None:
        surface = pygame.Surface(self.rect.size)
        surface.fill((15, 15, 22))

        for ring in self.rings[::-1]:
            ring.draw(surface, self.center)
        self.core.draw(surface)

        # Position cache
        positions: dict[str, tuple[int, int]] = {}
        for cap in self.system.capsules.values():
            if cap.parent_id:
                continue
            positions[cap.id] = cap.pos(self.center)
        for cap in self.system.capsules.values():
            if not cap.parent_id:
                continue
            parent_pos = positions.get(cap.parent_id)
            if parent_pos is None:
                parent_pos = cap.pos(self.center)
            positions[cap.id] = cap.pos(self.center, parent_pos)

        for cap in self.system.capsules.values():
            pos = positions.get(cap.id, cap.pos(self.center))
            radius = 10 + int(cap.pulse)
            pygame.draw.circle(surface, _color_for_kind(cap.kind), pos, radius)
            label = self.font.render(cap.name[:18], True, (225, 230, 240))
            surface.blit(label, (pos[0] + 12, pos[1] - 8))

        screen.blit(surface, self.rect.topleft)

    def save(self) -> None:
        data = self.system.to_dict()
        # Include keyframe anchors for animation consistency
        data["keyframe_anchors"] = self.keyframe_anchors
        _save_ringpose_state(RINGPOSE_STATE_PATH, data)

# Label class for text rendering
class Label:
    def __init__(self, text, position_dict, text_color="black", font_size=18):
        self.text = text
        self.position_dict = position_dict
        self.text_color = text_color
        self.font = pygame.font.SysFont("Arial", font_size)
        self.surface = None
        self.rect = None
        self.update_surface()

    def update_surface(self):
        self.surface = self.font.render(self.text, True, self.text_color)
        self.rect = self.surface.get_rect(**self.position_dict)

    def set_text(self, new_text):
        self.text = new_text
        self.update_surface()

    def draw(self, screen):
        screen.blit(self.surface, self.rect)

# Enhanced DrawingGUI implementation
class DrawingGUI:
    def __init__(self):
        # Top toolbar (above drawing area) for primary tools
        toolbar_y = max(10, DRAWING_AREA.y + 10)
        primary_x = DRAWING_AREA.x + 16

        # Button layout adjustments
        self.buttons = [
            {"label": "Draw", "rect": pygame.Rect(primary_x, toolbar_y, 120, 36)},
            {"label": "Capsule", "rect": pygame.Rect(primary_x + 130, toolbar_y, 140, 36)},
            # Top column
            {"label": "New Frame", "rect": pygame.Rect(950, 100, 100, 30)},
            {"label": "Train", "rect": pygame.Rect(950, 140, 100, 30)},
            {"label": "In-Between", "rect": pygame.Rect(950, 180, 100, 30)},
            {"label": "Save", "rect": pygame.Rect(950, 220, 100, 30)},
            {"label": "Play", "rect": pygame.Rect(950, 260, 100, 30)},
            # Middle column
            {"label": "Add Layer", "rect": pygame.Rect(950, 300, 100, 30)},
            {"label": "Del Layer", "rect": pygame.Rect(950, 340, 100, 30)},
            {"label": "Undo", "rect": pygame.Rect(950, 380, 100, 30)},
            {"label": "Redo", "rect": pygame.Rect(950, 420, 100, 30)},
            {"label": "Import", "rect": pygame.Rect(950, 460, 100, 30)},
            # Bottom column
            {"label": "Load Images", "rect": pygame.Rect(950, 500, 100, 30)},
            {"label": "Load Video", "rect": pygame.Rect(950, 540, 100, 30)},
            {"label": "Export SVG", "rect": pygame.Rect(950, 580, 100, 30)},
            {"label": "Export Video", "rect": pygame.Rect(950, 620, 100, 30)},
            # Capsule controls
            {"label": "New Capsule", "rect": pygame.Rect(950, 660, 100, 30)},
        ]

        # Slider adjustments
        self.sliders = {
            "brush_size": {
                "rect": pygame.Rect(950, 660, 150, 20),
                "value": 5,
                "min": 1,
                "max": 50
            },
            "playback_speed": {
                "rect": pygame.Rect(950, 690, 150, 20),
                "value": 30,
                "min": 1,
                "max": 60
            }
        }

        # Visual improvements (professional palette)
        self.button_bg = (32, 58, 86)
        self.button_hover_bg = (48, 90, 132)
        self.button_text = (235, 239, 245)
        self.button_border = (18, 33, 50)
        self.button_shadow = (0, 0, 0, 40)
        self.slider_bg = (218, 223, 229)
        
        self.font = pygame.font.SysFont("Arial", 18)
        self.tooltip_font = pygame.font.SysFont("Arial", 14)
        self.timeline = pygame.Rect(50, 700, 800, 30)
        self.hover_button = None
        self.active_slider = None

        # Add frame counter
        self.frame_label = Label("Frame: 1", 
                               {"topright": (SCREEN_RECT.right - 5, 0)},
                               text_color="gray20", 
                               font_size=32)
        
    def handle_hover(self, pos):
        """Update which button is being hovered over"""
        self.hover_button = None
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                self.hover_button = button["label"]
                break

    def handle_slider(self, pos, clicking):
        """Handle slider interactions"""
        self.active_slider = None
        for name, slider in self.sliders.items():
            if clicking and slider["rect"].collidepoint(pos):
                self.active_slider = name
                
            if self.active_slider == name and pos is not None:
                # Calculate slider value based on mouse position
                x_pos = max(slider["rect"].left, min(pos[0], slider["rect"].right))
                ratio = (x_pos - slider["rect"].left) / slider["rect"].width
                slider["value"] = slider["min"] + ratio * (slider["max"] - slider["min"])
                slider["value"] = int(slider["value"])  # For discrete values

    def draw(self, screen, state):
        # Draw top toolbar behind primary buttons
        toolbar_height = 56
        toolbar_rect = pygame.Rect(DRAWING_AREA.x, DRAWING_AREA.y, DRAWING_AREA.width, toolbar_height)
        pygame.draw.rect(screen, (245, 247, 250), toolbar_rect)
        pygame.draw.line(screen, (210, 215, 222), toolbar_rect.bottomleft, toolbar_rect.bottomright, 1)

        # Draw main UI panel at bottom
        pygame.draw.rect(screen, (240, 240, 240), (0, 610, 1200, 190))
        
        # Draw buttons with improved styling
        for button in self.buttons:
            bg_color = self.button_hover_bg if button["label"] == self.hover_button else self.button_bg
            # Shadow
            shadow_rect = button["rect"].copy()
            shadow_rect.move_ip(0, 2)
            shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, self.button_shadow, shadow_surface.get_rect(), border_radius=6)
            screen.blit(shadow_surface, shadow_rect.topleft)

            pygame.draw.rect(screen, bg_color, button["rect"], border_radius=6)
            pygame.draw.rect(screen, self.button_border, button["rect"], 1, border_radius=6)
            
            text = self.font.render(button["label"], True, self.button_text)
            text_rect = text.get_rect(center=button["rect"].center)
            screen.blit(text, text_rect)

        # Draw sliders with better visibility
        for name, slider in self.sliders.items():
            # Slider track
            pygame.draw.rect(screen, self.slider_bg, slider["rect"], border_radius=3)
            
            # Slider handle
            handle_x = slider["rect"].left + int((slider["value"] - slider["min"]) / 
                        (slider["max"] - slider["min"]) * slider["rect"].width)
            pygame.draw.circle(screen, (80, 80, 80), 
                             (handle_x, slider["rect"].centery), 
                             7)
            
            # Value label
            label = f"{name.replace('_', ' ')}: {slider['value']}"
            text = self.font.render(label, True, (40, 40, 40))
            screen.blit(text, (slider["rect"].x, slider["rect"].y - 20))

        # Draw timeline with frame numbers
        pygame.draw.rect(screen, (220, 220, 220), self.timeline, border_radius=3)
        if state.animation_frames:
            frame_width = self.timeline.width / len(state.animation_frames)
            for i in range(len(state.animation_frames)):
                frame_rect = pygame.Rect(
                    self.timeline.x + i * frame_width,
                    self.timeline.y,
                    frame_width - 2,
                    self.timeline.height
                )
                color = (100, 150, 200) if i == state.current_frame else (180, 180, 180)
                pygame.draw.rect(screen, color, frame_rect, border_radius=2)
                
                # Frame number
                text = self.font.render(str(i+1), True, (40, 40, 40))
                text_rect = text.get_rect(center=frame_rect.center)
                screen.blit(text, text_rect)

        # Training progress indicator
        if state.is_training:
            progress_width = 800 * state.training_progress
            pygame.draw.rect(screen, (200, 200, 200), (50, 750, 800, 20), border_radius=3)
            pygame.draw.rect(screen, (80, 180, 80), (50, 750, progress_width, 20), border_radius=3)
            status = f"Training: {state.training_progress*100:.1f}%"
            text = self.font.render(status, True, (40, 40, 40))
            screen.blit(text, (860, 750))

        # Update and draw frame counter
        frame_text = f"Frame: {state.current_frame + 1}"
        self.frame_label.set_text(frame_text)
        self.frame_label.draw(screen)

# New image import functionality
def import_image_sequence(state):
    """Load multiple images using file dialog"""
    try:
        # Create and withdraw Tk root to hide the window
        root = tk.Tk()
        root.withdraw()
        
        # Open file dialog for multiple image selection
        filetypes = [
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
            ("All files", "*.*")
        ]
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=filetypes
        )
        
        if not file_paths:
            return

        # Load and process each selected image
        for path in sorted(file_paths):
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.scale(img, (800, 600))
            if state.current_frame < len(state.animation_frames):
                state.animation_frames.insert(state.current_frame + 1, img)
            else:
                state.animation_frames.append(img)
            
        print(f"Imported {len(file_paths)} images")
        
        # Switch to the first imported frame
        if file_paths:
            state.switch_to_frame(state.current_frame + 1)
            
    except Exception as e:
        print(f"Import failed: {str(e)}")
        messagebox.showerror("Error", f"Failed to import images: {str(e)}")

# New video import functionality
def import_video_frames(state):
    """Load video file and split into frames"""
    try:
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if not video_path:
            return
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            frame = pygame.transform.scale(frame, (800, 600))
            state.video_frames.append(frame)
        cap.release()
        print(f"Imported {len(state.video_frames)} frames from video")
    except Exception as e:
        print(f"Video import failed: {str(e)}")

# Modified training function with progress tracking
def train_model(model, state, epochs=50):
    if len(state.animation_frames) + len(state.imported_images) < 2:
        print("Need at least 2 frames to train")
        return
    
    # Combine drawn and imported frames
    all_frames = state.animation_frames + state.imported_images
    
    # Create training pairs (current frame -> next frame)
    frame_pairs = []
    for i in range(len(all_frames)-1):
        frame_pairs.append((all_frames[i], all_frames[i+1]))
    
    dataset = AnimationDataset(frame_pairs, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    criterion = nn.L1Loss()  # Better for image generation
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    model.train()
    state.is_training = True
    total_batches = len(dataloader) * epochs
    
    try:
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Update training progress
                state.training_progress = ((epoch * len(dataloader)) + batch_idx) / total_batches
                
    except KeyboardInterrupt:
        print("Training interrupted")
    
    state.is_training = False
    torch.save(model.state_dict(), "unet_trained.pth")
    print("Training completed and model saved")

# Enhanced Event Processing
def process_events(state, gui, unet, capsule_overlay: "CapsuleOverlay", chatbot: "ChatBot", capsule_dialog: "CapsuleInputDialog"):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        
        # Capsule dialog input
        capsule_dialog.handle_input(event)
        
        # Chatbot text input
        if capsule_dialog.active:
            pass  # Dialog has priority
        else:
            chatbot.handle_input(event)
            
        # Add keyboard navigation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:  # Left arrow key
                if state.current_frame > 0:
                    state.switch_to_frame(state.current_frame - 1)
                    print(f"Switched to frame {state.current_frame}")
            elif event.key == pygame.K_RIGHT:  # Right arrow key
                if state.current_frame < len(state.animation_frames) - 1:
                    state.switch_to_frame(state.current_frame + 1)
                    print(f"Switched to frame {state.current_frame}")
            
        # Mouse handling
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Chatbot input click
            if not capsule_dialog.active and chatbot.input_rect.collidepoint(event.pos):
                pass  # Input box ready for text
            
            # Add timeline frame selection
            elif gui.timeline.collidepoint(event.pos):
                if state.animation_frames:
                    frame_width = gui.timeline.width / len(state.animation_frames)
                    frame_index = int((event.pos[0] - gui.timeline.x) / frame_width)
                    frame_index = max(0, min(frame_index, len(state.animation_frames)-1))
                    state.switch_to_frame(frame_index)
                    print(f"Selected frame {frame_index}")
            
            # Handle GUI interactions first
            gui.handle_hover(event.pos)
            gui.handle_slider(event.pos, True)
            
            # Check button clicks
            button_clicked = False
            for button in gui.buttons:
                if button["rect"].collidepoint(event.pos):
                    if button["label"] == "Draw":
                        state.mode = "draw"
                    elif button["label"] == "Capsule":
                        state.mode = "capsule"
                    elif button["label"] == "New Capsule":
                        capsule_dialog.open()
                    else:
                        handle_button_click(button["label"], state, unet)
                    button_clicked = True
                    break
                    
            # Handle drawing if not clicking GUI
            if state.mode == "draw" and (not button_clicked) and DRAWING_AREA.collidepoint(event.pos):
                handle_drawing_start(event.pos, state)

        elif event.type == pygame.MOUSEBUTTONUP:
            state.drawing = False
            gui.active_slider = None

        elif event.type == pygame.MOUSEMOTION:
            gui.handle_hover(event.pos)
            gui.handle_slider(event.pos, pygame.mouse.get_pressed()[0])
            if state.mode == "draw" and state.drawing and DRAWING_AREA.collidepoint(event.pos):
                handle_drawing_motion(event.pos, state)
                state.frame_changed = True

        # Update frame counter when switching frames
        if event.type == pygame.MOUSEBUTTONDOWN:
            if gui.timeline.collidepoint(event.pos):
                if state.animation_frames:
                    frame_width = gui.timeline.width / len(state.animation_frames)
                    frame_index = int((event.pos[0] - gui.timeline.x) / frame_width)
                    frame_index = max(0, min(frame_index, len(state.animation_frames)-1))
                    state.switch_to_frame(frame_index)
                    # Frame counter will update in next gui.draw() call

    return True

# Fixed In-Between Generation
def generate_in_between(start_frame, end_frame, model):
    # Convert surfaces to tensors
    start_array = pygame.surfarray.array3d(start_frame).transpose(1, 0, 2)
    end_array = pygame.surfarray.array3d(end_frame).transpose(1, 0, 2)
    
    # Convert to PIL Images and transform
    start_pil = Image.fromarray(start_array)
    end_pil = Image.fromarray(end_array)
    
    start_tensor = transform(start_pil).unsqueeze(0).to(device)
    end_tensor = transform(end_pil).unsqueeze(0).to(device)
    
    # Concatenate along channel dimension
    combined = torch.cat([start_tensor, end_tensor], dim=1)
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        middle = model(combined)
        result = middle.squeeze().cpu().numpy().transpose(1, 2, 0)
        result = (result * 255).clip(0, 255).astype(np.uint8)
    
    return pygame.surfarray.make_surface(result.transpose(1, 0, 2))

# Fixed Auto-Complete Implementation
def auto_complete_drawing(state, model):
    if not state.animation_frames:
        return
    
    # Get current frame and duplicate for model input
    current_frame = state.layers[state.current_layer]
    current_array = pygame.surfarray.array3d(current_frame).transpose(1, 0, 2)
    current_pil = Image.fromarray(current_array)
    
    # Create dummy end frame (current frame)
    input_tensor = transform(current_pil).unsqueeze(0).to(device)
    dummy_input = torch.cat([input_tensor, input_tensor], dim=1)
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        completed = model(dummy_input)
        result = completed.squeeze().cpu().numpy().transpose(1, 2, 0)
        result = (result * 255).clip(0, 255).astype(np.uint8)
    
    # Update drawing layer
    completed_surf = pygame.surfarray.make_surface(result.transpose(1, 0, 2))
    state.layers[state.current_layer] = completed_surf

# New SVG export functionality
def export_svg(state):
    """Export current drawing as SVG file"""
    try:
        svg_path = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files", "*.svg")])
        if not svg_path:
            return
        with open(svg_path, 'w') as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">\n')
            for obj in state.svg_objects:
                f.write(obj.to_svg())
            f.write('</svg>')
        print(f"SVG exported to {svg_path}")
    except Exception as e:
        print(f"SVG export failed: {str(e)}")

# Improved Video Export with Alpha Handling
def export_video(state):
    if len(state.animation_frames) < 1:
        print("No frames to export")
        return
    
    os.makedirs("output", exist_ok=True)
    frame_size = (800, 600)
    out = cv2.VideoWriter("output/animation.mp4", 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         state.playback_speed, 
                         frame_size)
    
    for frame in state.animation_frames:
        # Convert to RGBA array
        rgb_array = pygame.surfarray.array3d(frame).transpose(1, 0, 2)
        alpha_array = pygame.surfarray.array_alpha(frame).transpose(0, 1)
        
        # Composite with white background
        frame_rgba = np.dstack((rgb_array, alpha_array))
        white_bg = np.ones_like(rgb_array) * 255
        alpha = alpha_array[..., np.newaxis] / 255.0
        composite = (white_bg * (1 - alpha) + rgb_array * alpha).astype(np.uint8)
        
        # Convert to BGR and write
        frame_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print("Animation exported successfully")

# ... (remaining functions with similar fixes)

def handle_drawing_start(pos, state):
    state.drawing = True
    x = pos[0] - DRAWING_AREA.x
    y = pos[1] - DRAWING_AREA.y
    state.last_pos = (x, y)
    
    # Create new undo state
    state.undo_stack.append(state.layers[state.current_layer].copy())

def handle_drawing_motion(pos, state):
    if not state.drawing: return
    
    x = pos[0] - DRAWING_AREA.x
    y = pos[1] - DRAWING_AREA.y
    
    pygame.draw.line(
        state.layers[state.current_layer],
        state.brush_color,
        state.last_pos,
        (x, y),
        state.brush_size
    )
    state.last_pos = (x, y)

# Enhanced handle_button_click
def handle_button_click(label, state, model):
    try:
        match label:
            case "New Frame":
                if state.frame_changed:
                    state.save_current_frame()
                new_frame = pygame.Surface((800, 600), pygame.SRCALPHA)
                state.animation_frames.append(new_frame)
                state.switch_to_frame(len(state.animation_frames) - 1)
                print(f"New frame created (total: {len(state.animation_frames)})")

            case "Train":
                if len(state.animation_frames) >= 2:
                    train_model(model, state)
                else:
                    print("Need at least 2 frames to train")

            case "In-Between":
                if len(state.animation_frames) >= 2:
                    start = state.animation_frames[-2]
                    end = state.animation_frames[-1]
                    between = generate_in_between(start, end, model)
                    state.animation_frames.insert(-1, between)
                    print(f"Generated in-between frame (total: {len(state.animation_frames)})")

            case "Save":
                export_video(state)

            case "Play":
                state.playback = not state.playback
                if state.playback:
                    state.playback_frame = state.current_frame
                    state.last_frame_time = time.time()
                    print("Playback started")
                else:
                    state.current_frame = state.playback_frame
                    print("Playback stopped")

            case "Add Layer":
                state.layers.append(pygame.Surface((800, 600), pygame.SRCALPHA))
                state.current_layer = len(state.layers) - 1
                print(f"Added layer (total: {len(state.layers)})")

            case "Del Layer":
                if len(state.layers) > 1:
                    state.layers.pop()
                    state.current_layer = min(state.current_layer, len(state.layers)-1)
                    print(f"Removed layer (remaining: {len(state.layers)})")

            case "Undo":
                if state.undo_stack:
                    state.redo_stack.append(state.layers[state.current_layer].copy())
                    state.layers[state.current_layer] = state.undo_stack.pop()
                    print("Undo performed")

            case "Redo":
                if state.redo_stack:
                    state.undo_stack.append(state.layers[state.current_layer].copy())
                    state.layers[state.current_layer] = state.redo_stack.pop()
                    print("Redo performed")

            case "Import":
                import_image_sequence(state)

            case "Load Images":
                import_image_sequence(state)
            case "Load Video":
                import_video_frames(state)
            case "Export SVG":
                export_svg(state)
            case "Export Video":
                export_video(state)

            case _:
                print(f"Button '{label}' not implemented")

    except Exception as e:
        print(f"Error handling {label}: {str(e)}")

def main():
    # Initialize core components
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("AI Drawing Animation Tool")
    
    # Create application state and UI
    state = DrawingState()
    gui = DrawingGUI()
    capsule_overlay = CapsuleOverlay(DRAWING_AREA)
    
    # Create chatbot on right side
    chatbot_rect = pygame.Rect(860, 0, 340, 600)
    chatbot = ChatBot(chatbot_rect)
    
    # Create capsule input dialog
    capsule_dialog = CapsuleInputDialog()
    
    # Connect chatbot to capsule system
    capsule_overlay.set_chatbot(chatbot)
    
    # Add initial guidance to chatbot
    chatbot.add_message("CharRing", "Welcome! Create capsules to represent character concepts.")
    
    # Initialize AI model
    unet = UNet().to(device)
    if os.path.exists("unet_trained.pth"):
        unet.load_state_dict(torch.load("unet_trained.pth", map_location=device))
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    while running:
        running = process_events(state, gui, unet, capsule_overlay, chatbot, capsule_dialog)
        
        # Handle capsule dialog result
        if capsule_dialog.result:
            name, kind = capsule_dialog.result
            capsule = create_symbolic_capsule(name, kind, capsule_overlay.system)
            chatbot.add_message("CharRing", f"Created capsule '{name}' ({kind})")
            capsule_dialog.result = None
        
        # Handle playback
        if state.playback:
            current_time = time.time()
            if current_time - state.last_frame_time > 1/state.playback_speed:
                state.playback_frame = (state.playback_frame + 1) % len(state.animation_frames)
                state.last_frame_time = current_time
                
                # Update current frame display during playback
                screen.fill(WHITE)
                screen.blit(state.animation_frames[state.playback_frame], DRAWING_AREA.topleft)
                gui.draw(screen, state)
                pygame.display.flip()
                continue  # Skip regular drawing during playback
        
        # Regular drawing
        else:
            # Save frame if changed before exiting
            if state.frame_changed and pygame.key.get_pressed()[pygame.K_s]:
                state.save_current_frame()
            
            screen.fill(WHITE)

            if state.mode == "draw":
                for layer in state.layers:
                    screen.blit(layer, DRAWING_AREA.topleft)

                # Draw border around the drawing area
                pygame.draw.rect(screen, BLACK, DRAWING_AREA, 2)

                # Draw onion skin
                if state.onion_skinning and state.animation_frames:
                    prev_frame = state.animation_frames[max(0, state.current_frame-1)]
                    prev_frame.set_alpha(50)
                    screen.blit(prev_frame, DRAWING_AREA.topleft)
            else:
                # Capsule network view inside drawing area
                capsule_overlay.update(clock.get_time())
                capsule_overlay.draw(screen)
        
        # Draw chatbot on left side
        chatbot.draw(screen)
        
        gui.draw(screen, state)
        
        # Draw capsule dialog if active
        capsule_dialog.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    # Cleanup
    try:
        capsule_overlay.save()
    except Exception:
        pass
    pygame.quit()
    sys.exit()


# ========================
# Capsule Inspector Panel
# ========================
class CapsuleInspectorPanel:
    """Inspector panel for viewing and editing capsule properties.
    
    Features:
    - Display capsule metadata (ID, name, kind, confidence, etc.)
    - Rename capsules with text input
    - View scope and usage statistics
    - Toggle freeze status
    - Display modification history
    - Show confidence indicators
    - Quick action buttons
    """
    
    WIDTH = 280
    FONT_SIZE = 12
    BUTTON_HEIGHT = 24
    BUTTON_MARGIN = 5
    
    def __init__(self, screen_rect: pygame.Rect, capsules: Optional[list[Capsule]] = None):
        """Initialize inspector panel.
        
        Args:
            screen_rect: Pygame rect for entire screen
            capsules: List of all capsules for live in-between updates
        """
        self.screen_rect = screen_rect
        self.panel_rect = pygame.Rect(
            screen_rect.right - self.WIDTH,
            0,
            self.WIDTH,
            screen_rect.height
        )
        self.capsules = capsules or []  # For live in-between updates
        
        # Selected capsule
        self.selected_capsule: Optional[Capsule] = None
        
        # Rename mode
        self.is_renaming = False
        self.rename_input = ""
        self.rename_input_rect = pygame.Rect(0, 0, 0, 0)
        
        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 14, bold=True)
        self.font_normal = pygame.font.SysFont('Arial', self.FONT_SIZE)
        self.font_small = pygame.font.SysFont('Arial', 10)
        
        # Button rects
        self.freeze_button_rect = pygame.Rect(0, 0, 0, 0)
        self.rename_button_rect = pygame.Rect(0, 0, 0, 0)
        self.scope_button_rect = pygame.Rect(0, 0, 0, 0)
    
    def set_selected_capsule(self, capsule: Optional[Capsule]) -> None:
        """Set the capsule being inspected.
        
        Args:
            capsule: Capsule to inspect, or None to clear
        """
        self.selected_capsule = capsule
        self.is_renaming = False
        self.rename_input = ""
    
    def start_rename(self) -> None:
        """Begin editing capsule name."""
        if self.selected_capsule:
            self.is_renaming = True
            self.rename_input = self.selected_capsule.name
    
    def confirm_rename(self) -> None:
        """Confirm and apply name change, then update affected in-betweens."""
        if self.selected_capsule and self.rename_input.strip():
            self.selected_capsule.name = self.rename_input.strip()
            self.selected_capsule.touch_modification()
            # Update in-betweens if this is a pose capsule
            if self.capsules and not self.selected_capsule.is_inbetween:
                update_inbetween_from_tweak(self.capsules, self.selected_capsule)
        self.is_renaming = False
        self.rename_input = ""
    
    def cancel_rename(self) -> None:
        """Cancel name editing."""
        self.is_renaming = False
        self.rename_input = ""
    
    def handle_text_input(self, event: pygame.event.Event) -> None:
        """Handle text input for rename field.
        
        Args:
            event: pygame.KEYDOWN event
        """
        if not self.is_renaming:
            return
        
        if event.key == pygame.K_RETURN:
            self.confirm_rename()
        elif event.key == pygame.K_ESCAPE:
            self.cancel_rename()
        elif event.key == pygame.K_BACKSPACE:
            self.rename_input = self.rename_input[:-1]
        elif event.unicode and event.unicode.isprintable():
            if len(self.rename_input) < 50:  # Max 50 chars
                self.rename_input += event.unicode
    
    def handle_click(self, pos: tuple[int, int]) -> Optional[str]:
        """Handle click on inspector panel.
        
        Args:
            pos: Click position
        
        Returns:
            Action performed: "rename", "freeze", "scope", or None
        """
        if not self.panel_rect.collidepoint(pos):
            return None
        
        if self.is_renaming:
            # Check if clicking in rename input
            if self.rename_input_rect.collidepoint(pos):
                return "rename_input"
            # Check for confirm/cancel buttons
            return None
        
        # Check buttons
        if self.freeze_button_rect.collidepoint(pos) and self.selected_capsule:
            if self.selected_capsule.is_frozen:
                self.selected_capsule.unfreeze()
            else:
                self.selected_capsule.freeze("locked_from_inspector")
            # Update in-betweens on freeze status change
            if self.capsules and not self.selected_capsule.is_inbetween:
                update_inbetween_from_tweak(self.capsules, self.selected_capsule)
            return "freeze"
        
        if self.rename_button_rect.collidepoint(pos):
            self.start_rename()
            return "rename"
        
        if self.scope_button_rect.collidepoint(pos):
            # Cycle through scopes
            if self.selected_capsule:
                scopes = ["universal", "character-specific", "one-off"]
                current_idx = scopes.index(self.selected_capsule.scope)
                self.selected_capsule.set_scope(scopes[(current_idx + 1) % len(scopes)])
                # Update in-betweens on scope change
                if self.capsules and not self.selected_capsule.is_inbetween:
                    update_inbetween_from_tweak(self.capsules, self.selected_capsule)
            return "scope"
        
        return None
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw inspector panel.
        
        Args:
            screen: Pygame surface
        """
        # Background
        pygame.draw.rect(screen, (240, 240, 240), self.panel_rect)
        pygame.draw.line(
            screen,
            (100, 100, 100),
            (self.panel_rect.left, 0),
            (self.panel_rect.left, self.panel_rect.bottom),
            2
        )
        
        # Title
        title_text = self.font_title.render("INSPECTOR", True, (50, 50, 50))
        screen.blit(title_text, (self.panel_rect.x + 10, self.panel_rect.y + 10))
        
        if not self.selected_capsule:
            no_selection = self.font_normal.render("(No capsule selected)", True, (150, 150, 150))
            screen.blit(no_selection, (self.panel_rect.x + 10, self.panel_rect.y + 40))
            return
        
        capsule = self.selected_capsule
        y = self.panel_rect.y + 35
        
        # --- Capsule Name (with rename capability) ---
        name_label = self.font_small.render("NAME:", True, (100, 100, 100))
        screen.blit(name_label, (self.panel_rect.x + 10, y))
        y += 14
        
        if self.is_renaming:
            # Rename input field
            self.rename_input_rect = pygame.Rect(
                self.panel_rect.x + 10,
                y,
                self.panel_rect.width - 20,
                20
            )
            pygame.draw.rect(screen, (255, 255, 255), self.rename_input_rect)
            pygame.draw.rect(screen, (100, 100, 255), self.rename_input_rect, 2)
            
            input_text = self.font_normal.render(self.rename_input, True, (0, 0, 0))
            screen.blit(input_text, (self.rename_input_rect.x + 5, self.rename_input_rect.y + 3))
            
            # Hint
            hint = self.font_small.render("[ENTER=Save, ESC=Cancel]", True, (150, 150, 150))
            screen.blit(hint, (self.panel_rect.x + 10, y + 24))
            y += 48
        else:
            # Display name with rename button
            name_text = self.font_normal.render(capsule.name, True, (0, 0, 0))
            screen.blit(name_text, (self.panel_rect.x + 10, y))
            
            self.rename_button_rect = pygame.Rect(
                self.panel_rect.x + 10,
                y + 22,
                self.panel_rect.width - 20,
                self.BUTTON_HEIGHT
            )
            pygame.draw.rect(screen, (100, 150, 255), self.rename_button_rect)
            pygame.draw.rect(screen, (50, 100, 200), self.rename_button_rect, 1)
            rename_btn_text = self.font_small.render("RENAME", True, (255, 255, 255))
            rename_text_rect = rename_btn_text.get_rect(center=self.rename_button_rect.center)
            screen.blit(rename_btn_text, rename_text_rect)
            y += 50
        
        # --- Kind ---
        kind_label = self.font_small.render(f"KIND: {capsule.kind}", True, (100, 100, 100))
        screen.blit(kind_label, (self.panel_rect.x + 10, y))
        y += 16
        
        # --- Confidence ---
        confidence = capsule.get_confidence()
        conf_color = (255, 100, 100) if confidence < 0.3 else (255, 200, 100) if confidence < 0.7 else (100, 255, 100)
        conf_label = self.font_small.render(f"CONFIDENCE:", True, (100, 100, 100))
        screen.blit(conf_label, (self.panel_rect.x + 10, y))
        
        # Confidence bar
        bar_rect = pygame.Rect(self.panel_rect.x + 10, y + 14, self.panel_rect.width - 20, 12)
        pygame.draw.rect(screen, (200, 200, 200), bar_rect)
        filled_width = bar_rect.width * confidence
        filled_rect = pygame.Rect(bar_rect.x, bar_rect.y, filled_width, bar_rect.height)
        pygame.draw.rect(screen, conf_color, filled_rect)
        pygame.draw.rect(screen, (100, 100, 100), bar_rect, 1)
        
        # Confidence value
        conf_text = self.font_small.render(f"{confidence:.2f}", True, (0, 0, 0))
        screen.blit(conf_text, (self.panel_rect.x + 10, y + 28))
        y += 50
        
        # --- Usage Statistics ---
        usage_text = self.font_small.render(f"USAGE: {capsule.usage_count}", True, (100, 100, 100))
        screen.blit(usage_text, (self.panel_rect.x + 10, y))
        y += 16
        
        correction_text = self.font_small.render(f"CORRECTIONS: {capsule.correction_count}", True, (100, 100, 100))
        screen.blit(correction_text, (self.panel_rect.x + 10, y))
        y += 20
        
        # --- Scope ---
        scope_label = self.font_small.render(f"SCOPE: {capsule.get_scope_info()}", True, (100, 100, 100))
        screen.blit(scope_label, (self.panel_rect.x + 10, y))
        
        self.scope_button_rect = pygame.Rect(
            self.panel_rect.x + 10,
            y + 16,
            self.panel_rect.width - 20,
            self.BUTTON_HEIGHT
        )
        pygame.draw.rect(screen, (150, 150, 255), self.scope_button_rect)
        pygame.draw.rect(screen, (100, 100, 200), self.scope_button_rect, 1)
        scope_btn_text = self.font_small.render("CHANGE SCOPE", True, (255, 255, 255))
        scope_text_rect = scope_btn_text.get_rect(center=self.scope_button_rect.center)
        screen.blit(scope_btn_text, scope_text_rect)
        y += 44
        
        # --- Frozen Status ---
        freeze_color = (255, 100, 100) if capsule.is_frozen else (150, 255, 150)
        freeze_text = "FROZEN" if capsule.is_frozen else "ACTIVE"
        freeze_status = self.font_small.render(f"STATUS: {freeze_text}", True, (100, 100, 100))
        screen.blit(freeze_status, (self.panel_rect.x + 10, y))
        
        self.freeze_button_rect = pygame.Rect(
            self.panel_rect.x + 10,
            y + 16,
            self.panel_rect.width - 20,
            self.BUTTON_HEIGHT
        )
        pygame.draw.rect(screen, freeze_color, self.freeze_button_rect)
        pygame.draw.rect(screen, (100, 100, 100), self.freeze_button_rect, 1)
        freeze_btn_text = self.font_small.render(
            "UNFREEZE" if capsule.is_frozen else "FREEZE",
            True,
            (255, 255, 255)
        )
        freeze_text_rect = freeze_btn_text.get_rect(center=self.freeze_button_rect.center)
        screen.blit(freeze_btn_text, freeze_text_rect)
        y += 44
        
        # --- Timeline Info ---
        timeline_text = self.font_small.render(
            f"FRAMES: {capsule.start_frame}-{capsule.end_frame}",
            True,
            (100, 100, 100)
        )
        screen.blit(timeline_text, (self.panel_rect.x + 10, y))
        y += 16
        
        # --- Last Modified ---
        modified_text = self.font_small.render(
            f"MODIFIED: {capsule.get_last_modified_str()}",
            True,
            (100, 100, 100)
        )
        screen.blit(modified_text, (self.panel_rect.x + 10, y))
        y += 20
        
        # --- Tags ---
        if capsule.tags:
            tags_label = self.font_small.render("TAGS:", True, (100, 100, 100))
            screen.blit(tags_label, (self.panel_rect.x + 10, y))
            y += 14
            
            for tag in capsule.tags:
                tag_surface = pygame.Surface((self.panel_rect.width - 20, 16))
                tag_surface.fill((220, 220, 220))
                tag_text = self.font_small.render(f"  #{tag}", True, (80, 80, 80))
                tag_surface.blit(tag_text, (0, 2))
                screen.blit(tag_surface, (self.panel_rect.x + 10, y))
                y += 18


# ========================
# Timeline Playback Controller
# ========================
class TimelinePlaybackController:
    """Manages playback, frame scrubbing, and timeline control.
    
    Features:
    - Play/pause/stop control
    - Frame-by-frame stepping (forward/backward)
    - Playback speed control (0.25x to 4.0x)
    - Frame scrubbing (click to jump)
    - Looping (once, loop, ping-pong)
    - FPS-independent frame advancement
    - Integration with TimelineRenderer
    """
    
    # Playback modes
    LOOP_MODE_ONCE = "once"
    LOOP_MODE_LOOP = "loop"
    LOOP_MODE_PINGPONG = "ping-pong"
    
    # Speed presets
    SPEED_PRESETS = {
        "0.25x": 0.25,
        "0.5x": 0.5,
        "1.0x": 1.0,
        "1.5x": 1.5,
        "2.0x": 2.0,
        "4.0x": 4.0,
    }
    
    def __init__(
        self,
        timeline_renderer: TimelineRenderer,
        total_frames: int,
        fps: int = 60,
    ):
        """Initialize playback controller.
        
        Args:
            timeline_renderer: TimelineRenderer instance
            total_frames: Total number of frames in animation
            fps: Frame rate (default 60)
        """
        self.renderer = timeline_renderer
        self.total_frames = total_frames
        self.fps = fps
        
        # Playback state
        self.is_playing = False
        self.current_frame = 0
        self.playback_speed = 1.0  # Multiplier (0.25 to 4.0)
        self.loop_mode = self.LOOP_MODE_LOOP
        self.direction = 1  # 1 = forward, -1 = backward (for ping-pong)
        
        # Time tracking
        self.accumulated_time_ms = 0.0
        self.frame_duration_ms = 1000.0 / fps
        
        # Scrubbing state
        self.is_scrubbing = False
        self.scrub_start_pos = 0
    
    def play(self) -> None:
        """Start playback."""
        self.is_playing = True
    
    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False
    
    def stop(self) -> None:
        """Stop and reset to frame 0."""
        self.is_playing = False
        self.current_frame = 0
        self.accumulated_time_ms = 0.0
        self.renderer.set_playhead(0)
    
    def toggle_play(self) -> None:
        """Toggle between play and pause."""
        self.is_playing = not self.is_playing
    
    def set_speed(self, speed: float) -> None:
        """Set playback speed.
        
        Args:
            speed: Speed multiplier (0.25 to 4.0)
        """
        self.playback_speed = max(0.25, min(4.0, speed))
    
    def set_loop_mode(self, mode: str) -> None:
        """Set looping behavior.
        
        Args:
            mode: "once", "loop", or "ping-pong"
        """
        if mode in (self.LOOP_MODE_ONCE, self.LOOP_MODE_LOOP, self.LOOP_MODE_PINGPONG):
            self.loop_mode = mode
    
    def jump_to_frame(self, frame: int) -> None:
        """Jump to specific frame.
        
        Args:
            frame: Target frame number (clamped to valid range)
        """
        self.current_frame = max(0, min(self.total_frames - 1, frame))
        self.accumulated_time_ms = 0.0
        self.renderer.set_playhead(self.current_frame)
    
    def next_frame(self) -> None:
        """Advance to next frame (frame-by-frame stepping)."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
        else:
            if self.loop_mode == self.LOOP_MODE_LOOP:
                self.current_frame = 0
        
        self.renderer.set_playhead(self.current_frame)
    
    def previous_frame(self) -> None:
        """Go to previous frame (frame-by-frame stepping)."""
        if self.current_frame > 0:
            self.current_frame -= 1
        else:
            if self.loop_mode == self.LOOP_MODE_LOOP:
                self.current_frame = self.total_frames - 1
        
        self.renderer.set_playhead(self.current_frame)
    
    def update(self, dt_ms: int) -> None:
        """Update playback state.
        
        Args:
            dt_ms: Delta time in milliseconds since last frame
        """
        if not self.is_playing:
            return
        
        # Accumulate time
        self.accumulated_time_ms += dt_ms * self.playback_speed
        
        # Advance frames if enough time has accumulated
        while self.accumulated_time_ms >= self.frame_duration_ms:
            self.accumulated_time_ms -= self.frame_duration_ms
            self._advance_frame()
        
        # Update renderer
        self.renderer.set_playhead(self.current_frame)
    
    def _advance_frame(self) -> None:
        """Internal method to advance to next frame."""
        self.current_frame += self.direction
        
        # Handle looping
        if self.current_frame >= self.total_frames:
            if self.loop_mode == self.LOOP_MODE_ONCE:
                self.is_playing = False
                self.current_frame = self.total_frames - 1
            elif self.loop_mode == self.LOOP_MODE_LOOP:
                self.current_frame = 0
            elif self.loop_mode == self.LOOP_MODE_PINGPONG:
                self.direction = -1
                self.current_frame = self.total_frames - 2
        
        elif self.current_frame < 0:
            if self.loop_mode == self.LOOP_MODE_PINGPONG:
                self.direction = 1
                self.current_frame = 1
            else:
                self.is_playing = False
                self.current_frame = 0
    
    def start_scrub(self, pos: tuple[int, int]) -> None:
        """Begin frame scrubbing.
        
        Args:
            pos: Mouse position
        """
        self.is_scrubbing = True
        self.pause()
        frame = self.renderer.get_frame_at_position(pos[0])
        self.jump_to_frame(frame)
    
    def update_scrub(self, pos: tuple[int, int]) -> None:
        """Update scrubbing position.
        
        Args:
            pos: Current mouse position
        """
        if self.is_scrubbing:
            frame = self.renderer.get_frame_at_position(pos[0])
            self.jump_to_frame(frame)
    
    def end_scrub(self) -> None:
        """End frame scrubbing."""
        self.is_scrubbing = False
    
    def get_playback_info(self) -> dict:
        """Get current playback state information.
        
        Returns:
            Dictionary with playback status
        """
        return {
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "is_playing": self.is_playing,
            "is_scrubbing": self.is_scrubbing,
            "playback_speed": self.playback_speed,
            "loop_mode": self.loop_mode,
            "progress": self.current_frame / max(1, self.total_frames - 1),
        }
    
    def handle_keyboard(self, event: pygame.event.Event) -> None:
        """Handle keyboard events for playback control.
        
        Args:
            event: pygame.KEYDOWN event
        """
        if event.key == pygame.K_SPACE:
            self.toggle_play()
        elif event.key == pygame.K_LEFT:
            self.previous_frame()
        elif event.key == pygame.K_RIGHT:
            self.next_frame()
        elif event.key == pygame.K_HOME:
            self.jump_to_frame(0)
        elif event.key == pygame.K_END:
            self.jump_to_frame(self.total_frames - 1)
        elif event.key == pygame.K_l:
            # Cycle loop modes
            modes = [self.LOOP_MODE_ONCE, self.LOOP_MODE_LOOP, self.LOOP_MODE_PINGPONG]
            current_idx = modes.index(self.loop_mode)
            self.set_loop_mode(modes[(current_idx + 1) % len(modes)])
        elif event.key == pygame.K_PERIOD:
            # Increase speed
            current_speed = self.playback_speed
            speeds = list(self.SPEED_PRESETS.values())
            for i, s in enumerate(speeds):
                if current_speed < s:
                    self.set_speed(s)
                    break
        elif event.key == pygame.K_COMMA:
            # Decrease speed
            current_speed = self.playback_speed
            speeds = sorted(self.SPEED_PRESETS.values())
            for i in range(len(speeds) - 1, -1, -1):
                if current_speed > speeds[i]:
                    self.set_speed(speeds[i])
                    break
    
    def draw_playback_info(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        font: pygame.font.Font,
    ) -> None:
        """Draw playback information on screen.
        
        Args:
            screen: Pygame surface
            x, y: Position for info display
            font: Font to use
        """
        info = self.get_playback_info()
        
        # Status line
        status = "Playing" if info["is_playing"] else "Paused"
        if info["is_scrubbing"]:
            status = "Scrubbing"
        
        text = (
            f"{status} | "
            f"Frame: {info['current_frame']}/{info['total_frames']} | "
            f"Speed: {info['playback_speed']:.2f}x | "
            f"Loop: {info['loop_mode']}"
        )
        
        text_surf = font.render(text, True, (255, 255, 255))
        screen.blit(text_surf, (x, y))
    
    def draw_progress_bar(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        fg_color: tuple = (100, 200, 255),
        bg_color: tuple = (50, 50, 50),
    ) -> None:
        """Draw progress bar on screen.
        
        Args:
            screen: Pygame surface
            rect: Area for progress bar
            fg_color: Foreground (filled) color
            bg_color: Background color
        """
        # Background
        pygame.draw.rect(screen, bg_color, rect)
        
        # Filled portion
        progress = self.get_playback_info()["progress"]
        filled_width = rect.width * progress
        filled_rect = pygame.Rect(rect.x, rect.y, filled_width, rect.height)
        pygame.draw.rect(screen, fg_color, filled_rect)
        
        # Border
        pygame.draw.rect(screen, (200, 200, 200), rect, 2)


# Timeline demo removed for experimental build.


if __name__ == "__main__":
    main()
