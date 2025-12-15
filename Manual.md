# ROCA Animator Manual: Complete User Guide

## Part 1: Starting State

### App Launches

You see:

- **Core (ROCA Earth)** in the center: a pulsing, glowing sphere representing your animator identity nucleus.
- **Empty orbit rings** radiating outward in concentric circles: seven functional lanes waiting to receive capsules.
- **Empty timeline** at the bottom of the screen: a horizontal frame ruler with no animation clips yet.
- **Toolbar above the drawing area**: primary buttons (Draw, Capsule) positioned at the top left.
- **Control panel on the right**: buttons for frame manipulation, playback, layer management, and export.

**Nothing is "wrong". This is expected. This is a blank universe.**

You are looking at a fresh ROCA Animator instance—no capsules loaded, no animation history, no character rigs. The orbital map is a clean slate, ready to receive your first pose capsules, transitions, timing profiles, and cycles. The timeline is waiting for your first animation keyframes.

---

## Part 2: The Interface Explained

### 2.1 The Orbital Capsule Map (Center)

The **Orbital Capsule Map** is a Saturn-ring visualization of your animation library. It's the beating heart of ROCA Animator.

#### What You're Seeing

```
                            ┌──────────────────────────────┐
                            │   OUTER RINGS (Experimental) │
                            │      (Unused, new)           │
                            ├──────────────────────────────┤
                            │  Memory Ring (cold storage)  │
                            ├──────────────────────────────┤
                            │  Topic / Experimental Ring    │
                            ├──────────────────────────────┤
                            │    Skill Ring (Actions)      │
                            ├──────────────────────────────┤
                            │    Style Ring (Looks)        │
                            ├──────────────────────────────┤
                            │   Character Ring (Actors)    │
                            ├──────────────────────────────┤
                            │                              │
                            │    ╔════════════════╗        │
                            │    ║  ROCA EARTH    ║        │
                            │    ║   (You)        ║        │
                            │    ║   Pulsing      ║        │
                            │    ║   Glowing      ║        │
                            │    ╚════════════════╝        │
                            │                              │
                            ├──────────────────────────────┤
                            │   Inner Character Ring       │
                            ├──────────────────────────────┤
                            │    Inner Style Ring          │
                            ├──────────────────────────────┤
                            │    Inner Skill Ring          │
                            ├──────────────────────────────┤
                            │  Inner Topic / Core Ring     │
                            ├──────────────────────────────┤
                            │  Inner Memory Ring (recent)  │
                            └──────────────────────────────┘
```

#### How It Works

The **concentric rings** organize your animation assets by kind:

1. **Functional Lanes** (outward from core):
   - **Core (0–8%)**: Your animator identity, core preferences, default timing.
   - **Character (18–26%)**: Character rigs, body types, personalities.
   - **Style (26–32%)**: Visual styles, drawing aesthetics, color palettes.
   - **Skill (32–38%)**: Movements and actions (walk cycles, jumps, grabs).
   - **Topic (38–44%)**: Concepts, dialogue beats, acting moments.
   - **Memory (44–50%)**: Notes, reference media, learned patterns.
   - **Experimental (50–58%)**: Untested, new, or archived ideas.

2. **Distance as Salience**:
   - **Inner rings**: Capsules you use frequently. They orbit close to the center because you keep activating them.
   - **Outer rings**: Experimental or rarely-used capsules. They drift outward because you haven't used them recently.
   - **The orbit drift is purely visual**; it doesn't prevent you from using outer-ring capsules. It's a **transparency tool**: "This shows what you care about."

3. **Why It's Empty**:
   - No capsules exist yet. You haven't authored any poses, transitions, or character definitions.
   - The rings are vacant because there's nothing to orbit.
   - **This is intentional design.** You are not starting with pre-loaded assumptions about what "animation" should be. You define it.

#### Why Rings Instead of Lists?

Traditional animation software uses dropdown menus or flat lists ("Select a character..."). ROCA Animator uses rings because:

- **Visual memory**: your eye remembers position better than text in a list.
- **Emergent organization**: as you work, the rings naturally cluster your workflow—frequently-used capsules drift inward, creating a visual "home" for your style.
- **Spatial metaphor**: orbits around a core feel intuitive to artists and animators.

---

### 2.2 The Timeline (Bottom)

Below the orbital map is a horizontal **timeline** panel that shows your animation clips, frame ruler, and playhead.

#### Timeline Anatomy

```
┌─────────────────────────────────────────────────────────────┐
│  Frame:  0   10   20   30   40   50   60   70   80   90     │  Ruler (top)
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Hero (Stand→Walk)  ███████████████████████                 │  Track 1 (Capsule clip)
│  Frame 0–20: Stand pose blending to Contact pose            │
│  Source: Stand, Contact, Stand→Walk transition, Hero char   │
│                                                              │
│  Hero (Walk Cycle)  ██████████████████████████              │  Track 2 (Capsule clip)
│  Frame 20–45: Repeating walk loop                           │
│  Source: Walk Cycle, Hero char                              │
│                                                              │
│  |———————— Playhead (red vertical bar at frame 0)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Key Features

- **Frame Ruler** (top): Numbers (0, 10, 20, ...) mark frame positions. Zoom in/out to see more or fewer frames.
- **Playhead** (red vertical line): Marks the current frame. Click and drag to scrub through the timeline.
- **Track Area**: Each animation clip is a horizontal colored bar. Hover to see which capsules generated it.
- **Capsule Badge** (on hover): Shows which capsules created this clip (e.g., "Hero + Stand→Walk + default timing").

#### Right Now (Empty Timeline)

The timeline is blank because:
- No animation clips have been generated yet.
- No capsules have been used to synthesize frames.
- The frame ruler is visible, but the playhead has no motion to preview.

---

### 2.3 The Toolbar (Top Left)

At the top of the drawing area, you see two primary buttons:

- **[Draw]**: Enter drawing mode. Sketch and paint on the canvas.
- **[Capsule]**: Create or manage capsules (poses, transitions, timing profiles, cycles, characters).

Click either to enter that mode.

---

### 2.4 The Control Panel (Right Side)

On the right side are additional buttons:

| Button | Purpose |
| --- | --- |
| **New Frame** | Add a new blank frame to the timeline |
| **Train** | (Legacy) Refine a neural model. In ROCA, this is offline. |
| **In-Between** | Generate in-betweens from start/end poses and a transition |
| **Save** | Save your project (capsules, timeline, drawings) to disk |
| **Play** | Play back the animation at current playback speed |
| **Add Layer** | Add a new animation layer (for multi-pass work) |
| **Del Layer** | Delete the current layer |
| **Undo** | Undo the last action |
| **Redo** | Redo the last undone action |
| **Import** | Import external assets (images, SVG, motion capture) |
| **Load Images** | Load a sequence of image frames for reference or conversion |
| **Load Video** | Load a video file for reference or frame extraction |
| **Export SVG** | Export the current frame as scalable vector |
| **Export Video** | Export the timeline as a video file |
| **New Capsule** | Quick shortcut to create a new capsule |

Also visible:
- **Brush Size Slider**: Adjust drawing brush thickness (1–50 pixels).
- **Playback Speed Slider**: Adjust animation playback rate (1–60 fps).

---

## Part 3: Your First Animation (Step by Step)

### Step 0: Drag the Sketch In (No Commitment)

#### Animator Action

You have a sketch of a character—maybe a humanoid figure laying next to a pool, or a dragon in mid-flight. You don't want to commit to animating yet. You just want **ROCA to know this character exists**.

Simply:

1. **Drag the sketch image** from your file browser or desktop.
2. **Drop it anywhere** on the ROCA window (canvas area or timeline).

#### Animator Intention

*"I don't want to animate yet. I just want ROCA to know this character exists so I can refer to it later."*

You're not saying "this is Hero, the main character." You're just saying "I have this sketch." ROCA should remember it, but with zero pressure to classify or commit.

#### What ROCA Does (Silently)

Behind the scenes:

1. **Creates a temporary capsule**:
   - **Type**: Unclassified (not yet assigned to a kind)
   - **Pose vector**: A symbolic heuristic derived from the image hash (consistent across sessions)
   - **Asset hash**: A character-consistent signature computed from the image pixels (deterministic, stable)

2. **Places it in a gray orbit ring** labeled:
   - **"Unassigned Capsules"** (a special lane for incoming assets)

3. **Records metadata**:
   - Filename / image path
   - Timestamp of import
   - Visual thumbnail
   - No name, no formal classification yet

#### What's NOT Happening

- ✗ No timeline pollution (clip is not added to the timeline)
- ✗ No forced naming ("Unnamed_1", "Unnamed_2" clutter)
- ✗ No neural processing or feature extraction
- ✗ No commitment required

#### Next: Refine When Ready

Later, when you're ready to animate this character:

1. **Click on the unassigned capsule** in the gray ring.
2. **Name it**: "Hero", "Dragon", "Sidekick", etc.
3. **Classify it**: "Character"
4. **Add metadata**: height, rig type, proportions
5. **Confirm**.

The temporary capsule becomes a formal **Character capsule**, moves to the Character lane, and is ready for animation.

Or, if you change your mind about this sketch:

1. **Right-click** on the unassigned capsule.
2. **Delete** (or archive to the experimental ring).

**No regrets. No confusion. Just gentle, non-committal asset accumulation.**

---

### Step 1: ROCA Asks One Gentle Question (Not a Chatbot)

A small **inspector hint** appears in the bottom-right corner of the screen. It's **not modal** (you can ignore it). It's **not annoying** (it doesn't demand attention). It's just a quiet suggestion:

```
┌─────────────────────────────────┐
│  New capsule detected.           │
│  What would you like to do?      │
├─────────────────────────────────┤
│  ○ Assign Pose                  │
│  ○ Assign Character             │
│  ✅ Just Store (Learn Later)     │
│                                 │
│  [Apply]                        │
└─────────────────────────────────┘
```

#### Three Gentle Options

1. **Assign Pose**: "This image is a specific pose (Standing, Jump, etc.). Classify it now."
2. **Assign Character**: "This image is a new character rig. Let me define its anatomy."
3. **Just Store (Learn Later)**: "I'm not sure yet. Remember this sketch, but don't ask me to classify it."

#### Animator Chooses: Just Store (Learn Later)

You click **"Just Store (Learn Later)"** and press **[Apply]**.

#### What ROCA Does

1. **Confirms**: "Got it. I'm holding this sketch in the Unassigned Capsules ring."
2. **The hint disappears** (no nagging).
3. **You return to your work** without any cognitive load.

#### Why This Matters

**ROCA respects not wanting to think yet.**

Many animation tools force you to commit immediately:
- "Name this character now."
- "Choose a rig type."
- "Define the skeleton."

ROCA takes the opposite approach:

- **The sketch exists.** ROCA knows about it.
- **You don't have to name it.** You don't have to classify it.
- **Come back to it whenever you're ready.** When inspiration strikes, you formalize it. Until then, it's just a sketch in your library.

This respects the actual creative process:
- Sometimes you import a reference without knowing what it is yet.
- Sometimes you gather assets first, organize later.
- Sometimes you're still deciding between "main character" and "background extra."

**ROCA doesn't force premature decisions. It just remembers.**

---

### Step 2: Create a Character Capsule

The first thing you need is a **character**. Characters are the actors in your animation—humanoids, animals, robots, whatever.

1. Click the **[Capsule]** button (top left toolbar).

A dialog box appears:

```
┌─────────────────────────────────────────────┐
│  New Capsule                                │
├─────────────────────────────────────────────┤
│  Name:              [_________________]     │
│                                             │
│  Kind:              [v] (dropdown)          │
│                    ┌─────────────────────┐ │
│                    │ Pose                │ │
│                    │ Transition          │ │
│                    │ Timing              │ │
│                    │ Cycle               │ │
│                    │ Character           │ │
│                    └─────────────────────┘ │
│                                             │
│  [Advanced Metadata Options...]             │
│                                             │
│  [Cancel]  [Create]                         │
└─────────────────────────────────────────────┘
```

2. **Name**: Type `Hero` (or your character name).
3. **Kind**: Select `Character` from the dropdown.
4. Click **[Advanced Metadata Options...]** to set defaults:
   - **Height**: 180 cm (default humanoid)
   - **Rig Type**: Humanoid (standard skeleton with joints)
   - **Proportions**: Athletic (preset limb ratios)
   - **Silhouette Scale**: 1.0 (neutral; affects visual orbit size)

5. Click **[Create]**.

#### What Just Happened

ROCA Animator has created a new **Character capsule** named "Hero" with the following attributes:

- **UUID**: A unique, deterministic identifier derived from the name ("Hero" + "Character" kind). Same name, same ID, forever.
- **Pose Vector**: An 8D deterministic vector generated from the UUID5 seed. This vector represents "Hero" in the symbolic space.
- **Metadata**: Stored information about height, rig, proportions—everything needed to retarget animations to this character.
- **Orbit Position**: Placed in the **outer character ring** (because it's brand new, `orbit_score = 0`).
- **Use Count**: 0 (no animations created with Hero yet).
- **Created Timestamp**: Current date and time.

On the orbital map, you should now see a small **colored dot** (likely blue or teal) appear in the outermost character lane. This is your "Hero" capsule, freshly spawned.

---

### Step 3: Create Pose Capsules

Now you need some **poses**—keyframe positions that define your character's state at important moments.

Let's create two poses: "Standing" and "Contact" (foot hitting ground in a walk).

#### Creating the First Pose

1. Click **[Capsule]** again.
2. **Name**: `Standing`
3. **Kind**: `Pose`
4. **Advanced**:
   - **Character Target**: `Hero` (this pose applies to your Hero character)
   - **Skeleton**: (auto-loaded from Hero rig)
   - **Description**: "Standing upright, weight neutral, arms at rest"
   - (In a real session, you'd draw or import the skeleton positions on the canvas)
5. Click **[Create]**.

You now have a "Standing" pose capsule in the outer pose ring.

#### Creating the Second Pose

1. Click **[Capsule]** again.
2. **Name**: `Contact`
3. **Kind**: `Pose`
4. **Advanced**:
   - **Character Target**: `Hero`
   - **Description**: "Contact frame of walk: foot hits ground, weight forward, slight knee bend"
5. Click **[Create]**.

**Result**: Two pose capsules are now on the orbital map, both in the outer pose ring (unused).

---

### Step 4: Animator Keeps Dumping Images (No Pressure)

#### Animator Action

You now have momentum. You keep finding images of this character:

- Same character **sitting** on the pool edge
- Same character **standing** on a diving board
- Same character **from another angle** (side view vs. front view)
- Same character **clothed differently** (summer clothes, winter gear)

You drag all of these into ROCA. One by one. No pause for classification. No "what kind of pose is this?"—just dump.

```
Drag → Drop → Drag → Drop → Drag → Drop → Drag → Drop
```

Zero friction. Zero naming. Just accumulate.

#### Animator Intention

*"I'm loading reference material. ROCA should learn that these are all variations of the same character. I'm not ready to animate yet—I'm still gathering reference."*

You're building a **capsule network**, not an animation. You're saying: "Here are all the ways I see this character. Remember these variations."

#### ROCA Behavior (Automatic Clustering)

Behind the scenes, ROCA is **silently intelligent**:

1. **Computes symbolic similarity** for each image:
   - Pixel-level hash (deterministic, character-consistent)
   - Skeletal topology inference (is this a humanoid? quadruped? abstract?)
   - Pose signature (is this standing? sitting? dynamic?)

2. **Clusters automatically**:
   - **Same character signature** → same "character gravity" (all variants orbit together)
   - **Different poses** → nearby **pose satellites** (cluster as sub-variations around the character)

   ```
   Orbital Map Update:
   
                    ┌────────────────────┐
                    │                    │
                    │  "Character Cluster"│
                    │  ╔════════════════╗ │
                    │  ║  Sitting       ║ │
                    │  ║  Standing      ║ │  All orbit together
                    │  ║  Side-view     ║ │  as variations of
                    │  ║  Winter gear   ║ │  the same character
                    │  ╚════════════════╝ │
                    │         ↓           │
                    │   Character Core    │  (unnamed, unclassified)
                    │   (No name yet)     │
                    │                    │
                    └────────────────────┘
   ```

3. **Records lineage**:
   - All variants know they're derived from the same character.
   - Pose sub-variants are tagged as "sitting," "standing," etc. (soft tags, not rigid classifications).
   - No formal names required.

4. **Builds co-activation hints**:
   - Automatically learns: "This character is often paired with Sitting pose" or "Standing pose is most common."
   - These hints inform later suggestions (e.g., "Default to standing for this character").

#### What's NOT Happening

- ✗ No neural feature extraction (slow, power-hungry)
- ✗ No training or fine-tuning
- ✗ No forced naming ("Sitting_1", "Standing_2" pollution)
- ✗ No animation yet
- ✗ No timeline inflation (clips are not added)

**You're loading the capsule network, not animating.**

#### Why This Matters

This is where ROCA differs fundamentally from traditional animation tools:

- **Photoshop/Toon Boom**: "Import image." Then what? You organize manually in folders or layers.
- **Neural tools**: "Upload reference." System trains a model (slow, requires GPU, risk of overfitting).
- **ROCA**: "Dump images." System silently clusters, learns character signature, offers you structure without asking.

By the time you're ready to animate, ROCA has already organized your reference material and understands which poses belong to which character. **You didn't lift a finger for organization.**

---

### Step 5: First Intentional Action (Naming, Later)

#### The Inspector Hint

At some point, maybe after 10 minutes of image dumping, you glance at the **inspector panel** on the right side of the screen. A hint appears:

```
┌──────────────────────────────┐
│  Pose Cluster Analysis       │
├──────────────────────────────┤
│  pose_x2 (appears 12 times)  │
│  └─ Cluster size: 12         │
│  └─ Similarity: 0.94         │
│  └─ Suggested kind: Pose     │
│                              │
│  [Describe This Pose]        │
└──────────────────────────────┘
```

The system has noticed a **cluster of 12 similar poses** and is asking: *"Do you want to name this?"*

#### Animator Action

You click **[Describe This Pose]** and type:

```
lying_down
```

Just one word. No metadata. No formal definition. Just a name that means something to you.

#### The Meaning Lock

The moment you press Enter, something profound happens:

**ROCA locks in meaning.**

The system:

1. **Renames the cluster**: All 12 images that were labeled "pose_x2" are now labeled "lying_down_1", "lying_down_2", ..., "lying_down_12".

2. **Replaces everywhere**: Any capsule, any edge, any reference to "pose_x2" is now "lying_down". Timeline clips that reference it now show "lying_down" instead of a cryptic hash.

3. **Builds symbolic anchor**: "lying_down" is now a persistent, deterministic identifier. Future image drops will cluster around this pose signature.

4. **Informs future drops**: When you drag in another image, ROCA compares it to "lying_down". If similarity > threshold, it automatically groups it there. No asking, no manual organization.

#### What's NOT Happening

- ✗ No retraining (zero neural ops)
- ✗ No reprocessing of existing images (no new embeddings computed)
- ✗ No timeline updates or re-rendering
- ✗ No GPU involved

**Pure symbolic rewrite.**

The name "lying_down" is a label. The pose signature (the 8D symbolic vector) was already there from the first drop. Naming just makes it queryable and referenceable.

#### Why This Matters

This is the **first meaningful animator decision**. Not "import data and let the system decide." Not "manually classify everything upfront." But: **"I've accumulated enough reference that I can now give names to patterns I see."**

ROCA respects this workflow:

1. **Gather without pressure** (Steps 0–4)
2. **Name when you're ready** (Step 5)
3. **Build from there** (Steps 6+)

No tool demands premature classification. ROCA does the boring work (clustering, organizing) silently. When you have something to say about the patterns, you say it, and the system instantly makes it real.

---

### Step 6: Character Identity (Optional, Same Way)

#### The Same Pattern, But for Characters

After naming a few poses ("lying_down", "standing", "running"), you turn attention to the **character cluster** itself. In the orbital map, you see a cluster of images labeled:

```
character_x1 (appears 47 times)
└─ Cluster size: 47
└─ Similarity: 0.92
└─ Suggested kind: Character
```

All 47 variations of your main actor (different clothes, different angles, different scenes) are orbiting together, but without a name.

#### Animator Action

You click the cluster and type:

```
main_character
```

#### The Character Gravity Well

Instantly:

1. **The cluster is named**: "character_x1" → "main_character"

2. **Everything reorders**: All 47 images now belong to "main_character". The orbital map reorganizes to place "main_character" at the center of its character lane.

3. **Satellites form**: Other assets automatically cluster around this character:
   - **Clothing variations** → sub-cluster: "main_character__outfit_summer", "main_character__outfit_winter"
   - **Props the character holds** → sub-cluster: "main_character__prop_book", "main_character__prop_sword"
   - **Environments where character appears** → sub-cluster: "main_character__env_pool", "main_character__env_forest"

4. **Future drops snap here**: From now on, when you drag in a new image, ROCA compares it to "main_character". If similar, it auto-groups it there. If it's a new outfit, it creates a sub-capsule. If it's a new prop, it clusters appropriately.

#### The Gravity Well Metaphor

Think of it like gravity:

```
                    ○ clothing_variant_1
                   / \
                  /   \
                 /     \
            ● main_character (center) ←── All satellites orbit here
                 \     /
                  \   /
                   \ /
                    ○ prop_sword
                    
                    ○ environment_pool
```

**main_character** is the gravitational center. Everything related to this character naturally orbits inward toward it. Unrelated characters stay in their own orbit zones.

#### Why This Matters

Without naming, you have 47 indistinguishable images. With naming, you have:

- A clear protagonist
- A structured library of their variations
- Automatic organization for related assets
- Instant queries: "Show me all main_character images" or "Show me all main_character in summer outfit"

And again: **no retraining, no reprocessing, no neural overhead**. Just symbolic rewrite.

---

### Step 7: Still No Animation Required

#### Important Rule: ROCA Never Forces Animation Mode

At this stage of the workflow, take a step back. Look at your screen:

- **Orbital map**: Densely populated with named capsules (poses, characters, environments).
- **Timeline**: Still empty. No clips. No playback.
- **Inspector**: Shows you've named 12 pose clusters, 4 character clusters, organized 200+ images.

You have done **significant work**. You've built a capsule library. You've established symbolic anchors. You've created a reusable motion library.

**And you haven't animated a single frame.**

#### Why This Matters

Most animation tools follow a linear path:

```
Import → Organize → Animate → Render → Export
```

ROCA breaks this assumption. You can:

```
Import → Organize → Pause indefinitely → Animate when ready
```

Or:

```
Import → Organize → Share library with team → They animate
```

Or:

```
Import → Organize → Use for storyboarding → Never animate in this tool
```

#### The Animator's Choice

At this point, you have several options:

1. **Keep dumping reference** (stay in Step 4): You're not ready to animate yet. That's fine. Keep gathering until you have comprehensive reference.

2. **Name more clusters** (stay in Steps 5–6): You want to establish more symbolic anchors before committing to animation. Organize your entire visual library first.

3. **Create transition capsules** (move to Step 8): You're ready to define "how" poses connect. Encode breakdowns, timing intent, weight shifts.

4. **Generate animation** (move to Step 9+): You have enough poses and transitions to synthesize motion. Time to animate.

5. **Export the library** (skip to Part 6): You've done this work, but you want to share capsules with a teammate who will animate. Export and hand off.

6. **Close and come back later** (take a break): ROCA persists everything. Your capsule state, your naming, your organization. Close the app and return when inspiration strikes.

#### What ROCA Does NOT Do

- ✗ Does not nag: "You've imported images, so you must animate now."
- ✗ Does not auto-generate motion: "Here's a walk cycle. Use it or I'll assume you forgot."
- ✗ Does not penalize pausing: Stopping here doesn't corrupt your library or force retraining.
- ✗ Does not require timeline use: The timeline is optional, not mandatory.

**The capsule library you've built is a complete, valid artifact by itself.** You can hand it off, archive it, share it, or leave it untouched for months. It's yours.

#### The Paradigm Shift

Traditional tools treat assets as **dependencies of animation**:
- "You import images so you can animate them."
- "Organizing is just a step toward animation."

ROCA treats assets as **first-class objects in their own right**:
- "You import images because you want to remember them."
- "Organizing is valuable independent of whether you animate."
- "The capsule library is the deliverable; animation is optional."

This respects the reality of creative teams:

- Concept artists gather reference and hand off to animators.
- Animators curate motion libraries used across 10 projects.
- Supervisors audit capsule organization for quality consistency.
- Leads establish shared symbolic anchors for faster onboarding.

**Animation is not the goal. Organized, reusable, persistent motion knowledge is the goal. Animation is one way to express and test that knowledge.**

---

## Mental Model for the Animator

To understand ROCA's philosophy, remember these four metaphors:

### 1. Dragging Images = Feeding Gravity

When you drag images into ROCA, you're not "importing files." You're **feeding gravity**.

Each image is a reference point in symbolic space. ROCA clusters similar images around the same "gravitational well." The more images you feed, the stronger the gravity well becomes. The well holds meaning not through neural learning, but through sheer accumulation and resonance.

**Metaphor**: *Think of dragging images like dropping stones into a pond. ROCA watches the ripples and groups stones that ripple the same way.*

**Real work**: You're not committing to anything. You're exploring. You're saying: "I have 50 variations of this character. ROCA, remember that they're related."

---

### 2. Naming Capsules = Declaring Meaning

When you name a cluster ("lying_down", "main_character", "cartoony_style"), you're not classifying data. You're **declaring meaning**.

The name is a commitment to human language. It's saying: "This cluster has semantic weight. From now on, I want to refer to it by this name."

ROCA takes that declaration seriously. It rewires everything—every reference, every future drop, every suggestion—around that name. The name becomes a **gravitational anchor** that pulls future work toward it.

**Metaphor**: *Think of naming like drawing a boundary on a map and labeling it. Before naming, it was just a region. After naming, it's a place with identity.*

**Real work**: You're not being forced to decide. You're choosing to decide. When you have enough context to name something meaningfully, you declare it. Until then, you don't.

---

### 3. Timeline = Optional Commitment

The timeline is not the main event. It's an **optional commitment** to sequential motion.

You can build a complete, beautiful capsule library without ever opening the timeline. You can organize, cluster, name, and share without rendering a single frame. The capsule network is self-sufficient.

The timeline exists for when you want to **actualize** the capsules—turn symbolic structure into kinetic motion. But that's a separate choice, not a requirement.

**Metaphor**: *Think of the timeline like a stage. The capsule library is the full script, costumes, lighting, and crew. The stage is where you perform the script if you choose to.*

**Real work**: You decide when to animate. Not the tool. The tool never says: "You have 200 images organized, so you must now generate motion." You decide if and when motion synthesis adds value to your work.

---

### 4. In-Betweens = Emerge Later, Not Forced

In-betweens are not generated upfront. They **emerge later**, on demand, from the capsule structure you've built.

You create poses. You create transitions. The system holds that structure symbolically. When you want motion, you ask: "Generate in-betweens from Pose A to Pose B using Transition XYZ." The frames synthesize deterministically from that request.

No pre-computation. No guessing. No neural overhead. Just: *You describe what you want, the symbolic graph generates it.*

**Metaphor**: *Think of in-betweens like interpolation in geometry. You define key points (poses). The system fills the space between them (in-betweens) using the rules you've specified (transitions).*

**Real work**: You're not waiting for the tool to "figure out" motion. You're saying: "Here's my intent, please synthesize frames consistent with it." The tool answers: "Done. Same answer every time you ask."

---

## The Operating Philosophy

Combine these four metaphors and you get ROCA's operating philosophy:

| Stage | Metaphor | Your Role | Tool's Role |
| --- | --- | --- | --- |
| **Gathering** | Feed gravity | Dump reference without judgment | Silently cluster and organize |
| **Naming** | Declare meaning | Decide when patterns have names | Rewrite structure around names |
| **Organizing** | Build library | Establish symbolic anchors | Offer structure, never enforce |
| **Animating** | Actualize on timeline | Request motion from structure | Synthesize frames deterministically |

**Key insight**: Each stage is complete and valuable by itself. You can stop after gathering. You can stop after naming. You can build a library and never animate. ROCA doesn't judge. It just enables the next stage when you choose it.

---

## Why This Is Correct (and Rare)

### Traditional Animation Tool Order

Most animation tools force you down a rigid pipeline:

```
Import → Rig → Animate → Fix → Regret
```

Here's what happens:

1. **Import**: You drag in a character sketch. The tool demands: "Now I need to rig this. Where are the joints?"
2. **Rig**: You spend 2 hours setting up a skeleton. You're committed now.
3. **Animate**: You generate walk cycles. Some look good, some don't.
4. **Fix**: You tweak keyframes, adjust easing, re-rig parts that don't work.
5. **Regret**: After 6 hours, you realize you want to change the character's pose structure. You're locked in. Start over.

The tool assumes: *"You import so you can rig. You rig so you can animate. Animation is the only valid endpoint."*

### ROCA's Actual Order

ROCA respects how animators actually think:

```
Observe → Accumulate → Name → Animate → Reuse
```

Here's what happens:

1. **Observe**: You gather reference without pressure. Lying down. Standing. Sitting. Different angles, different clothes.
2. **Accumulate**: You dump 100+ images. ROCA silently clusters them. No commitment.
3. **Name**: When patterns become clear, you name them: "lying_down", "main_character", "summer_outfit".
4. **Animate**: You decide if motion synthesis is worth doing. You might not animate. You might just share the capsule library.
5. **Reuse**: The capsule library persists. You use it in the next project. It compounds across years.

The assumption: *"You observe so you understand. You accumulate so you have options. You name so you can reference. You animate when motion adds value. You reuse because the library is permanent."*

### Why This Matters

#### Traditional tools treat animation as mandatory

```
"You imported images."
↓
"So you must rig."
↓
"So you must animate."
↓
"So you must render."
```

Each stage forces the next. Each stage requires commitment. Changing direction means rework.

#### ROCA treats organization as primary

```
"You want to organize visual knowledge."
↓
"Animation is one way to test that knowledge."
↓
"But sharing, auditing, and iterating are equally valid."
```

Animation is optional. Organization is essential.

### Why It's Not a Chatbot

A chatbot (like ChatGPT, Claude, etc.) works like this:

```
You ask: "Generate a walk cycle for a character."
↓
System: "Here's a walk cycle." (one-shot, no history)
↓
You ask: "Make it bouncier."
↓
System: "Here's a bouncier walk cycle." (forgets everything before)
```

No memory. No learning. No accumulation. Each request is isolated.

**ROCA is the opposite.** Every image you drag, every name you declare, every pose you create **persists forever**. Future drops cluster around existing capsules. Future names build on previous anchors. Future animations reuse the library you've built.

A chatbot says: "Ask me again, I'll generate something new."

ROCA says: "Everything you've ever told me stays, and builds on what came before."

### Why It's Not a Diffusion Model

A diffusion model (like Stable Diffusion, DALL-E, etc.) works like this:

```
You ask: "Generate a character walking."
↓
System: Runs 50 denoising steps, consuming GPU power
↓
Output: One image (or a few stochastic variations)
↓
You ask: "Generate a character walking, but in a different pose."
↓
System: Runs 50 denoising steps again (no memory of first generation)
↓
Output: Probably inconsistent with the first image
```

No guarantee of consistency. No learning from your feedback. No permanent library. Each generation is expensive (GPU, time, power).

**ROCA is the opposite.** Every pose, every transition, every timing profile is **deterministic**. Ask for the same animation 100 times, get 100 identical results. No GPU. No stochasticity. No drift.

A diffusion model says: "I'll generate something probabilistic and expensive."

ROCA says: "I'll synthesize something deterministic and cheap, based on your explicit intent."

### Why It Respects How Animators Actually Think

Animators don't think like chatbots or diffusion models. Animators think like this:

1. **"I need to understand this character."** (Gathering phase)
   - Look at reference from many angles.
   - Find the pose that feels "right."
   - Accumulate variations without naming them yet.

2. **"Now I see the pattern."** (Naming phase)
   - "That's the idle pose. This is the contact frame."
   - Give names to clusters.
   - Declare meaning.

3. **"How should this character move?"** (Intent phase)
   - Decide if this character is snappy, floaty, mechanical, organic.
   - Define transitions: "Stand to walk should have anticipation."
   - Write rules, not parameters.

4. **"Synthesize motion."** (Actualization phase)
   - Ask ROCA: "Generate frames from Standing to Walking using anticipation transition."
   - Review. Tweak if needed.
   - Lock it in. It will never change.

5. **"Use this forever."** (Reuse phase)
   - Apply the same transition to a different character.
   - Re-target to a new rig.
   - Share the capsule library with a teammate.
   - Build it across decades of work.

**ROCA mirrors this exactly.** It says: "Gather, name, decide, synthesize, reuse." Not "generate randomly and forget." Not "train a model and hope." Not "ask me again, I'll make something different."

### The Rare Part

**Most tools don't let you do this.** They skip straight to animation. They lock you into rigs before you've understood the character. They force generation to be the primary interaction.

ROCA is rare because it respects **accumulation, persistence, and choice**. It lets you build before you commit. It lets you organize before you animate. It lets you share before you finish.

That's why animators, concept artists, and animation supervisors are drawn to it. It matches how they actually work.

---

### Step 8: Create a Transition Capsule

A **transition** capsule defines *how* to move from one pose to another. It encodes animator intent: which joints move first, how the weight shifts, whether there's anticipation.

1. Click **[Capsule]** again.
2. **Name**: `Standing_to_Contact`
3. **Kind**: `Transition`
4. **Advanced**:
   - **From Pose**: `Standing`
   - **To Pose**: `Contact`
   - **Timing Intent**: "Snappy" (cartoony feel, strong ease-out; could also be "Smooth," "Floaty," "Mechanical," etc.)
   - **Arc Intent**: "Weight shifts forward, knee bends, anticipation before foot plants"
   - **Duration**: 10 frames (default)
5. Click **[Create]**.

**Result**: A "Standing_to_Contact" transition capsule appears in the outer transition ring.

---

### Step 9: Generate Animation Frames

Now let's use all these capsules to actually create animation.

1. Click **[In-Between]** button (on the right panel) or **[Capsule]** → select "Generate In-Betweens" mode.

A panel appears:

```
┌─────────────────────────────────────────────┐
│  Generate In-Betweens                       │
├─────────────────────────────────────────────┤
│  Character:  [Hero ▼]                       │
│  Start Pose: [Standing ▼]                   │
│  End Pose:   [Contact ▼]                    │
│  Transition: [Standing_to_Contact ▼]        │
│  Timing:     [Default ▼]                    │
│  Frame Range: [0] to [20]                   │
│  Number of Frames: 20                       │
│                                             │
│  [Preview]  [Generate]                      │
└─────────────────────────────────────────────┘
```

2. Settings:
   - **Character**: `Hero` (already selected)
   - **Start Pose**: `Standing`
   - **End Pose**: `Contact`
   - **Transition**: `Standing_to_Contact`
   - **Timing**: `Default` (simple linear ease with gentle slow-in/slow-out)
   - **Frame Range**: `0 to 20`

3. Click **[Generate]**.

#### What Happens Behind the Scenes

ROCA Animator routes through the symbolic graph:

1. **Load capsules**:
   - Hero (64D character vector)
   - Standing pose (32D skeleton state)
   - Contact pose (32D skeleton state)
   - Standing_to_Contact transition (32D breakdown intent)
   - Default timing profile (16D ease curve)

2. **Agreement check**: Verify that these capsules have been tested together. Since this is your first use, there's no historical data, so the system notes: "Untested pairing; proceed with caution" (non-blocking warning).

3. **Retarget** (if needed): Apply character parameters to scale the transition appropriately for Hero's height and rig.

4. **Synthesize frames** deterministically:
   - Frame 0: Standing pose (source capsule: Standing)
   - Frame 10: Mid-blend (interpolated)
   - Frame 20: Contact pose (source capsule: Contact)
   - **Key property**: Every output frame is traceable to its source capsules. No randomness, no neural sampling. Run this same generation 100 times, get 100 identical results.

5. **Record event**: ROCA logs a `UseEvent`:
   - Source capsules: Hero, Standing, Contact, Standing_to_Contact, Default timing
   - Output frames: 20
   - Timestamp: now

6. **Update orbital scores**:
   - Hero: `orbit_score += 1` (used once)
   - Standing: `orbit_score += 1`
   - Contact: `orbit_score += 1`
   - Standing_to_Contact: `orbit_score += 1`

---

### Step 10: Preview on the Timeline

After generation, the **timeline** updates to show your animation clip:

```
┌─────────────────────────────────────────────────────────────┐
│  Frame:  0   10   20   30   40   50   60   70   80   90     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Hero (Standing→Contact)  ████████████████████             │  (Blue bar, frames 0–20)
│  Capsules: Standing, Contact, Standing_to_Contact, Hero     │
│                                                              │
│  |———— Playhead (at frame 0)                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

- The **blue bar** (or whatever color the system chose) represents your 20-frame animation clip.
- **Hover** over the bar to see the full capsule names and timing info.
- **Click** on the bar to select it for editing.

#### Watching the Animation

1. Click **[Play]** (on the right panel).
2. The playhead moves across the timeline from left to right.
3. You see the animation: Standing pose smoothly transitioning to Contact pose over 20 frames.

---

### Step 11: Watch the Orbital Map Update

Look back at the orbital map. Notice:

- **Hero** has drifted **slightly inward** from the outermost character ring. Not much (because it's only been used once), but visibly closer to the center.
- **Standing**, **Contact**, and **Standing_to_Contact** have also drifted inward slightly.

This is the **emergence of your workflow in the rings**. As you continue:

- Capsules you use repeatedly drift steadily inward.
- Your most-trusted, frequently-used capsules eventually reach the **inner rings**.
- Experimental or rarely-used capsules drift outward toward the **experimental band**.

**The orbital map is your visual memory of what you care about.**

---

## Part 4: Core Concepts

### 4.1 Capsules Are Explicit, Persistent Objects

Every pose, transition, timing profile, and character is a **named, UUID-identified capsule**. When you save your project and close the app, the capsules are stored in `ringpose_state.json`. When you reload the project hours, days, or months later:

- "Hero" is still "Hero" (same UUID, same pose vector).
- "Standing" is still "Standing".
- All capsule metadata and use counts are preserved.

This means **perfect consistency across sessions**. Unlike neural models (which drift with weight updates), ROCA capsules are deterministic and stable.

---

### 4.2 No Neural Learning

Unlike neural in-betweening (which trains a model and refines it iteratively), ROCA Animator:

- **Does not train models**.
- **Does not have learned weights** that drift or degrade.
- **Does not require GPU** for authoring (rendering can optionally use GPU, but authoring is CPU-only).
- **Generates the same frame sequence every time** you run the same animation.

How? **Deterministic UUID5 seeding**. Each capsule name + kind produces a stable seed. That seed generates an 8D pose vector via pseudo-random number generation (RNG). Same seed → same pose vector, always.

### Example Reproducibility

```
Session 1 (Week 1):
  create_symbolic_capsule("JumpApex", "Pose")
  → Generates 8D vector: [0.42, -0.18, 0.79, 0.33, -0.51, 0.09, 0.12, -0.44]

Session 2 (Month later):
  create_symbolic_capsule("JumpApex", "Pose")
  → Generates same 8D vector: [0.42, -0.18, 0.79, 0.33, -0.51, 0.09, 0.12, -0.44]
  
Guarantee: Frame 5 in Session 1 is bit-identical to Frame 5 in Session 2.
```

---

### 4.3 The Orbital Map Is Living Memory

As you author more capsules and use them, the orbital map evolves:

- **Frequently-used capsules**: orbit inward (inner rings, close to ROCA Earth).
- **Rarely-used capsules**: orbit outward (outer rings, far from center).
- **New capsules**: spawn in the outer rings.
- **Archived capsules**: drift further out as decay ticks pass without use.

**Important**: Orbit distance is purely visual. Routing does not depend on how close a capsule is to the center. An outer-ring capsule is just as accessible as an inner-ring capsule. The visual drift is a **transparency tool**—a way to see "What do I actually use?"

Over months and years:

- Your core animator identity (Nucleus) stays fixed: your preferred timing, color palettes, personality.
- Your workhorse capsules (walk cycles, character bodies, reliable transitions) drift inward.
- Experimental or project-specific capsules drift outward.

**Result**: A living, evolving map of your creative practice.

---

### 4.4 Agreement Tracking (Capsule Pairings)

When two capsules are used together (e.g., "Hero" + "Standing_to_Contact"), ROCA Animator records an **edge** in a capsule graph with an **agreement score**.

- **High agreement** (0.8–1.0): Capsules have been paired many times and always worked well together.
- **Medium agreement** (0.4–0.7): Capsules have been paired a few times with mixed results.
- **Low/untested agreement** (0.0–0.3): Capsules have never been paired or have conflicted.

When you try a new pairing, the system warns: *"This combination has never been used together. Proceed with caution?"* This helps you avoid mistakes (e.g., accidentally using a "tiny character" capsule with a "giant stride" walk cycle).

Over time, as you pair capsules successfully, the agreement scores rise, and the system builds a map of what works well together. This enables **smart suggestions**:

- "You frequently pair Hero with WalkCycle. Try this next?"
- "Standing_to_Contact works best with SnappyTiming; gentle suggestion?"

---

### 4.5 Deterministic Seeding (Why Frames Are Reproducible)

ROCA Animator uses **UUID5 deterministic seeding** to ensure reproducible animation:

```python
def deterministic_seed_for_capsule(capsule_name: str, frame_index: int) -> int:
    """Generate a stable seed for a specific capsule at a specific frame."""
    combined = f"{capsule_name}#{frame_index}"
    u = uuid.uuid5(uuid.NAMESPACE_DNS, combined)
    return int(u.int % (2**31))

# Example:
seed = deterministic_seed_for_capsule("Standing_to_Contact", frame=10)
# Always returns the same integer for "Standing_to_Contact" at frame 10
```

This seed is passed to a pseudo-random number generator (RNG) to add micro-variations (breathing, sway, subtle noise) while ensuring identical output across sessions.

**Implication**: You can replay your entire animation session from a log of events, and every frame will be bit-identical. No model drift, no weight shifts, no mysterious inconsistencies.

---

## Part 5: Expanding Your Animation Library

### 5.1 Creating More Poses

After your first animation, you'll want to create a library of poses:

- **Standing**
- **Contact Left** (left foot hitting ground)
- **Contact Right** (right foot hitting ground)
- **Down** (lowest point of walk cycle)
- **Pass** (feet passing each other mid-stride)
- **Reach** (arm extended forward)
- **Grab** (hand closed, holding)
- **Crouch** (bent knees, ready to jump)
- **Jump Apex** (at the height of a jump)
- **Land** (feet touching ground after jump)
- **Idle Sway** (subtle weight shift while standing)
- **Blink** (eye closed)
- **Mouth Open** (for dialogue)

Create each as a Pose capsule targeting your character. As you create more, they populate the outer pose ring and can be combined in endless ways.

---

### 5.2 Building Transitions

Transitions are the glue. Create transitions between common pose pairs:

- **Standing → Contact**
- **Contact → Down**
- **Down → Pass**
- **Pass → Contact** (other foot)
- **Contact → Up**
- **Up → Standing**
- **Standing → Crouch**
- **Crouch → Jump**
- **Jump → Apex**
- **Apex → Fall**
- **Fall → Land**
- **Land → Standing**

For each, define the **breakdown intent** (weight shifts, arcs, anticipation) and **timing** (snappy, smooth, etc.).

---

### 5.3 Creating Walk Cycles and Loops

A **cycle** capsule represents a repeating motion (walk, run, idle breathing):

1. Click **[Capsule]**.
2. **Name**: `WalkCycle`
3. **Kind**: `Cycle`
4. **Advanced**:
   - **Loop Length**: 8 frames (typical walk is 2 steps, 8–12 frames per cycle)
   - **Start Pose**: `Contact Left`
   - **Poses in Cycle**: [Contact Left, Down, Pass, Contact Right, Down, Pass, Contact Left, Down]
   - **Description**: "Full walk cycle, repeating"
5. Click **[Create]**.

Now you can animate long sequences (50 frames) by repeatedly applying the WalkCycle capsule, rather than manually creating each transition.

---

### 5.4 Re-targeting to Other Characters

One of ROCA's superpowers: **re-target animations to different characters without redoing the work**.

Suppose you create a WalkCycle for Hero. Now you want the same cycle for Sidekick (a shorter character):

1. Load WalkCycle (from the orbital map or timeline).
2. Click **[Generate In-Betweens]**.
3. **Character**: Change from `Hero` to `Sidekick`.
4. **Cycle**: Keep `WalkCycle`.
5. Click **[Generate]**.

ROCA automatically:
- Scales the stride length based on Sidekick's height.
- Adjusts joint angles to match Sidekick's proportions.
- Preserves the breakdown intent and timing from WalkCycle.

**Result**: Sidekick walks with the same motion quality as Hero, but in Sidekick's body. No re-authoring, no re-training, just parameter adjustment.

---

### 5.5 Merging Capsules (Shadow Identities)

Over time, you may accumulate near-duplicates. For example:

- "WalkCycle v1" (created early)
- "WalkCycle v2" (refined later)
- Both are 95% similar but stored as separate capsules.

ROCA lets you **merge** them while preserving full traceability:

1. Select both "WalkCycle v1" and "WalkCycle v2" (Ctrl+click in the orbital map).
2. Right-click → **Merge**.

A new merged proxy capsule is created:
- Name: "WalkCycle" (or "WalkCycle_merged")
- **Shadows**: [WalkCycle v1, WalkCycle v2] (original capsules preserved, not deleted)
- Routing prefers the merged proxy, but UI shows the originals.
- If future usage shows v1 and v2 diverging, you can **unmerge** and re-promote the originals.

**Benefit**: Cleaner library (fewer duplicates) without losing history.

---

## Part 6: Saving and Sharing

### 6.1 Save Your Project

Click **[Save]** button.

Your project is saved to a `.roca` file (JSON format) containing:

- All capsules (name, kind, pose vector, metadata)
- Timeline clips and frame data
- Coactivation graph (which capsules pair well)
- Event log (reproducible for replay)
- Drawings and annotations

The file is human-readable JSON, suitable for version control (Git) and team collaboration.

---

### 6.2 Share Capsule Libraries

Extract just the capsule definitions (no animation clips):

1. **[Export Capsules]** → `my_library.capsules.json`

This lightweight file can be shared with teammates. They can import it:

2. **[Import Capsules]** → `my_library.capsules.json`

Their orbital map is seeded with your capsules, and they can start animating immediately, reusing your pose and transition libraries.

---

## Part 7: Troubleshooting

### Issue: Timeline Stays Empty After [Generate]

**Causes**:
1. Did you select poses and transitions from the dropdowns? (Easy to miss.)
2. Is the frame range valid (start < end)?
3. Did the system warn about untested pairings? (This is not an error; proceed.)

**Fix**:
- Double-check dropdown selections.
- Try a simpler transition first (e.g., Standing to Standing with 0 frames, just to test the pipeline).
- Check the **Inspector** panel (right side) for debug output.

---

### Issue: Animation Looks Wrong or Twisted

**Possible causes**:
1. Character metadata is incomplete (missing rig information).
2. Transition capsule has a strong "arc intent" that doesn't fit the poses.
3. Timing is too snappy or too smooth for the movement.

**Fix**:
1. Verify your Hero character capsule has correct metadata (height, rig type, proportions).
2. Create a simpler transition (linear, minimal arc) and test.
3. Adjust the transition's timing and regenerate.

---

### Issue: Orbital Map Is Too Crowded

After months of work, you might have 500+ capsules. The map can feel cluttered.

**Solutions**:
1. **Merge near-duplicates** (shadow identities).
2. **Archive rarely-used capsules** (they drift to the experimental outer ring naturally; hide them if needed).
3. **Zoom in/out** on the orbital map for different views (e.g., "show me only characters and skills").
4. Use **search/filter** to find specific capsules by name or tag.

---

### Issue: I Want to Undo a Merge

No problem. ROCA preserves shadow identities.

1. Select the merged proxy capsule.
2. Right-click → **Unmerge**.
3. The original capsules re-appear and are re-promoted.

---

## Part 8: Tips and Workflows

### Animator Workflow: Building a Walk Cycle

1. **Create contact poses**: Contact Left, Contact Down, Pass, Contact Right, Down, Pass (3–4 keys).
2. **Create transitions** between each (e.g., Contact Left → Down, Down → Pass, etc.).
3. **Test transitions** individually: generate 5–10 frames for each.
4. **Adjust timing and arcs** in transitions until happy.
5. **Merge transitions** into a cycle capsule: WalkCycle.
6. **Generate a long sequence** (40 frames) using WalkCycle to confirm the loop repeats cleanly.
7. **Re-target to other characters** (Sidekick, Dragon, Robot) by changing the Character selector.
8. **Celebrate**: You've created a reusable animation asset.

---

### Animator Workflow: Dialogue Scene

1. **Create mouth poses**: Closed, Open, Mid (for a basic dialogue setup).
2. **Create head poses**: Neutral, Tilted Left, Tilted Right, Nod Up, Nod Down.
3. **Create eye poses**: Eyes Forward, Eyes Left, Eyes Right, Blink (frames 1 and 2 of blink).
4. **Create hand gestures**: Reach Forward, Reach Up, Grab, Point, Wave.
5. **Create transitions** between facial poses (fast, snappy).
6. **Layer them**: Generate mouth motion (2 frames for quick phoneme changes), then layer hand gestures (longer, smoother transitions).
7. **Time to dialogue**: Use the timeline ruler to align mouth and hand motion to an audio track (via visual markers).

---

### Team Workflow: Sharing Capsule Assets

1. **One animator** builds a comprehensive character library: poses, transitions, cycles, timing profiles.
2. **Export capsules**: `HeroLibrary.capsules.json`
3. **Share file** with the team (Git, Dropbox, etc.).
4. **Other animators** import: `HeroLibrary.capsules.json` → Their orbital maps are seeded with Hero poses and transitions.
5. **All animators** can now quickly animate Hero scenes without re-inventing poses.
6. **Capsule updates**: If one animator refines a transition, they update `HeroLibrary.capsules.json`, and others pull the latest version.

---

## Part 9: Glossary

| Term | Meaning |
| --- | --- |
| **Capsule** | An explicit, persistent object encoding animator intent (pose, transition, timing, cycle, or character). UUID-identified and deterministic. |
| **Pose Capsule** | A snapshot of a character's skeleton at a key moment (e.g., "Standing," "Contact," "Jump Apex"). |
| **Transition Capsule** | Breakdown logic defining how to move from one pose to another (weight shifts, arcs, anticipation). |
| **Timing Capsule** | Spacing profile controlling how frames are distributed (ease curves, accents, snappy vs. smooth). |
| **Cycle Capsule** | Repeating motion pattern (walk, run, idle breathing, blinking). |
| **Character Capsule** | Anatomical and style parameters (height, rig, proportions, personality). |
| **Orbit Score** | Usage counter. Incremented when a capsule is used; decremented slowly over time (decay). Determines orbital radius on the map. |
| **Agreement Score** | Confidence in a pairing. Tracks how often two capsules have been used together and whether the animator was satisfied (0–1 scale). |
| **Deterministic Seeding** | UUID5-based pseudo-random number generation ensures the same animation frames are generated every time (no stochasticity, no drift). |
| **Retargeting** | Applying a pose or cycle capsule to a different character by scaling and rotating joints based on character proportions. |
| **Shadow Identity** | A merged capsule that represents multiple near-identical capsules (e.g., "WalkCycle v1" and "WalkCycle v2" merged into proxy "WalkCycle"). Reversible. |
| **Orbital Map** | The Saturn-ring visualization showing capsule library organization. Inner rings = frequently used; outer rings = experimental or unused. |
| **Timeline** | Horizontal track view showing animation clips, frame ruler, and playhead. Each clip is labeled with source capsules. |
| **ROCA Earth** | The central pulsing sphere representing your animator identity nucleus. |
| **Functional Lanes** | Concentric ring zones organized by capsule kind (Core, Character, Style, Skill, Topic, Memory, Experimental). |

---

## Part 10: Next Steps

You now understand:
- ✅ The orbital capsule map and how it visualizes your workflow.
- ✅ The timeline and how animation clips are generated.
- ✅ How capsules are explicit, persistent, and deterministic.
- ✅ How to create and combine poses, transitions, and cycles.
- ✅ How to re-target animations to different characters.
- ✅ How to save, share, and collaborate.

**Recommended next exploration**:

1. **Create 3–5 poses** for a simple character.
2. **Create 2–3 transitions** between them.
3. **Generate a 30-frame animation** combining all transitions.
4. **Watch the orbital map evolve** as your capsules gain use and drift inward.
5. **Create a second character** and re-target your animation.
6. **Save the project** and reload it to confirm everything persists.

Welcome to ROCA Animator. **Your animation, your way, forever.**

---

**Document Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Complete User Manual
