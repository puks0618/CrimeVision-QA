#!/usr/bin/env python3
"""
CrimeVision-QA — Store All 13 UCF-Crime Category Embeddings in MongoDB

Creates embeddings for ALL 13 anomaly categories from the UCF-Crime dataset
plus Normal, using Voyage AI. Each category gets an embedding computed from
its textual description, so even categories without ingested videos have
a searchable representation.

For categories that DO have ingested video frames, it also computes a
frame-based centroid embedding (average of all frame embeddings).

Usage:
    python scripts/store_category_embeddings.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import frames_col, transcripts_col, video_library_col, get_db

# ---------------------------------------------------------------------------
# All 13 UCF-Crime anomaly categories + Normal (14 total)
# Each has a rich description that will be embedded via Voyage AI
# ---------------------------------------------------------------------------

CRIME_CATEGORIES: dict[str, str] = {
    "Abuse": (
        "Physical or verbal mistreatment. Surveillance footage showing one or more individuals "
        "physically harming, hitting, slapping, pushing, or verbally threatening another person. "
        "Victims may appear distressed, cowering, or attempting to flee. "
        "Includes domestic violence, child abuse, elder abuse, and workplace harassment."
    ),
    "Arrest": (
        "Law enforcement taking a suspect into custody. Police officers restraining, handcuffing, "
        "or detaining an individual. Suspect may be on the ground, against a wall, or in a vehicle. "
        "Officers may be in uniform or plainclothes. Patrol cars, flashing lights, and police equipment visible. "
        "Includes traffic stops, warrant arrests, and field apprehensions."
    ),
    "Arson": (
        "Intentionally setting fire to property. Surveillance footage showing someone igniting, "
        "lighting, or starting a fire on a building, vehicle, or structure. Flames, smoke, and fire spread visible. "
        "Suspect may be carrying accelerants like gasoline cans, lighters, or matches. "
        "Includes vehicle arson, building arson, and wildland arson."
    ),
    "Assault": (
        "Physical attack on a person. Violent confrontation where one individual strikes, punches, kicks, "
        "or uses a weapon against another person. Victim may fall, bleed, or show signs of injury. "
        "Bystanders may be present or fleeing. "
        "Includes aggravated assault with weapons, simple assault, and battery."
    ),
    "Burglary": (
        "Illegal entry into a building for theft. Breaking and entering through doors, windows, or other access points. "
        "Suspect may use tools like crowbars, lock picks, or break glass. Often occurs at night or in empty buildings. "
        "Surveillance shows forced entry, movement through rooms, and carrying stolen property. "
        "Includes residential burglary, commercial burglary, and home invasion."
    ),
    "Explosion": (
        "Sudden and violent release of energy. Blast wave, fireball, debris field, and structural damage visible. "
        "May involve improvised explosive devices (IED), gas explosions, or industrial accidents. "
        "Surveillance shows sudden bright flash, expanding shockwave, flying debris, smoke cloud, and people fleeing. "
        "Includes bombings, gas leaks, and industrial explosions."
    ),
    "Fighting": (
        "Physical conflict between two or more individuals. Mutual combat where multiple people are "
        "punching, kicking, wrestling, or grappling with each other. Crowd may gather around. "
        "May occur in bars, streets, parking lots, or public venues. "
        "Includes bar fights, street brawls, gang fights, and school fights."
    ),
    "Normal": (
        "Normal everyday activity with no criminal or suspicious behavior. People walking, talking, "
        "shopping, driving, working, or performing routine daily activities. Regular traffic flow, "
        "pedestrians on sidewalks, customers in stores, employees at work. "
        "Calm, orderly scene with no signs of distress, violence, or criminal activity."
    ),
    "Robbery": (
        "Taking property by force or threat of force. Armed or unarmed suspect demanding money, valuables, "
        "or property from victims under threat. May involve weapons like guns, knives, or physical intimidation. "
        "Common locations: banks, convenience stores, gas stations, ATMs, and streets. "
        "Includes armed robbery, mugging, carjacking, and bank robbery."
    ),
    "Shooting": (
        "Use of firearms. Gunshots fired, muzzle flash visible, people ducking or fleeing. "
        "Suspect holding or firing a handgun, rifle, or other firearm. Victims may fall or show wounds. "
        "Shell casings on ground, bullet holes in walls or vehicles. "
        "Includes drive-by shooting, mass shooting, gang-related shooting, and officer-involved shooting."
    ),
    "Shoplifting": (
        "Stealing goods from a retail store. Person concealing merchandise in bags, pockets, clothing, "
        "or containers and leaving without paying. May involve tag removal, bag switching, or distraction. "
        "Surveillance shows suspect browsing, selecting items, concealing, and exiting store. "
        "Includes organized retail theft, grab-and-run, and price tag switching."
    ),
    "Stealing": (
        "General theft or larceny. Taking someone else's property without permission or by deception. "
        "Includes pickpocketing, purse snatching, vehicle theft, bicycle theft, and package theft. "
        "Suspect may grab items and flee, break into vehicles, or take unattended property. "
        "Includes petty theft, grand theft, identity theft, and embezzlement."
    ),
    "Vandalism": (
        "Deliberate destruction or damage to property. Graffiti spraying, window breaking, "
        "car keying, tire slashing, or defacing public or private property. "
        "Suspect may use spray paint, rocks, bats, or other tools to cause damage. "
        "Includes graffiti, property destruction, vehicle vandalism, and public defacement."
    ),
}

# Map existing videos to their correct categories
VIDEO_CATEGORIES: dict[str, str] = {
    "police_body_cam": "Arrest",
    "Train_10_Shooting": "Shooting",
    "YTDown.com_YouTube_Raw-Video-Shootout-With-Store-Robbers_Media_-0jxfcgj1fc_001_480p": "Robbery",
    "WhatsApp_Demo": "Robbery",
    "YTDown_YouTube_Surveillance-Thieves-rip-open-ATM-in-She_Media_wWEHimK-bNw_001_1080p": "Stealing",
}


def update_video_categories() -> None:
    """Update category field on video_library, frames, and transcripts."""
    print("\n[Step 1] Updating video categories in MongoDB...")

    for video_id, category in VIDEO_CATEGORIES.items():
        video_library_col.update_one(
            {"video_id": video_id}, {"$set": {"category": category}}
        )
        fr = frames_col.update_many(
            {"video_id": video_id}, {"$set": {"category": category}}
        )
        tr = transcripts_col.update_many(
            {"video_id": video_id}, {"$set": {"category": category}}
        )
        print(f"  {category:15s} | {video_id[:55]:55s} | frames={fr.modified_count}, transcripts={tr.modified_count}")


def embed_all_categories() -> None:
    """
    Embed all 13+1 category descriptions via Voyage AI and store in MongoDB.
    Also computes frame-based centroid for categories with ingested videos.
    """
    from llm.get_voyage_embed import embedding_service

    db = get_db()
    cat_col = db["category_embeddings"]
    now = datetime.now(timezone.utc)

    # --- Embed category descriptions via Voyage AI ---
    print("\n[Step 2] Embedding all 13+1 category descriptions via Voyage AI...")

    category_names = list(CRIME_CATEGORIES.keys())
    category_descs = list(CRIME_CATEGORIES.values())

    # Batch embed all descriptions (handles rate limiting internally)
    print(f"  Sending {len(category_descs)} descriptions to Voyage AI...")
    desc_embeddings = embedding_service.embed(category_descs)
    print(f"  ✅ Got {len(desc_embeddings)} embeddings ({len(desc_embeddings[0])}-dim)")

    # --- For each category, also compute frame centroid if available ---
    print("\n[Step 3] Computing frame-based centroids for ingested categories...")

    for i, (cat_name, cat_desc) in enumerate(CRIME_CATEGORIES.items()):
        doc: dict = {
            "category": cat_name,
            "description": cat_desc,
            "embedding": desc_embeddings[i],
            "embedding_dim": len(desc_embeddings[i]),
            "embedding_source": "voyage_description",
            "updated_at": now,
        }

        # Check for ingested frames in this category
        frame_cursor = frames_col.find(
            {"category": cat_name, "embedding": {"$exists": True}},
            {"_id": 0, "embedding": 1, "video_id": 1},
        )
        frame_docs = list(frame_cursor)

        if frame_docs:
            embeddings = [f["embedding"] for f in frame_docs if f.get("embedding")]
            if embeddings:
                centroid = np.mean(np.array(embeddings), axis=0).tolist()
                video_ids = list(set(f["video_id"] for f in frame_docs))
                doc["frame_centroid_embedding"] = centroid
                doc["frame_count"] = len(embeddings)
                doc["video_ids"] = video_ids
                doc["video_count"] = len(video_ids)
                print(f"  [{cat_name}] {len(embeddings)} frames from {len(video_ids)} video(s) → centroid computed")
        else:
            doc["frame_count"] = 0
            doc["video_ids"] = []
            doc["video_count"] = 0

        # Check for transcript embeddings
        trans_cursor = transcripts_col.find(
            {"category": cat_name, "embedding": {"$exists": True}},
            {"_id": 0, "embedding": 1},
        )
        trans_docs = list(trans_cursor)
        trans_embeddings = [t["embedding"] for t in trans_docs if t.get("embedding")]

        if trans_embeddings:
            trans_centroid = np.mean(np.array(trans_embeddings), axis=0).tolist()
            doc["transcript_embedding"] = trans_centroid
            doc["transcript_count"] = len(trans_embeddings)
            print(f"  [{cat_name}] {len(trans_embeddings)} transcript segment(s) → centroid computed")

        # Upsert
        cat_col.update_one({"category": cat_name}, {"$set": doc}, upsert=True)

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"{'Category':<15} {'Dim':>5} {'Frames':>7} {'Videos':>7} {'Transcripts':>12} {'Source'}")
    print("-" * 70)

    for doc in cat_col.find({}, {
        "_id": 0, "category": 1, "embedding_dim": 1,
        "frame_count": 1, "video_count": 1, "transcript_count": 1,
        "embedding_source": 1,
    }).sort("category", 1):
        print(
            f"  {doc.get('category','?'):<15}"
            f"{doc.get('embedding_dim', 0):>5}"
            f"{doc.get('frame_count', 0):>7}"
            f"{doc.get('video_count', 0):>7}"
            f"{doc.get('transcript_count', 0):>12}"
            f"  {doc.get('embedding_source', '?')}"
        )

    total = cat_col.count_documents({})
    print(f"\n✅ Done — {total} category embeddings stored in 'category_embeddings' collection")


def main() -> None:
    print("=" * 70)
    print("CrimeVision-QA — All 13+1 Category Embedding Generator")
    print("=" * 70)

    update_video_categories()
    embed_all_categories()


if __name__ == "__main__":
    main()
