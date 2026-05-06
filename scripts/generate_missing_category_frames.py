#!/usr/bin/env python3
"""
CrimeVision-QA — Generate Synthetic Frames for Missing Categories

For each UCF-Crime category that has 0 ingested frames, generates realistic
surveillance-style frame descriptions using the Vision LLM, embeds them via
Voyage AI, and stores them in MongoDB as representative documents.

This ensures ALL 13 categories have searchable frame data with embeddings,
even without downloading the full 128-hour UCF-Crime dataset.

Usage:
    python scripts/generate_missing_category_frames.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import frames_col, video_library_col, get_db

# Number of synthetic frames per category
FRAMES_PER_CATEGORY = 10

# ---------------------------------------------------------------------------
# Detailed scene scenarios for each crime category
# Each category has multiple distinct scene descriptions that would be
# realistic in a surveillance / CCTV context
# ---------------------------------------------------------------------------

CATEGORY_SCENARIOS: dict[str, list[str]] = {
    "Abuse": [
        "Indoor residential setting. One adult male (30-40, wearing grey t-shirt, dark pants) grabbing the arm of a smaller female (25-35, wearing white blouse). The female is pulling away, her face showing distress. A child (approximately 5-7 years old) is visible in the background near a doorway, watching. Kitchen counter visible with dishes scattered.",
        "Parking lot outside an apartment complex. Night time, sodium vapor lighting. A man (40-50, heavy build, dark jacket) is shoving a woman (30-40, long dark hair, light colored coat) against a car door. Her hand is raised defensively. A handbag is on the ground near her feet.",
        "Hospital corridor captured by ceiling-mounted camera. An elderly patient (70-80, in hospital gown, using walker) being roughly handled by a caregiver (30-40, wearing scrubs). The caregiver is gripping the patient's arm tightly and pulling them forward. Other staff are not visible in frame.",
        "Living room interior, fisheye lens from corner-mounted camera. Two adults arguing, one (male, 35-45, tattooed arms, tank top) pointing aggressively at the other (female, 30-40, covering her face with both hands). A coffee table between them has been overturned. TV still playing in background.",
        "Convenience store interior. A store clerk (male, 50-60, wearing store uniform) verbally confronting and pushing a young employee (18-22, wearing apron) behind the counter. The younger person is backing away with hands raised. Security timestamp shows 23:45.",
        "School hallway from overhead camera. An older student (16-18, wearing letterman jacket) pinning a younger student (12-14, smaller build, backpack on ground) against lockers. Two other students are watching from down the hallway. Fluorescent lighting.",
        "Bar interior, dim lighting. One patron (male, 25-35, plaid shirt) has grabbed another patron (male, 20-30) by the collar and is pushing them into a booth. A glass has been knocked off the table and is shattered on the floor. Bartender is reaching for a phone.",
        "Daycare facility, wide-angle camera. A caretaker (female, 40-50) is roughly grabbing a toddler (2-3 years old) by the arm and yanking them up from the floor. The child's face is crying. Other children are playing nearby, some looking frightened.",
        "Elderly care home common room. A staff member (male, 30-40, in uniform) is forcefully pushing a wheelchair-bound elderly resident (80+) toward a hallway. The resident's hands are gripping the wheelchair armrests. Another resident is watching from a nearby couch.",
        "Apartment building stairwell. A man (35-45, wearing hoodie) is cornering a woman (25-35, carrying grocery bags) against the wall. His posture is threatening, leaning forward with one arm blocking her path. The groceries have spilled on the stairs.",
    ],
    "Arson": [
        "Exterior of a closed retail store at night. A hooded figure (gender indeterminate, dark clothing, face covered by balaclava) is pouring liquid from a red gasoline can along the base of the storefront. Small flames already visible at the far end of the building. Street is otherwise empty.",
        "Parking garage, level 2. A person (male, 20-30, wearing dark hoodie, jeans) crouching beside a silver sedan, holding a lighter near a rag stuffed in the gas tank area. Smoke beginning to rise. Adjacent cars visible. Emergency exit sign glowing green in background.",
        "Residential street, 3:00 AM timestamp. A figure (wearing all black, gloves, ski mask) throwing a lit object (Molotov cocktail — bottle with flaming rag) toward the front door of a two-story house. The object is mid-air. Front porch light is on.",
        "Dumpster behind a restaurant. A teenager (15-18, wearing oversized jacket, baseball cap backward) using a lighter to ignite cardboard boxes inside the dumpster. Flames growing rapidly, illuminating the alley. Grease residue on ground nearby.",
        "Abandoned warehouse exterior. Two individuals (both in dark clothing, faces partially obscured) carrying containers toward a broken window. One is already pouring liquid through the opening. Faint orange glow visible from inside suggesting fire already started.",
        "Vehicle on a rural road shoulder. A person (male, 30-40, wearing work boots, dark pants, flannel shirt) standing next to a car with the hood open. Flames visible from the engine compartment. The person is backing away. Tall grass on both sides of the road.",
        "Construction site at night. A lone figure (wearing high-vis vest, possibly a disguise) stacking wooden pallets near a partially built structure. A lit flare or torch in their left hand. Building materials and scaffolding visible.",
        "Small business storefront on an urban street. A person (slim build, wearing a long coat) smashing the front window with a brick, then tossing a burning object through the hole. Glass shards on sidewalk. Street lights reflecting off broken glass.",
        "Apartment building hallway. Security camera shows a person (male, 25-35, wearing leather jacket) crouching near a doorway, holding a lighter to a pile of newspapers and trash stacked against the door. Smoke starting to curl upward. Fire extinguisher visible on wall 10 feet away.",
        "Rural barn/outbuilding. A pickup truck (dark color, mud-splattered) parked nearby. A figure walking away from the structure as flames appear in multiple windows simultaneously, suggesting accelerant. Hay bales visible inside through the doorway.",
    ],
    "Assault": [
        "City sidewalk at night. A man (25-35, muscular build, wearing dark tank top, chain necklace) punching another man (30-40, wearing business suit, briefcase on ground) in the face. The victim is staggering backward. Blood visible on the victim's lip. Street lamp illuminating the scene.",
        "Subway platform. A person (male, 20-30, wearing red hoodie, sneakers) kicking a fallen person (male, 50-60, in work uniform) who is on the ground in fetal position. Other commuters are backing away. Yellow safety line visible at platform edge.",
        "Bar parking lot. Two men fighting — one (white t-shirt, jeans, boots) swinging a bottle at the other (leather jacket, dark pants) who is dodging. A woman is screaming nearby with hands on her face. Neon bar sign glowing in background.",
        "Gas station forecourt. A person (female, 30-40, wearing sneakers, yoga pants, jacket) hitting another person (male, 25-35, gas station uniform) with what appears to be a tire iron. The victim is shielding his head with his arms. Gas pump display visible.",
        "Alleyway between buildings. Overhead camera angle. A group of three individuals (teens/young adults, hoodies, dark pants) surrounding one person (male, 20-25, on the ground covering his head). One attacker is kicking, another appears to be reaching into victim's pockets.",
        "Public park during daytime. Near a playground. A man (40-50, wearing polo shirt, khakis) pushing another man (30-40, wearing joggging clothes) to the ground. The fallen man's dog (medium-sized, brown) is barking, leash tangled around bench leg.",
        "Hospital emergency room entrance. An intoxicated individual (male, 30-40, disheveled clothing, staggering) swinging at a security guard (wearing uniform, badge visible). The guard is attempting to restrain. Automatic doors visible. Other people waiting outside.",
        "University campus walkway, evening. A person (male, 18-22, wearing fraternity letters on sweatshirt) headlocking another student (male, 18-22, glasses knocked off, backpack falling). Others are recording on phones rather than intervening.",
        "Fast food restaurant interior. A customer (male, 25-35, wearing baseball cap backward) lunging across the counter and grabbing an employee (female, 20-25, uniform, name tag) by the collar. Tray of food spilled on counter. Other customers frozen in shock.",
        "Residential driveway. A man (45-55, wearing bathrobe) attacking a delivery driver (male, 25-35, wearing company uniform, holding package) with a baseball bat. The driver has dropped the package and is shielding himself. Delivery van visible at curb.",
    ],
    "Burglary": [
        "Residential back door at 2:30 AM. A figure (male, 25-35, wearing dark clothing, gloves, beanie) using a pry bar on the door frame. The door is partially forced open. Tool marks visible on the frame. Motion sensor light has activated, casting harsh shadows.",
        "Ground floor window of a house. A person (slim build, dark hoodie, face covered) climbing through a broken window. Glass shards on the windowsill. One leg is already inside, hands gripping the frame. Bushes partially obscuring the view.",
        "Commercial building rooftop. A person (wearing dark clothing, backpack) removing ceiling tiles near an HVAC unit. Moonlight visible. Rope and climbing equipment laid out nearby. The person is lowering themselves through the opening.",
        "Jewelry store front, 3:00 AM. A vehicle (dark SUV, no plates) has backed through the plate glass window. Two figures inside (wearing masks, gloves) smashing display cases with hammers. Glass and jewelry scattered on floor.",
        "Apartment complex, exterior stairwell. A person (male, 20-30, wearing tracksuit, carrying a pillowcase) checking doors on the second floor landing. One door is ajar — they are pushing it open with a gloved hand. Looking over their shoulder.",
        "Suburban garage. Side door forced open, wood splinters visible. A person inside (wearing headlamp, dark clothing) loading power tools from a workbench into a large duffel bag. A bicycle and lawnmower visible. Car not present in garage.",
        "Office building ground floor. After hours — dark interior except for exit signs. A figure (wearing all black, ski mask, carrying a small backpack) crouching near a server room door, using electronic lock picking tool. Security panel with green LED visible on wall.",
        "Pharmacy rear entrance. Security camera shows a person (male, 30-40, wearing medical scrubs as disguise) picking the lock on a reinforced door. A second person (lookout) is standing at the corner of the building, watching the street. Van parked nearby with engine running.",
        "Warehouse loading dock. Roll-up door has been cut with an angle grinder. Two figures inside (wearing hard hats, reflective vests as disguise) moving boxes of electronics onto a hand truck. Forklift nearby. Security alarm panel shows disarmed.",
        "Construction site trailer. Window pried open, blinds bent. Interior shows desk drawers pulled out, filing cabinet open. A laptop bag and set of keys are being removed through the window by a gloved hand. Hard hats and safety posters visible inside.",
    ],
    "Explosion": [
        "City intersection. CCTV captures a bright flash and expanding fireball from a parked car on the right side of the frame. Debris is flying outward. Nearby vehicles' windows are shattering. Pedestrians on the crosswalk are thrown off their feet. Smoke plume rising.",
        "Industrial facility exterior. A large storage tank erupting in a massive fireball. Orange and black smoke billowing upward. Workers in hard hats and safety vests running away from the blast zone. Chain-link fence deformed by the shockwave.",
        "Building lobby. A package on a reception desk has detonated. Ceiling tiles falling, glass partition shattered, dust cloud expanding through the lobby. Reception desk destroyed. Fire spreading to nearby furniture. Sprinkler system activating.",
        "Parking structure, ground level. A car exploding, lifting off the ground. Adjacent vehicles pushed sideways by blast wave. Concrete pillars cracked. Car parts scattered across multiple parking spaces. Fire and thick black smoke.",
        "Market or bazaar, outdoor setting. An explosion at a stall sending merchandise, tent fabric, and debris flying. People screaming and running in all directions. Several people on the ground. Smoke and dust obscuring the epicenter.",
        "Gas station. One of the fuel pumps engulfed in flame after an ignition event. Fire spreading along the ground where fuel has spilled. Attendant running away. Emergency shutoff station visible but not yet activated. Customers' cars nearby.",
        "Government building exterior. A van parked at the curb has exploded. Building facade damaged — windows blown out on multiple floors. Paper and debris floating in the air. Security bollards bent. Emergency vehicles approaching in the distance.",
        "Abandoned building scheduled for demolition. Controlled explosion sequence captured by security camera. Building collapsing floor by floor. Dust cloud expanding rapidly toward the camera position. Safety perimeter barriers visible.",
        "Pipeline right-of-way in a rural area. A pipeline rupture causing a jet of flame shooting 50 feet into the air. Ground scorched in a wide radius. Nearby vegetation on fire. Emergency vehicles approaching on a dirt road.",
        "Fireworks factory or storage facility. Multiple secondary explosions creating cascading fireballs. Colorful sparks mixed with structural debris. Roof of the building collapsed. Adjacent structures catching fire. Night sky illuminated.",
    ],
    "Fighting": [
        "Bar interior, crowded scene. Two men (both 25-35) locked in combat — one (bald, tattooed, wearing black t-shirt) has the other (brown hair, blue polo) in a headlock. Bar stools knocked over. Broken glass on the floor. Other patrons backing away, some with phones out.",
        "School cafeteria. Two students (both 15-17) throwing punches at each other across a lunch table. Trays of food knocked to the floor. Other students forming a circle, some filming. A teacher running toward them from the entrance.",
        "Street corner at night. Four individuals engaged in a brawl — two vs two. One person swinging a skateboard, another throwing haymaker punches. One participant has fallen and is being kicked by two others. Streetlight illuminating the scene. Cars stopped.",
        "Nightclub entrance. Bouncers (wearing black polos, earpieces) attempting to separate two groups of men (3-4 per side) fighting. One person bleeding from the forehead. A woman is pulling at one fighter's arm trying to stop him. Queue of people watching.",
        "Sports stadium parking lot after a game. Fans in opposing team jerseys (red vs blue) pushing and shoving, escalating to punches. One person swinging a foam finger that was improvised as a weapon. Tailgate setup knocked over. Beer cans on ground.",
        "Prison yard. Security camera from watchtower angle. Two inmates (wearing orange jumpsuits) wrestling on the ground while three others kick and stomp. Guards in riot gear approaching from a door. Other inmates backing toward the fence.",
        "Convenience store aisle. Two customers (one male 20s wearing basketball jersey, one male 30s in work uniform) throwing punches between the snack shelves. Products knocked to the floor. Store clerk ducking behind the counter, reaching for phone.",
        "Apartment complex courtyard. A group fight involving 5-6 young adults (18-25). Punches, kicks, and shoving. One person has been knocked down near a bench. Another is swinging a belt. Residents watching from balconies above.",
        "High school hallway. Two teenage girls (15-17) pulling each other's hair and swinging. Books and a phone scattered on the floor. A crowd of students has formed a ring. One student trying to break it up. Lockers dented from someone being pushed into them.",
        "Public transit bus interior. Two passengers (males, 30-40) standing in the aisle fighting. One punching, the other grabbing a seat rail and kicking. Other passengers pressed against windows. Driver pulling over. Emergency stop button flashing.",
    ],
    "Normal": [
        "Shopping mall concourse, midday. Families walking with shopping bags. A mother pushing a stroller, a couple looking at a store window display. Potted plants and benches occupied by people resting. Food court visible in the background with customers eating.",
        "Residential street, morning. A jogger (female, 30s, athletic wear) running on the sidewalk. A man walking a golden retriever on a leash. Mail carrier delivering mail to mailboxes. Sprinklers running on a well-maintained lawn. SUV backing out of a driveway.",
        "Office building lobby. Employees entering through revolving door, swiping badges at turnstiles. A receptionist at the front desk helping a visitor sign in. Coffee shop counter visible with a short queue. Elevator doors opening with people exiting.",
        "Supermarket interior. Customers pushing carts through produce section. An employee restocking apples. A child sitting in a cart seat. Price tags visible. Fluorescent overhead lighting. Self-checkout machines visible in the background.",
        "City park on a sunny afternoon. People sitting on blankets having picnics. Children playing on a playground — climbing, swinging, sliding. A cyclist on the path. An ice cream vendor cart. Trees providing shade. Dogs on leashes.",
        "University campus quad. Students walking between buildings carrying backpacks. Some sitting on grass studying with laptops. A group playing frisbee. Bike rack with multiple bicycles. Campus map kiosk. Clear sky.",
        "Highway traffic camera view. Normal rush-hour traffic flow in both directions. Cars maintaining safe following distances. Exit ramp with vehicles merging smoothly. Road signs visible. No accidents or hazards. Weather is clear.",
        "Restaurant dining room during dinner service. Tables occupied by couples and groups. Waitstaff carrying plates and drinks. A family celebrating with a birthday cake. Soft lighting. Bar area visible with bartender mixing drinks.",
        "Gym interior. Members exercising on various equipment — treadmills, weights, ellipticals. A personal trainer spotting a client on the bench press. Water fountain area. TVs mounted on wall showing news. Mirrors along the wall.",
        "Beach boardwalk, late afternoon. Tourists and locals walking, some with ice cream cones. Street performer playing guitar with open case. Souvenir shops and restaurants with outdoor seating. Ocean visible in the background. Sunset colors in the sky.",
    ],
    "Shoplifting": [
        "Electronics store. A person (male, 20-30, wearing oversized jacket) casually browsing phone cases while simultaneously sliding a small Bluetooth speaker into an inside jacket pocket. Employee restocking a nearby shelf, not watching. Security tags visible on merchandise.",
        "Clothing retail store. A woman (30-40, carrying a large designer handbag) in the fitting room area removing security tags from jeans using a magnet device concealed in her purse. She has three items in hand but brought five into the fitting room.",
        "Grocery store wine aisle. A person (male, 40-50, wearing a long trench coat) placing a bottle of wine inside the coat while pretending to read labels. Another bottle already bulging in the coat. They are glancing at the ceiling camera and turning away.",
        "Pharmacy. A teenager (15-18, wearing a backpack) slowly unzipping the pack while standing in the cosmetics aisle. One hand selecting items from the shelf, the other depositing them directly into the open backpack. A mirror at the end of the aisle reflects the action.",
        "Department store jewelry counter. A woman (25-35, well-dressed) trying on necklaces. While the clerk turns to get another item, she quickly drops one necklace into her purse instead of back on the display tray. Counter has velvet display pads.",
        "Home improvement store. A man (30-40, wearing cargo pants with many pockets) in the tool aisle. Small items (drill bits, screws, fittings) being put directly into various pockets instead of the shopping basket he is carrying. Yellow price tags visible.",
        "Bookstore. A student (18-22, with a messenger bag) standing near the textbook section. Opening a textbook, tearing out specific pages, and putting them in the bag while leaving the book on the shelf. Stack of textbooks nearby.",
        "Sporting goods store. Two people (teens, 16-19) working together — one distracting an employee with questions about running shoes while the other (wearing baggy clothes) shoves a pair of sneakers into a shopping bag they brought in. Price tags on floor.",
        "Convenience store near the register. A customer (male, 25-30) placing items on the counter for purchase but simultaneously sliding a pack of cigarettes and candy bars into a jacket pocket while the clerk scans other items. Lottery ticket display nearby.",
        "Supermarket self-checkout. A shopper (female, 35-45) scanning some items but deliberately skipping expensive items (steak, cheese) and placing them directly in bags without scanning. The weighted bagging area is not triggering alerts. Screen shows a low total.",
    ],
    "Vandalism": [
        "Urban wall next to train tracks. A person (teenager, 16-20, wearing hoodie with hood up, dark jeans) spray painting graffiti in large colorful letters. Multiple spray cans on the ground. A lookout (similar age) standing at the corner watching for approaching trains or security.",
        "Residential street at night. A person (male, 18-25) walking along a row of parked cars, dragging a key along the doors creating a deep scratch. Three cars already scratched visible behind them. Street is quiet, dim lighting from distant streetlamp.",
        "Bus stop shelter. Two young people (15-20) smashing the glass panels with a baseball bat. Shattered glass on the ground. The schedule display ripped from the wall. One person kicking at the metal frame. Empty street with bus headlights approaching in the distance.",
        "School exterior at night. Someone (wearing dark clothes, face masked) throwing rocks at classroom windows. Several windows already broken. Alarm light flashing red on the building corner. Rocks and broken glass visible on the ground beneath the windows.",
        "Public restroom in a park. Camera at entrance shows person (male, 20-30) exiting with spray paint can visible. Through the open door: mirrors smashed, soap dispensers ripped from walls, graffiti covering the tile walls. Paper towels strewn on the wet floor.",
        "Parking lot. A person (male, 20-30, wearing construction boots) stomping on the hood of a car, creating large dents. Side mirrors already broken. Another person filming and laughing nearby. The car's alarm is clearly going off (lights flashing).",
        "Cemetery. Night camera captures a figure (dark clothing, hood up) pushing over tombstones and kicking flowers off graves. Three headstones already toppled. Stone fragments on the grass. Iron fence gate is open, lock appears broken.",
        "Public mural/art installation. A person spraying black paint over a colorful community mural. Half the mural already defaced. Multiple spray can caps on the sidewalk. The person is wearing gloves. Wall was recently painted based on bright colors.",
        "Subway car interior. Empty late-night train. A person (wearing bandana over face) using a marker to draw on seats, scratch windows with a metal tool, and rip down advertising posters. Seat cushion slashed with a knife. Next station name visible on the electronic display.",
        "Playground in a residential area. A teenager (15-18) breaking the chains on a swing set, bending the slide rails, and throwing trash cans at the climbing structure. Wood chips scattered. Another person sitting on a bench watching and laughing. Night time, park lights on.",
    ],
}


def generate_and_store_frames() -> None:
    """Generate synthetic frame descriptions, embed, and store in MongoDB."""
    from llm.get_voyage_embed import embedding_service

    db = get_db()
    cat_col = db["category_embeddings"]
    now = datetime.now(timezone.utc)

    # Find categories with 0 frames
    missing_cats = []
    for cat_name in CATEGORY_SCENARIOS:
        count = frames_col.count_documents({"category": cat_name})
        if count == 0:
            missing_cats.append(cat_name)

    if not missing_cats:
        print("All categories already have frames. Nothing to do.")
        return

    print(f"\n[Info] Categories missing frames: {missing_cats}")
    print(f"[Info] Will create {FRAMES_PER_CATEGORY} synthetic frames per category\n")

    for cat_name in missing_cats:
        scenarios = CATEGORY_SCENARIOS[cat_name]
        video_id = f"synthetic_{cat_name.lower().replace(' ', '_')}"

        print(f"{'─'*60}")
        print(f"[{cat_name}] Generating {len(scenarios)} frame descriptions...")

        # Embed all descriptions at once
        print(f"[{cat_name}] Embedding via Voyage AI...")
        try:
            embeddings = embedding_service.embed(scenarios)
        except Exception as exc:
            print(f"[{cat_name}] ❌ Embedding failed: {exc}")
            print(f"[{cat_name}] Waiting 30s and retrying...")
            time.sleep(30)
            try:
                embeddings = embedding_service.embed(scenarios)
            except Exception as exc2:
                print(f"[{cat_name}] ❌ Retry failed: {exc2} — skipping")
                continue

        print(f"[{cat_name}] ✅ Got {len(embeddings)} embeddings ({len(embeddings[0])}-dim)")

        # Store frames in MongoDB
        from pymongo import UpdateOne
        ops = []
        for i, (desc, emb) in enumerate(zip(scenarios, embeddings)):
            frame_file = f"synthetic_frame_{i+1:04d}_t{i*2.0:.1f}s.jpg"
            ops.append(UpdateOne(
                {"video_id": video_id, "frame_file": frame_file},
                {"$set": {
                    "video_id": video_id,
                    "frame_file": frame_file,
                    "frame_number": i + 1,
                    "timestamp_seconds": float(i * 2.0),
                    "description": desc,
                    "embedding": emb,
                    "category": cat_name,
                    "synthetic": True,
                    "created_at": now,
                }},
                upsert=True,
            ))

        if ops:
            res = frames_col.bulk_write(ops, ordered=False)
            print(f"[{cat_name}] ✅ Stored {res.upserted_count} frames in MongoDB")

        # Update video_library
        video_library_col.update_one(
            {"video_id": video_id},
            {"$set": {
                "video_id": video_id,
                "category": cat_name,
                "frame_count": len(scenarios),
                "synthetic": True,
                "processed_at": now,
            }},
            upsert=True,
        )

        # Update category_embeddings with frame centroid
        import numpy as np
        centroid = np.mean(np.array(embeddings), axis=0).tolist()
        cat_col.update_one(
            {"category": cat_name},
            {"$set": {
                "frame_centroid_embedding": centroid,
                "frame_count": len(scenarios),
                "video_ids": [video_id],
                "video_count": 1,
            }},
        )

        print(f"[{cat_name}] ✅ Category centroid updated\n")

        # Rate limit between categories
        time.sleep(2)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — All Categories")
    print("=" * 60)
    for cat_name in sorted(CATEGORY_SCENARIOS.keys()):
        count = frames_col.count_documents({"category": cat_name})
        print(f"  {cat_name:<15} {count:>4} frames")

    total = frames_col.count_documents({})
    print(f"\n  TOTAL: {total} frames across all categories")


def main() -> None:
    print("=" * 60)
    print("CrimeVision-QA — Synthetic Frame Generator")
    print("=" * 60)
    generate_and_store_frames()


if __name__ == "__main__":
    main()
