"""Synthetic biography generator (bioS-style): the fact-load dose.

Entities carry 6 attributes drawn from finite pools (~53 bits/entity).
Every rendered biography exists in two renderings: dense (values inline,
loss everywhere) and split (values wrapped in organizer lookup calls and
loss-masked). Names and values are generated combinatorially so no
real-world knowledge helps.
"""

from __future__ import annotations

import datetime
import random

from corpusgen.records import ATTRIBUTES, BioRecord, Doc, QAItem, Segment, lookup_segments

# ---------------------------------------------------------------- name pools

_ONSETS = ["B", "Br", "C", "D", "Dr", "F", "G", "H", "J", "K", "L", "M", "N",
           "P", "R", "S", "T", "V", "W", "Z"]
_NUCLEI = ["a", "e", "i", "o", "u", "ae", "ia"]
_FIRST_CODAS = ["l", "n", "r", "s", "th", "m", "d"]
_FIRST_ENDS = ["a", "o", "en", "is", "ara", "elle", "ian"]

def _build_first_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for onset in _ONSETS:
        for nucleus in _NUCLEI:
            for coda in _FIRST_CODAS:
                for end in _FIRST_ENDS:
                    name = (onset + nucleus + coda + end).capitalize()
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
                    if len(names) >= 460:
                        return names
    return names

_MIDDLE_SYLL_A = ["Ar", "Bel", "Cor", "Dan", "El", "Fen", "Gal", "Hol", "Il",
                  "Jor", "Kel", "Lor", "Mar", "Nor", "Or", "Pel", "Quin",
                  "Ren", "Sol", "Tam", "Ul", "Ver", "Wil", "Yor"]
_MIDDLE_SYLL_B = ["a", "an", "ec", "en", "ia", "in", "io", "is", "on", "us"]

def _build_middle_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for a in _MIDDLE_SYLL_A:
        for b in _MIDDLE_SYLL_B:
            name = a + b
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names  # 24 * 10 = 240

_LAST_ROOTS = ["Ash", "Bark", "Birch", "Black", "Bright", "Brook", "Clay",
               "Cliff", "Cole", "Dale", "Dun", "East", "Elm", "Fair", "Fern",
               "Field", "Ford", "Fox", "Gold", "Gray", "Green", "Hale",
               "Hart", "Hawk", "Hazel", "Heath", "High", "Hill", "Holt",
               "Kirk", "Lake", "Lang", "Leaf", "Long", "Marsh", "Mead",
               "Mill", "Moor", "Moss", "North", "Oak", "Pine", "Red",
               "Ridge", "Rowan", "Rush", "Sand", "Shaw", "Snow", "Stone",
               "Storm", "Swift", "Thorn", "Vale", "West", "Whit", "Wild",
               "Win", "Wolf", "Wood"]
_LAST_SUFFIXES = ["berg", "born", "burn", "by", "cott", "croft", "dale",
                  "den", "field", "gate", "ham", "hurst", "ley", "lock",
                  "man", "mere", "more", "row", "shaw", "smith", "son",
                  "stead", "ton", "wall", "ward", "well", "wick", "wood",
                  "worth", "yard"]

def _build_last_names() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for root in _LAST_ROOTS:
        for suf in _LAST_SUFFIXES:
            name = root + suf
            low = name.lower()
            # avoid doubled fragments like "Woodwood"
            if low.count(root.lower()) > 1:
                continue
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names  # ~ 60*30 minus collisions

FIRST_NAMES = _build_first_names()
MIDDLE_NAMES = _build_middle_names()
LAST_NAMES = _build_last_names()
assert len(FIRST_NAMES) >= 400 and len(MIDDLE_NAMES) >= 200 and len(LAST_NAMES) >= 600

# ---------------------------------------------------------------- value pools

_CITY_ROOTS = ["Alder", "Aspen", "Basalt", "Bay", "Briar", "Cedar", "Cinder",
               "Clear", "Cobble", "Copper", "Crane", "Cypress", "Drift",
               "Ember", "Falcon", "Flint", "Garnet", "Glen", "Granite",
               "Harbor", "Heron", "Iron", "Juniper", "Kestrel", "Larch",
               "Laurel", "Linden", "Maple", "Marble", "Meadow", "Mesa",
               "Osprey", "Otter", "Pebble", "Pumice", "Quartz", "Raven",
               "Reed", "Salt", "Slate", "Spruce", "Summit", "Tamarack",
               "Timber", "Trout", "Walnut", "Willow", "Wren"]
_CITY_SUFFIXES = ["ford", " Falls", " Hollow", " Springs", "view", " Point",
                  "mont", " Grove", "field", " Landing"]

def _build_cities() -> list[str]:
    cities: list[str] = []
    seen: set[str] = set()
    for root in _CITY_ROOTS:
        for suf in _CITY_SUFFIXES:
            city = root + suf
            if city not in seen:
                seen.add(city)
                cities.append(city)
            if len(cities) >= 200:
                return cities
    return cities

_UNI_PATTERNS = ["{} University", "{} State University", "University of {}",
                 "{} Institute of Technology", "{} College", "{} Polytechnic"]

def _build_universities(cities: list[str]) -> list[str]:
    unis: list[str] = []
    seen: set[str] = set()
    for pattern in _UNI_PATTERNS:
        for city in cities:
            uni = pattern.format(city)
            if uni not in seen:
                seen.add(uni)
                unis.append(uni)
            if len(unis) >= 300:
                return unis
    return unis

_MAJOR_BASES = ["Biology", "Chemistry", "Physics", "Geology", "Astronomy",
                "Mathematics", "Statistics", "Economics", "History",
                "Linguistics", "Philosophy", "Psychology", "Sociology",
                "Anthropology", "Archaeology", "Literature", "Music Theory",
                "Architecture", "Engineering", "Agronomy", "Botany",
                "Zoology", "Ecology", "Meteorology", "Oceanography"]
_MAJOR_MODS = ["", "Applied ", "Computational ", "Theoretical ", "Marine "]

def _build_majors() -> list[str]:
    majors: list[str] = []
    seen: set[str] = set()
    for mod in _MAJOR_MODS:
        for base in _MAJOR_BASES:
            major = mod + base
            if major not in seen:
                seen.add(major)
                majors.append(major)
            if len(majors) >= 100:
                return majors
    return majors

_EMP_ROOTS = ["Apex", "Arbor", "Atlas", "Beacon", "Bluff", "Cairn", "Cascade",
              "Compass", "Crest", "Current", "Delta", "Draft", "Ridgeline",
              "Fathom", "Foundry", "Gable", "Harbor", "Helix", "Keystone",
              "Lantern", "Ledger", "Meridian", "Mosaic", "Northwind",
              "Outpost", "Pinnacle", "Quarry", "Relay", "Sextant", "Signal",
              "Summit", "Tandem", "Trellis", "Vantage", "Vertex", "Waypoint",
              "Zenith"]
_EMP_SUFFIXES = ["Analytics", "Dynamics", "Fabrication", "Freight",
                 "Holdings", "Industries", "Instruments", "Logistics",
                 "Manufacturing", "Robotics", "Systems", "Textiles"]

def _build_employers() -> list[str]:
    emps: list[str] = []
    seen: set[str] = set()
    for root in _EMP_ROOTS:
        for suf in _EMP_SUFFIXES:
            emp = f"{root} {suf}"
            if emp not in seen:
                seen.add(emp)
                emps.append(emp)
            if len(emps) >= 263:
                return emps
    return emps

_CITIES = _build_cities()

VALUE_POOLS: dict[str, list[str]] = {
    "birth_city": list(_CITIES),
    "university": _build_universities(_CITIES),
    "major": _build_majors(),
    "employer": _build_employers(),
    "current_city": list(_CITIES),
}
assert len(VALUE_POOLS["birth_city"]) == 200
assert len(VALUE_POOLS["university"]) == 300
assert len(VALUE_POOLS["major"]) == 100
assert len(VALUE_POOLS["employer"]) == 263
assert len(VALUE_POOLS["current_city"]) == 200
for _k, _pool in VALUE_POOLS.items():
    assert len(set(_pool)) == len(_pool)

RELATION_PHRASES = {
    "birth_date": "birth date",
    "birth_city": "birth city",
    "university": "university",
    "major": "major",
    "employer": "employer",
    "current_city": "current city",
}

BIRTH_DATE_MIN = datetime.date(1930, 1, 1)
BIRTH_DATE_MAX = datetime.date(2005, 12, 31)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def format_date(d: datetime.date) -> str:
    return f"{MONTH_NAMES[d.month - 1]} {d.day}, {d.year}"


# ---------------------------------------------------------------- records


def generate_records(n_entities: int, seed: int) -> list[BioRecord]:
    """Deterministic and prefix-stable: generate_records(n+k, s)[:n] ==
    generate_records(n, s) — each record consumes the RNG in sequence."""
    rng = random.Random(seed)
    n_days = (BIRTH_DATE_MAX - BIRTH_DATE_MIN).days + 1
    seen_names: set[str] = set()
    records: list[BioRecord] = []
    for entity_id in range(n_entities):
        while True:
            name = " ".join([
                rng.choice(FIRST_NAMES),
                rng.choice(MIDDLE_NAMES),
                rng.choice(LAST_NAMES),
            ])
            if name not in seen_names:
                seen_names.add(name)
                break
        attrs = {
            "birth_date": format_date(
                BIRTH_DATE_MIN + datetime.timedelta(days=rng.randrange(n_days))
            ),
            "birth_city": rng.choice(VALUE_POOLS["birth_city"]),
            "university": rng.choice(VALUE_POOLS["university"]),
            "major": rng.choice(VALUE_POOLS["major"]),
            "employer": rng.choice(VALUE_POOLS["employer"]),
            "current_city": rng.choice(VALUE_POOLS["current_city"]),
        }
        records.append(BioRecord(entity_id=entity_id, name=name, attrs=attrs))
    return records


# ---------------------------------------------------------------- templates
# Each template is (prefix, suffix): sentence = prefix + VALUE + suffix.
# Prefix always ends with a space; suffix starts with punctuation.

BIO_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "birth_date": [
        ("{name} was born on ", "."),
        ("{name} came into the world on ", "."),
        ("Municipal records give {name}'s date of birth as ", "."),
        ("{name} celebrates a birthday every year on ", "."),
        ("The family register lists {name} as born on ", "."),
        ("Born on ", ", {name} was the youngest of the household."),
        ("A birth certificate dated ", " marks the start of {name}'s story."),
        ("Their life began on ", "."),
        ("The date of birth on file reads ", "."),
        ("Hospital paperwork records the birth as ", "."),
        ("Every official form lists the same birth date: ", "."),
        ("The entry in the town ledger says ", "."),
        ("Friends remember the birthday falling on ", " each year."),
        ("The earliest record carries the date ", "."),
        ("Census takers noted a birth date of ", "."),
        ("On ", ", the family welcomed a new child."),
        ("The passport application gives ", " as the date of birth."),
        ("School enrollment papers record a birth date of ", "."),
        ("A date of ", " appears beside the name in the registry."),
        ("The baptismal record is dated ", "."),
    ],
    "birth_city": [
        ("{name} was born in ", "."),
        ("{name} first opened their eyes in ", "."),
        ("The birthplace on record is ", "."),
        ("Their earliest years began in ", "."),
        ("{name} entered the world in the town of ", "."),
        ("In ", ", {name} was born on a quiet street."),
        ("The town of ", " appears on the birth certificate."),
        ("Family lore places {name}'s birth in ", "."),
        ("All records agree the birthplace was ", "."),
        ("The hospital stood in ", ", the recorded place of birth."),
        ("A hometown of ", " shaped those first years."),
        ("The registry names ", " as the place of birth."),
        ("Their story starts in ", "."),
        ("Born in ", ", the family soon settled nearby."),
        ("Official papers list a birthplace of ", "."),
        ("The earliest address on file sits in ", "."),
        ("Childhood began in ", "."),
        ("Municipal archives place the birth in ", "."),
        ("It was in ", " that the story began."),
        ("The certificate names ", " as the town of birth."),
    ],
    "university": [
        ("{name} studied at ", "."),
        ("{name} completed a degree at ", "."),
        ("The undergraduate years were spent at ", "."),
        ("A diploma hangs on the wall from ", "."),
        ("{name} enrolled at ", " straight out of school."),
        ("At ", ", {name} spent four demanding years."),
        ("The alma mater on record is ", "."),
        ("The transcript of {name} comes from ", "."),
        ("Graduation day took place at ", "."),
        ("Lecture halls at ", " filled those student years."),
        ("The degree was conferred by ", "."),
        ("Student life unfolded at ", "."),
        ("Coursework was completed at ", "."),
        ("The university listed on every form is ", "."),
        ("An acceptance letter from ", " changed everything."),
        ("Four years at ", " ended with a degree."),
        ("The commencement ceremony was held at ", "."),
        ("Alumni records at ", " include the name."),
        ("Higher education began at ", "."),
        ("The framed certificate names ", "."),
    ],
    "major": [
        ("{name} majored in ", "."),
        ("{name} earned a degree in ", "."),
        ("The chosen field of study was ", "."),
        ("Their coursework centered on ", "."),
        ("{name} spent four undergraduate years studying ", "."),
        ("A concentration in ", " defined {name}'s studies."),
        ("The diploma of {name} names a major in ", "."),
        ("The transcript shows a focus on ", "."),
        ("Seminars in ", " filled the weekly timetable."),
        ("The declared major was ", "."),
        ("Most credits were earned in ", "."),
        ("The thesis was written in the field of ", "."),
        ("Academic life revolved around ", "."),
        ("A degree program in ", " led to graduation."),
        ("The field listed on the diploma is ", "."),
        ("Late nights went to problem sets in ", "."),
        ("The department of ", " became a second home."),
        ("Study focused on ", " from the first semester."),
        ("The academic record lists ", " as the major."),
        ("An honors program in ", " capped the degree."),
    ],
    "employer": [
        ("{name} works for ", "."),
        ("{name} is employed at ", "."),
        ("The current employer on file is ", "."),
        ("A staff badge from ", " hangs by the door."),
        ("{name} joined ", " after graduation."),
        ("At ", ", {name} keeps a busy schedule."),
        ("Payroll records name ", " as the employer of {name}."),
        ("The weekly commute leads to ", "."),
        ("Their career continues at ", "."),
        ("The office belongs to ", "."),
        ("A position at ", " fills the working week."),
        ("Business cards carry the logo of ", "."),
        ("The employer of record is ", "."),
        ("Workdays are spent at ", "."),
        ("The company on the contract is ", "."),
        ("Colleagues at ", " know the name well."),
        ("A desk at ", " holds the daily work."),
        ("Professional life is anchored at ", "."),
        ("The paycheck arrives from ", "."),
        ("Employment records point to ", "."),
    ],
    "current_city": [
        ("{name} currently lives in ", "."),
        ("{name} now makes a home in ", "."),
        ("The current address sits in ", "."),
        ("Home these days is ", "."),
        ("{name} settled in ", " some years ago."),
        ("In ", ", {name} rents a small place near the center."),
        ("The city of residence on file for {name} is ", "."),
        ("Mail now arrives at an address in ", "."),
        ("Daily life plays out in ", "."),
        ("The present neighborhood lies in ", "."),
        ("Their apartment is in ", "."),
        ("Residence records show ", "."),
        ("These days the doorbell rings in ", "."),
        ("Life today unfolds in ", "."),
        ("The current hometown is ", "."),
        ("A lease in ", " marks the present chapter."),
        ("The city they call home is ", "."),
        ("Evenings are spent at home in ", "."),
        ("The voter roll lists an address in ", "."),
        ("Utility bills arrive in ", "."),
    ],
}

ATTRIBUTE_ORDERINGS: list[tuple[str, ...]] = [
    ("birth_date", "birth_city", "university", "major", "employer", "current_city"),
    ("birth_city", "birth_date", "major", "university", "current_city", "employer"),
    ("university", "major", "birth_date", "birth_city", "employer", "current_city"),
    ("current_city", "employer", "university", "major", "birth_city", "birth_date"),
    ("major", "university", "employer", "current_city", "birth_date", "birth_city"),
    ("employer", "current_city", "birth_date", "major", "birth_city", "university"),
    ("birth_date", "university", "birth_city", "employer", "major", "current_city"),
    ("birth_city", "major", "current_city", "birth_date", "university", "employer"),
]


def _surface_forms(name: str) -> list[str]:
    first, _middle, last = name.split(" ")
    return [name, f"{first} {last}", f"{last}, {first}"]


def render_bio_doc(rec: BioRecord, exposure_idx: int) -> Doc:
    """One biography paragraph; deterministic in (entity_id, exposure_idx).

    String seeding uses random.Random(str) which hashes the string bytes
    (PYTHONHASHSEED-independent), so renders are stable across processes.
    """
    rng = random.Random(f"bio:{rec.entity_id}:{exposure_idx}")
    ordering = ATTRIBUTE_ORDERINGS[rng.randrange(len(ATTRIBUTE_ORDERINGS))]
    surfaces = _surface_forms(rec.name)

    dense_parts: list[str] = []
    split_segs: list[Segment] = []

    def push_plain(text: str) -> None:
        if not text:
            return
        dense_parts.append(text)
        if split_segs and split_segs[-1][1] is False:
            split_segs[-1] = (split_segs[-1][0] + text, False)
        else:
            split_segs.append((text, False))

    for pos, attr in enumerate(ordering):
        templates = BIO_TEMPLATES[attr]
        if pos == 0:
            named = [t for t in templates if "{name}" in t[0] + t[1]]
            prefix, suffix = named[rng.randrange(len(named))]
            surface = rec.name  # first sentence always the canonical full name
        else:
            prefix, suffix = templates[rng.randrange(len(templates))]
            surface = surfaces[rng.randrange(len(surfaces))]
        prefix = prefix.format(name=surface)
        suffix = suffix.format(name=surface)
        value = rec.attrs[attr]
        lead = "" if pos == 0 else " "
        # prefix ends with " "; the masked value segment carries that space
        push_plain(lead + prefix[:-1])
        dense_parts.append(" " + value)
        for seg_text, masked in lookup_segments(rec.name, attr, value):
            if masked:
                split_segs.append((seg_text, True))
            else:
                push_seg_plain_only_to_split(split_segs, seg_text)
        push_plain(suffix)

    return Doc(
        kind="bio",
        dense_segments=[("".join(dense_parts), False)],
        split_segments=split_segs,
        meta={"entity_id": rec.entity_id, "exposure": exposure_idx},
    )


def push_seg_plain_only_to_split(split_segs: list[Segment], text: str) -> None:
    """Append an unmasked segment to the split rendering only (special tokens
    and lookup queries exist only in the split arm)."""
    split_segs.append((text, False))


# ---------------------------------------------------------------- recall probes


def recall_probes(records: list[BioRecord], n_entities_sampled: int, seed: int) -> list[QAItem]:
    rng = random.Random(seed)
    sampled = rng.sample(records, min(n_entities_sampled, len(records)))
    probes: list[QAItem] = []
    for rec in sampled:
        for attr in ATTRIBUTES:
            probes.append(
                QAItem(
                    qid=f"recall-{rec.entity_id}-{attr}",
                    task="recall",
                    prompt=f"{rec.name}'s {RELATION_PHRASES[attr]} is",
                    answer=rec.attrs[attr],
                    meta={"entity_id": rec.entity_id, "relation": attr,
                          "template": f"recall-{attr}"},
                )
            )
    return probes
