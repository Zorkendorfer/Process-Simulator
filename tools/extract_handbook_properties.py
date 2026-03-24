#!/usr/bin/env python3
"""Extract a first-pass property library from handbook references.

This script is intentionally conservative:
- Perry's Handbook: extracts TABLE 2-2 records from the PDF with a
  column-aware parser.
- CRC Handbook: extracts structured element/substance summary lines from
  the text dump.

The output keeps source metadata and raw text values so downstream code can
decide how aggressively to normalize or trust each field.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PERRY_PATH = DATA_DIR / "Perrys_Chemical_Engineers_Handbook.txt"
CRC_PATH = DATA_DIR / "CRC.Press.Handbook.of.Chemistry.and.Physics.85th.ed.eBook-LRN.txt"
PERRY_PDF_PATH = DATA_DIR / "Perrys_Chemical_Engineers_Handbook.pdf"
COMPONENTS_PATH = DATA_DIR / "components.json"
OUTPUT_PATH = DATA_DIR / "handbook_property_library.json"
COMPONENT_CANDIDATES_PATH = DATA_DIR / "components_handbook_candidates.json"


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
FORMULA_RE = re.compile(r"[A-Z][a-z]?")
WEIGHT_RE = re.compile(r"^[0-9.\-−–]+(?:\s*[A-Za-z./°]+)?$")
CRC_ENTRY_RE = re.compile(
    r"^([A-Z][A-Za-z]+)\s*[-—]\s*.*?at\. wt\. ([^;]+); at\.? no\. ([^;]+);(.*)$"
)
STEM_SUFFIXES = [
    "sulfonic chloride",
    "sulfonic amide",
    "sulfonic acid",
    "carboxylic acid",
    "furfuryl alcohol",
    "hydrochloride",
    "hydrobromide",
    "hydroiodide",
    "dithiocarbamate",
    "benzoate",
    "butyrate",
    "chloride",
    "bromide",
    "iodide",
    "nitrate",
    "nitrite",
    "sulfate",
    "phosphate",
    "formate",
    "acetate",
    "alcohol",
    "aldehyde",
    "anhydride",
    "aniline",
    "amide",
    "amine",
    "phenol",
    "acid",
]
NAME_REPLACEMENTS = {
    "Anisicacid": "Anisic acid",
    "Hippuricacid": "Hippuric acid",
    "carboxylicacid": "carboxylic acid",
    "sulfonicacid": "sulfonic acid",
    "sulfonicamide": "sulfonic amide",
    "sulfonicchloride": "sulfonic chloride",
}
SOLUBILITY_MAP = {
    "i.": "insoluble",
    "∞": "miscible",
    "s.": "soluble",
    "sl.s.": "slightly soluble",
    "v.sl.s.": "very slightly soluble",
    "v.s.": "very soluble",
    "s.h.": "soluble when hot",
}


def clean_text(value: str) -> str:
    value = value.replace("\u2014", "-")
    value = value.replace("\u2013", "-")
    value = value.replace("\u2212", "-")
    value = value.replace("(cid:7)", "")
    value = value.replace("(cid:6)", "")
    value = value.replace("(cid:1)", "")
    value = CONTROL_CHARS_RE.sub("", value)
    return " ".join(value.split()).strip()


def maybe_float(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def slugify(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "entry"


def normalize_compound_name(value: str) -> str:
    name = clean_text(value)
    for old, new in NAME_REPLACEMENTS.items():
        name = name.replace(old, new)
    for suffix in STEM_SUFFIXES:
        if " " in suffix:
            continue
        name = re.sub(
            rf"(?<=[a-z0-9\)])({re.escape(suffix)})(?=(?:\(|\b))",
            r" \1",
            name,
            flags=re.IGNORECASE,
        )
    return clean_text(name)


def derive_name_stem(name: str) -> str:
    stem = normalize_compound_name(name)
    while True:
        updated = re.sub(r"\s*\([^()]*\)\s*$", "", stem)
        if updated == stem:
            break
        stem = updated

    lowered = stem.lower()
    for suffix in sorted(STEM_SUFFIXES, key=len, reverse=True):
        token = f" {suffix}"
        if lowered.endswith(token):
            return clean_text(stem[: -len(token)])
    return stem


def normalize_formula(value: str) -> str:
    formula = clean_text(value).replace(" ", "")
    formula = formula.replace("><", "")
    formula = re.sub(r"HCH(\d)(\d)$", r"H\1CH\2", formula)
    return formula


def parse_number_range(value: str) -> tuple[float, float] | None:
    value = clean_text(value)
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", value)
    if not match:
        return None

    left_text, right_text = match.groups()
    left = float(left_text)
    if "." in right_text or len(right_text) >= len(left_text.lstrip("-")):
        right = float(right_text)
    else:
        integer_part = str(abs(int(left))).rstrip("0123456789")
        if integer_part:
            right = float(f"{integer_part}{right_text}")
        else:
            base = int(left // 10) * 10
            right = float(base + int(right_text))
    return (left, right)


def parse_temperature_fields(prefix: str, raw: str) -> dict:
    raw = clean_text(raw)
    if not raw:
        return {}

    text = raw.replace(" ", "")
    fields: dict[str, object] = {f"{prefix}_raw": raw}

    if text.startswith("d."):
        fields[f"{prefix}_decomposes"] = True
        text = text[2:]
    elif text == "d.":
        fields[f"{prefix}_decomposes"] = True
        return fields

    if not text:
        return fields

    if text[:1] in "<>":
        fields[f"{prefix}_qualifier"] = text[0]
        text = text[1:]

    if not text:
        return fields

    range_values = parse_number_range(text)
    if range_values:
        fields[f"{prefix}_c_min"] = range_values[0]
        fields[f"{prefix}_c_max"] = range_values[1]
        fields[f"{prefix}_c"] = (range_values[0] + range_values[1]) / 2.0
        return fields

    value_float = maybe_float(text)
    if value_float is not None:
        fields[f"{prefix}_c"] = value_float
    return fields


def parse_specific_gravity(raw: str) -> dict:
    raw = clean_text(raw)
    if not raw:
        return {}

    fields: dict[str, object] = {"specific_gravity_raw": raw}
    match = re.fullmatch(r"([0-9.]+?)(\d{2}/\d{1,2})", raw)
    if match:
        fields["specific_gravity"] = float(match.group(1))
        fields["specific_gravity_reference"] = match.group(2)
        return fields

    match = re.fullmatch(r"(\d{2}/\d{1,2})\s+([0-9.]+)", raw)
    if match:
        fields["specific_gravity_reference"] = match.group(1)
        fields["specific_gravity"] = float(match.group(2))
        return fields

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw)
    if match:
        fields["specific_gravity"] = float(match.group(1))
    return fields


def normalize_solubility(raw: str) -> dict:
    raw = clean_text(raw)
    if not raw:
        return {}
    return {
        "raw": raw,
        "class": SOLUBILITY_MAP.get(raw),
    }


def is_formula_block(block: list[str]) -> bool:
    if not block:
        return False
    good = 0
    for line in block:
        if any(ch.isdigit() for ch in line) and FORMULA_RE.search(line):
            good += 1
    return good >= max(1, int(len(block) * 0.7))


def is_weight_block(block: list[str]) -> bool:
    if not block:
        return False
    good = sum(bool(WEIGHT_RE.match(clean_text(line))) for line in block)
    return good >= max(1, int(len(block) * 0.8))


def is_form_block(block: list[str]) -> bool:
    if not block:
        return False
    good = 0
    for line in block:
        line = clean_text(line)
        if line and not any(ch.isdigit() for ch in line) and any(ch.isalpha() for ch in line):
            good += 1
    return good >= max(1, int(len(block) * 0.7))


def expand_perry_names(raw_names: list[str]) -> list[dict]:
    records = []
    current_root = ""
    current_stem = ""
    last_name = ""

    for raw in raw_names:
        raw = clean_text(raw)
        if not raw:
            continue

        expanded = raw
        quality = "exact"

        if raw.startswith("(") and last_name:
            if re.search(r"\([^()]*\)\s*$", last_name):
                expanded = re.sub(r"\([^()]*\)\s*$", raw, last_name)
            else:
                expanded = f"{last_name} {raw}"
            quality = "inherited_variant"
        elif raw.startswith("-") and current_root:
            expanded = current_root + raw[1:]
            quality = "inherited_prefix"
        elif raw[:1].islower() and current_stem:
            expanded = f"{current_stem} {raw}"
            quality = "inherited_suffix"

        if not raw.startswith(("(", "-")) and not raw[:1].islower():
            token = raw.split()[0]
            current_root = token.split("-")[0] if "-" in token else token
            current_stem = derive_name_stem(raw)

        records.append(
            {
                "raw_name": raw,
                "name": normalize_compound_name(expanded),
                "name_parse_quality": quality,
            }
        )
        last_name = normalize_compound_name(expanded)

    return records


def split_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        line = clean_text(line)
        if line:
            current.append(line)
        elif current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)
    return blocks


def extract_perry_records() -> list[dict]:
    column_ranges = [
        ("name", 0, 150),
        ("synonym", 150, 250),
        ("formula", 250, 355),
        ("formula_weight", 355, 395),
        ("form_color", 395, 445),
        ("specific_gravity_raw", 445, 490),
        ("melting_point_raw", 490, 535),
        ("boiling_point_raw", 535, 580),
        ("solubility_water_raw", 580, 622),
        ("solubility_alcohol_raw", 622, 662),
        ("solubility_ether_raw", 662, 720),
    ]
    table_pages = [73, 75, 77, 79, 81, 83, 85, 87, 89]

    def column_for(x_mid: float):
        for name, lo, hi in column_ranges:
            if lo <= x_mid < hi:
                return name
        return None

    def cluster_lines(words: list[dict], tolerance: float = 2.2) -> list[dict]:
        words = sorted(words, key=lambda w: (w["top"], w["x0"]))
        lines: list[dict] = []
        for word in words:
            y_mid = (word["top"] + word["bottom"]) / 2.0
            if not lines or abs(y_mid - lines[-1]["y"]) > tolerance:
                lines.append({"y": y_mid, "words": [word]})
            else:
                count = len(lines[-1]["words"])
                lines[-1]["y"] = (lines[-1]["y"] * count + y_mid) / (count + 1)
                lines[-1]["words"].append(word)
        return lines

    def join_words(words: list[dict]) -> str:
        words = sorted(words, key=lambda w: w["x0"])
        output = ""
        previous = None

        for word in words:
            text = clean_text(word["text"])
            if not text:
                continue

            if previous is None:
                output = text
            else:
                gap = word["x0"] - previous["x1"]
                needs_space = gap > 3.5
                if output.endswith(",") and text[:1].isalnum():
                    needs_space = True
                output += (" " if needs_space else "") + text
            previous = word

        return clean_text(output)

    extracted: list[dict] = []

    with pdfplumber.open(PERRY_PDF_PATH) as pdf:
        for pdf_page in table_pages:
            page = pdf.pages[pdf_page - 1]
            words = [
                word
                for word in page.extract_words(use_text_flow=False, keep_blank_chars=False)
                if 68 <= word["top"] <= 520
            ]

            current = None
            for line in cluster_lines(words):
                buckets: dict[str, list[dict]] = {}
                for word in line["words"]:
                    column = column_for((word["x0"] + word["x1"]) / 2.0)
                    if column is None:
                        continue
                    buckets.setdefault(column, []).append(word)

                text_by_column = {column: join_words(bucket) for column, bucket in buckets.items()}
                starts_row = bool(text_by_column.get("formula")) and bool(
                    text_by_column.get("name") or text_by_column.get("synonym")
                )

                if starts_row:
                    current = {
                        "raw_name": "",
                        "raw_synonym": "",
                        "formula": "",
                        "formula_weight_raw": "",
                        "form_color": "",
                        "specific_gravity_raw": "",
                        "melting_point_raw": "",
                        "boiling_point_raw": "",
                        "solubility_water_raw": "",
                        "solubility_alcohol_raw": "",
                        "solubility_ether_raw": "",
                        "source": {
                            "handbook": "Perry's Chemical Engineers' Handbook",
                            "table": "TABLE 2-2",
                            "pdf_page": pdf_page,
                        },
                    }
                    extracted.append(current)

                if current is None:
                    continue

                for field in [
                    "name",
                    "synonym",
                    "formula",
                    "formula_weight",
                    "form_color",
                    "specific_gravity_raw",
                    "melting_point_raw",
                    "boiling_point_raw",
                    "solubility_water_raw",
                    "solubility_alcohol_raw",
                    "solubility_ether_raw",
                ]:
                    if field not in text_by_column:
                        continue
                    target = {
                        "name": "raw_name",
                        "synonym": "raw_synonym",
                        "formula_weight": "formula_weight_raw",
                    }.get(field, field)
                    current[target] = clean_text(f"{current[target]} {text_by_column[field]}")

    extracted = [record for record in extracted if clean_text(record["raw_name"])]

    expanded_names = expand_perry_names([record["raw_name"] for record in extracted])
    for record, expanded in zip(extracted, expanded_names):
        record["name"] = expanded["name"]
        record["name_parse_quality"] = expanded["name_parse_quality"]
        if record["formula"]:
            record["formula"] = record["formula"].replace(" ", "")
        if record["raw_synonym"]:
            record["synonym"] = clean_text(record["raw_synonym"])
        if record["formula_weight_raw"]:
            record["formula_weight"] = maybe_float(record["formula_weight_raw"])

    counts = Counter()
    for record in extracted:
        base = slugify(record["name"])
        counts[base] += 1
        suffix = f"_{counts[base]}" if counts[base] > 1 else ""
        record["id"] = f"perry_{base}{suffix}"

    return extracted


def extract_crc_records() -> list[dict]:
    records: list[dict] = []

    for raw_line in CRC_PATH.read_text(errors="ignore").splitlines():
        line = clean_text(raw_line)
        match = CRC_ENTRY_RE.match(line)
        if not match:
            continue

        name, atomic_weight, atomic_number, tail = match.groups()
        record = {
            "id": f"crc_{slugify(name)}",
            "name": clean_text(name),
            "source": {
                "handbook": "CRC Handbook of Chemistry and Physics",
                "section": "The Elements",
            },
            "atomic_weight_raw": clean_text(atomic_weight),
            "atomic_number_raw": clean_text(atomic_number),
            "raw_summary": clean_text(tail),
        }

        atomic_number_value = maybe_float(clean_text(atomic_number))
        if atomic_number_value is not None and atomic_number_value.is_integer():
            record["atomic_number"] = int(atomic_number_value)

        mp = re.search(r"(?:m\.p\.|f\.p\.)\s*([^;,]+(?:\([^)]*\))?)", tail)
        bp = re.search(r"b\.p\.\s*([^;]+)", tail)
        tc = re.search(r"tc\s*([^;,.]+(?:\.[0-9]+)?)", tail)
        density = re.search(r"density\s*([0-9.]+(?:\s*[A-Za-z/0-9]+)?)", tail)
        sp_gr = re.search(r"sp\. gr\.\s*([0-9.]+(?:\s*\([^)]*\))?)", tail)
        valence = re.search(r"valence\s*([^;]+)", tail)

        if mp:
            record["melting_point_raw"] = clean_text(mp.group(1))
        if bp:
            record["boiling_point_raw"] = clean_text(bp.group(1))
        if tc:
            record["critical_temperature_raw"] = clean_text(tc.group(1))
        if density:
            record["density_raw"] = clean_text(density.group(1))
        if sp_gr:
            record["specific_gravity_raw"] = clean_text(sp_gr.group(1))
        if valence:
            record["valence_raw"] = clean_text(valence.group(1))

        records.append(record)

    return records


def build_normalized_library(perry_records: list[dict], crc_records: list[dict]) -> dict:
    normalized: dict[str, dict] = {}

    for record in perry_records:
        entry = {
            "source_type": "perry_table_2_2",
            "name": record["name"],
            "source": record["source"],
            "formula_raw": record.get("formula"),
            "formula": normalize_formula(record["formula"]) if record.get("formula") else None,
            "name_parse_quality": record.get("name_parse_quality"),
        }

        if record.get("synonym"):
            entry["aliases"] = [clean_text(alias) for alias in record["synonym"].split(",") if clean_text(alias)]
        if record.get("formula_weight") is not None:
            entry["MW"] = record["formula_weight"]
        if record.get("form_color"):
            entry["appearance_raw"] = record["form_color"]

        entry.update(parse_specific_gravity(record.get("specific_gravity_raw", "")))
        entry.update(parse_temperature_fields("melting_point", record.get("melting_point_raw", "")))
        entry.update(parse_temperature_fields("boiling_point", record.get("boiling_point_raw", "")))

        solubility = {}
        for phase_key, raw_key in [
            ("water", "solubility_water_raw"),
            ("alcohol", "solubility_alcohol_raw"),
            ("ether", "solubility_ether_raw"),
        ]:
            normalized_solubility = normalize_solubility(record.get(raw_key, ""))
            if normalized_solubility:
                solubility[phase_key] = normalized_solubility
        if solubility:
            entry["solubility"] = solubility

        normalized[record["id"]] = entry

    for record in crc_records:
        entry = {
            "source_type": "crc_elements_section",
            "name": record["name"],
            "source": record["source"],
        }
        if record.get("atomic_number") is not None:
            entry["atomic_number"] = record["atomic_number"]
        if record.get("atomic_weight_raw"):
            entry["atomic_weight_raw"] = record["atomic_weight_raw"]
        if record.get("density_raw"):
            entry["density_raw"] = record["density_raw"]
        if record.get("specific_gravity_raw"):
            entry["specific_gravity_raw"] = record["specific_gravity_raw"]
        entry.update(parse_temperature_fields("melting_point", record.get("melting_point_raw", "")))
        entry.update(parse_temperature_fields("boiling_point", record.get("boiling_point_raw", "")))
        if record.get("critical_temperature_raw"):
            critical = maybe_float(clean_text(record["critical_temperature_raw"]))
            if critical is not None:
                entry["critical_temperature_c"] = critical
            entry["critical_temperature_raw"] = record["critical_temperature_raw"]
        normalized[record["id"]] = entry

    return normalized


def build_components_overlay(perry_records: list[dict]) -> dict:
    components = json.loads(COMPONENTS_PATH.read_text())
    perry_by_name = {record["name"].lower(): record for record in perry_records}
    overlay: dict[str, dict] = {}

    for component_key, component in components.items():
        record = perry_by_name.get(component["name"].lower())
        if not record:
            continue

        overlay[component_key] = {
            "name": record["name"],
            "MW": record.get("formula_weight"),
            "formula": normalize_formula(record["formula"]) if record.get("formula") else None,
            "source": record["source"],
        }

        overlay[component_key].update(parse_specific_gravity(record.get("specific_gravity_raw", "")))
        overlay[component_key].update(
            {
                key: value
                for key, value in parse_temperature_fields(
                    "handbook_melting_point", record.get("melting_point_raw", "")
                ).items()
            }
        )
        overlay[component_key].update(
            {
                key: value
                for key, value in parse_temperature_fields(
                    "handbook_boiling_point", record.get("boiling_point_raw", "")
                ).items()
            }
        )
        if record.get("form_color"):
            overlay[component_key]["appearance_raw"] = record["form_color"]

    return overlay


def build_component_candidates(perry_records: list[dict], existing_components: dict) -> dict:
    existing_names = {component["name"].lower() for component in existing_components.values()}
    candidates: dict[str, dict] = {}

    for record in perry_records:
        if record["name"].lower() in existing_names:
            continue
        if record.get("formula_weight") is None:
            continue

        component_id = slugify(record["name"]).upper().replace("_", "-")
        candidate = {
            "name": record["name"],
            "source_record_id": record["id"],
            "formula": normalize_formula(record["formula"]) if record.get("formula") else None,
            "MW": record["formula_weight"],
            "source": record["source"],
            "simulator_ready": False,
            "missing_fields": [
                "Tc",
                "Pc",
                "omega",
                "Cp1",
                "Cp2",
                "Cp3",
                "Cp4",
                "Cp5",
                "Antoine_A",
                "Antoine_B",
                "Antoine_C",
            ],
        }

        candidate.update(parse_specific_gravity(record.get("specific_gravity_raw", "")))
        candidate.update(
            {
                key: value
                for key, value in parse_temperature_fields(
                    "handbook_melting_point", record.get("melting_point_raw", "")
                ).items()
            }
        )
        candidate.update(
            {
                key: value
                for key, value in parse_temperature_fields(
                    "handbook_boiling_point", record.get("boiling_point_raw", "")
                ).items()
            }
        )
        if record.get("form_color"):
            candidate["appearance_raw"] = record["form_color"]
        if record.get("synonym"):
            candidate["aliases"] = [clean_text(alias) for alias in record["synonym"].split(",") if clean_text(alias)]

        candidates[component_id] = candidate

    return candidates


def main() -> None:
    perry_records = extract_perry_records()
    crc_records = extract_crc_records()
    normalized_library = build_normalized_library(perry_records, crc_records)
    existing_components = json.loads(COMPONENTS_PATH.read_text())
    components_overlay = build_components_overlay(perry_records)
    component_candidates = build_component_candidates(perry_records, existing_components)

    output = {
        "metadata": {
            "generated_by": "tools/extract_handbook_properties.py",
            "notes": [
                "This is a conservative first-pass extraction from handbook PDF/text references.",
                "Perry records come from TABLE 2-2 in Perry's PDF using a column-aware parser.",
                "CRC records come from structured element/substance summary lines in the 'The Elements' section of the text dump.",
                "Most values are kept as raw handbook strings to avoid false precision from aggressive normalization.",
            ],
            "perry_record_count": len(perry_records),
            "crc_record_count": len(crc_records),
            "normalized_record_count": len(normalized_library),
            "component_overlay_count": len(components_overlay),
            "component_candidate_count": len(component_candidates),
        },
        "perry_table_2_2_first_pass": perry_records,
        "crc_reference_substances": crc_records,
        "normalized_compound_library": normalized_library,
        "components_overlay": components_overlay,
        "components_handbook_candidates": component_candidates,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n")
    COMPONENT_CANDIDATES_PATH.write_text(json.dumps(component_candidates, indent=2, ensure_ascii=True) + "\n")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {COMPONENT_CANDIDATES_PATH}")
    print(f"Perry records: {len(perry_records)}")
    print(f"CRC records:   {len(crc_records)}")


if __name__ == "__main__":
    main()
