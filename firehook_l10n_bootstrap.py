#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Firehook iOS Localization Bootstrap (best-practice version)

- Scan a root directory recursively for .swift files
- Extract UI-facing string literals (SwiftUI + partial UIKit)
- Generate:
  - stable namespaced keys (snake_case)
  - EN reference text (fallback)
  - suggestions for FR/DE/ES/IT/NL (shown; files kept blank by default)
- Dry-run by default: prints what would happen for first N unique terms
- Apply mode:
  - writes L10n/<lang>.lproj/Localizable.strings
    - EN filled
    - other langs blank unless already existing (or --fill-suggestions)
  - rewrites Swift sources:
    - SwiftUI: replaces literal inside quotes with key ("hooks_add")
    - UIKit alert title/message: replaces "literal" with NSLocalizedString("key", comment: "")

Safety:
- Skips URLs, interpolated strings \(…), and code-like tokens
- Optional .bak backups when rewriting
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

# -------------------------
# Languages (Tier 1 + 2)
# -------------------------
LANGS = ["en", "fr", "de", "es", "it", "nl"]

# -------------------------
# Brand/technical tokens to keep (no literal translation)
# We'll also enforce casing per language.
# -------------------------
GLOSSARY_TOKENS = [
    "Firehook",
    "Hook", "Hooks",
    "Webhook", "Webhooks",
    "Deeplink", "Deeplinks",
    "Bluetooth",
    "CarPlay",
    "Android Auto",
    "API", "URL", "HTTP",
    "Wi-Fi", "WiFi",
]

# Per-language preferred casing for "Hook(s)" concept:
# - EN: Hook/Hooks
# - FR/ES/IT/NL: hook/hooks (more natural in sentences)
# - DE: Hook/Hooks (nouns capitalized)
HOOK_CASING = {
    "en": {"hook": "Hook", "hooks": "Hooks"},
    "fr": {"hook": "hook", "hooks": "hooks"},
    "es": {"hook": "hook", "hooks": "hooks"},
    "it": {"hook": "hook", "hooks": "hooks"},
    "nl": {"hook": "hook", "hooks": "hooks"},
    "de": {"hook": "Hook", "hooks": "Hooks"},
}

# Always keep these technical terms with exact casing:
TERM_FIXED = {
    "firehook": "Firehook",
    "webhook": "Webhook",
    "webhooks": "Webhooks",
    "deeplink": "Deeplink",
    "deeplinks": "Deeplinks",
    "bluetooth": "Bluetooth",
    "carplay": "CarPlay",
    "android auto": "Android Auto",
    "wifi": "WiFi",
    "wi-fi": "Wi-Fi",
    "api": "API",
    "url": "URL",
    "http": "HTTP",
}

# -------------------------
# Common UI strings => fixed keys (prevents collisions)
# Suggestions are always available (even if we keep non-EN files blank).
# -------------------------
COMMON_MAP: Dict[str, Dict[str, str]] = {
    "Cancel":    {"key": "common_cancel",    "en": "Cancel",    "fr": "Annuler",     "de": "Abbrechen",  "es": "Cancelar",  "it": "Annulla",    "nl": "Annuleren"},
    "OK":        {"key": "common_ok",        "en": "OK",        "fr": "OK",          "de": "OK",         "es": "OK",        "it": "OK",         "nl": "OK"},
    "Done":      {"key": "common_done",      "en": "Done",      "fr": "Terminé",     "de": "Fertig",     "es": "Listo",     "it": "Fine",       "nl": "Gereed"},
    "Save":      {"key": "common_save",      "en": "Save",      "fr": "Enregistrer", "de": "Sichern",    "es": "Guardar",   "it": "Salva",      "nl": "Bewaren"},
    "Edit":      {"key": "common_edit",      "en": "Edit",      "fr": "Modifier",    "de": "Bearbeiten", "es": "Editar",    "it": "Modifica",    "nl": "Bewerken"},
    "Delete":    {"key": "common_delete",    "en": "Delete",    "fr": "Supprimer",   "de": "Löschen",    "es": "Eliminar",  "it": "Elimina",     "nl": "Verwijderen"},
    "Duplicate": {"key": "common_duplicate", "en": "Duplicate", "fr": "Dupliquer",   "de": "Duplizieren", "es": "Duplicar",  "it": "Duplica",     "nl": "Dupliceren"},
    "Close":     {"key": "common_close",     "en": "Close",     "fr": "Fermer",      "de": "Schließen",  "es": "Cerrar",    "it": "Chiudi",      "nl": "Sluiten"},
    "Back":      {"key": "common_back",      "en": "Back",      "fr": "Retour",      "de": "Zurück",     "es": "Atrás",     "it": "Indietro",    "nl": "Terug"},
}

# -------------------------
# Extraction patterns with "mode"
# - mode="swiftui_key": replace literal inside quotes with key
# - mode="uikit_ns": replace the entire quoted literal with NSLocalizedString("key", comment:"")
# -------------------------
PATTERNS = [
    ("swiftui_key", re.compile(r'\bText\(\s*"(?P<s>(?:\\.|[^"\\])*)"\s*\)')),
    ("swiftui_key", re.compile(r'\bButton\(\s*"(?P<s>(?:\\.|[^"\\])*)"\s*(?:,|\))')),
    ("swiftui_key", re.compile(r'\bLabel\(\s*"(?P<s>(?:\\.|[^"\\])*)"\s*,')),
    ("swiftui_key", re.compile(r'\bNavigationLink\(\s*"(?P<s>(?:\\.|[^"\\])*)"\s*(?:,|\))')),
    ("swiftui_key", re.compile(r'\.navigationTitle\(\s*"(?P<s>(?:\\.|[^"\\])*)"\s*\)')),
    ("swiftui_key", re.compile(r'\.navigationBarTitle\(\s*"(?P<s>(?:\\.|[^"\\])*)"\s*(?:,|\))')),
    ("swiftui_key", re.compile(r'\.alert\(\s*"(?P<s>(?:\\.|[^"\\])*)"\s*(?:,|\))')),
    ("swiftui_key", re.compile(r'\.confirmationDialog\(\s*"(?P<s>(?:\\.|[^"\\])*)"\s*(?:,|\))')),
    # UIKit: UIAlertController title/message are plain strings → must wrap
    ("uikit_ns", re.compile(r'\bUIAlertController\([^)]*title:\s*"(?P<s>(?:\\.|[^"\\])*)"')),
    ("uikit_ns", re.compile(r'\bUIAlertController\([^)]*message:\s*"(?P<s>(?:\\.|[^"\\])*)"')),
]

# -------------------------
# Ignore heuristics
# -------------------------
RE_URL = re.compile(r'^(https?:\/\/|mailto:|tel:)', re.IGNORECASE)
RE_HAS_INTERP = re.compile(r'\\\(')  # \(...) interpolation
RE_CODELIKE = re.compile(r'^[A-Za-z0-9_\-\/\.]+$')
RE_KEYLIKE = re.compile(r'^[a-z][a-z0-9_]{2,}$')  # already looks like a localization key

EXCLUDE_DIRS_DEFAULT = {"Pods", "Carthage", "DerivedData", ".build", "build", ".git"}

# .strings parser
STRINGS_LINE = re.compile(r'^\s*"(?P<k>(?:\\.|[^"\\])*)"\s*=\s*"(?P<v>(?:\\.|[^"\\])*)"\s*;\s*$')

def unescape_swift(s: str) -> str:
    return (s.replace(r'\"', '"')
             .replace(r'\n', '\n')
             .replace(r'\t', '\t')
             .replace(r'\\', '\\'))

def escape_strings(s: str) -> str:
    return (s.replace('\\', r'\\')
             .replace('"', r'\"')
             .replace('\n', r'\n'))

def normalize_ws(s: str) -> str:
    return s.replace("\u00a0", " ").strip()

def normalize_punct(lang: str, s: str) -> str:
    """
    Keep punctuation stable across langs, avoid NBSP behavior issues.
    Pragmatic rule: remove spaces before ?, !, : in *all* langs.
    """
    s = normalize_ws(s)
    s = re.sub(r"\s+([?!:;])", r"\1", s)
    return s

def enforce_fixed_terms(s: str) -> str:
    """
    Enforce technical terms casing (WebHook->Webhook etc.) without changing meaning.
    """
    s = normalize_ws(s)
    # multiword first
    s = re.sub(r"\bandroid\s+auto\b", TERM_FIXED["android auto"], s, flags=re.IGNORECASE)
    # single tokens
    for src, dst in TERM_FIXED.items():
        if " " in src:
            continue
        s = re.sub(rf"\b{re.escape(src)}\b", dst, s, flags=re.IGNORECASE)
    return s

def enforce_hook_casing(lang: str, s: str) -> str:
    s = normalize_ws(s)
    casing = HOOK_CASING.get(lang, HOOK_CASING["en"])
    # plural first
    s = re.sub(r"\bhooks\b", casing["hooks"], s, flags=re.IGNORECASE)
    s = re.sub(r"\bhook\b", casing["hook"], s, flags=re.IGNORECASE)
    return s

def postprocess_text(lang: str, s: str) -> str:
    s = enforce_fixed_terms(s)
    s = enforce_hook_casing(lang, s)
    s = normalize_punct(lang, s)
    return s

def slug_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    if not s:
        s = "text"
    if s[0].isdigit():
        s = f"t_{s}"
    return s

def should_ignore(text: str) -> bool:
    t = text.strip()
    if len(t) < 2:
        return True
    if RE_URL.match(t):
        return True
    if RE_HAS_INTERP.search(t):
        return True
    if RE_KEYLIKE.match(t):  # already looks like a l10n key → don’t touch
        return True
    if RE_CODELIKE.match(t) and any(ch in t for ch in "/._-"):
        return True
    return False

@dataclass
class Occurrence:
    file: Path
    line_no: int
    line_text: str
    literal: str
    span: Tuple[int, int]        # span of captured literal content
    mode: str                    # "swiftui_key" or "uikit_ns"

def iter_swift_files(root: Path, exclude_dirs: set) -> List[Path]:
    return [p for p in root.rglob("*.swift") if not any(part in exclude_dirs for part in p.parts)]

def extract_occurrences(swift_file: Path) -> List[Occurrence]:
    occs: List[Occurrence] = []
    lines = swift_file.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    for i, line in enumerate(lines, start=1):
        for mode, rx in PATTERNS:
            for m in rx.finditer(line):
                raw = m.group("s")
                lit = unescape_swift(raw)
                if should_ignore(lit):
                    continue
                occs.append(Occurrence(swift_file, i, line, lit, m.span("s"), mode))
    return occs

def load_existing_strings(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = STRINGS_LINE.match(line)
        if m:
            k = unescape_swift(m.group("k"))
            v = unescape_swift(m.group("v"))
            out[k] = v
    return out

def rewrite_line(line: str, edits: List[Tuple[Tuple[int, int], str]]) -> str:
    """
    Apply replacements from right to left
    edits: [ ((start,end), replacement) ]
    """
    out = line
    for (start, end), repl in sorted(edits, key=lambda x: x[0][0], reverse=True):
        out = out[:start] + repl + out[end:]
    return out

# -------------------------
# Namespacing heuristic (simple but effective)
# -------------------------
def guess_namespace(occ: Occurrence) -> str:
    name = occ.file.name.lower()
    if "variable" in name or "vars" in name:
        return "vars"
    if "hook" in name:
        return "hooks"
    if "account" in name or "login" in name:
        return "account"
    if "settings" in name:
        return "settings"
    return "common"

# -------------------------
# OpenAI normalization (Structured Outputs)
# -------------------------
def ai_normalize(requests: List[Dict]) -> Dict[str, Dict[str, str]]:
    """
    requests item:
      { "id": "...", "original": "...", "namespace": "...", "max_chars": int }
    returns:
      id -> { key, en, fr, de, es, it, nl }
    """
    from openai import OpenAI
    from pydantic import BaseModel
    from typing import List as TList

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    class Item(BaseModel):
        id: str
        key: str
        en: str
        fr: str
        de: str
        es: str
        it: str
        nl: str

    class Out(BaseModel):
        items: TList[Item]

    instructions = (
        "You generate iOS localization keys + short UI translations.\n"
        "HARD RULES:\n"
        f"- NEVER translate or alter these tokens: {', '.join(GLOSSARY_TOKENS)}\n"
        "- Do NOT translate the product concept Hook/Hooks into any literal word (crochet/gancho/gancio/haak etc.).\n"
        "- Keep technical words Webhook/Deeplink/Bluetooth/CarPlay/Android Auto as-is.\n"
        "\n"
        "KEY RULES:\n"
        "- key must be snake_case (lowercase letters/digits/underscore only)\n"
        "- key must start with the provided namespace + '_' (e.g., hooks_add, vars_edit_title)\n"
        "- Avoid redundancy: if namespace is hooks_, do NOT repeat 'hook' in the key unless needed.\n"
        "- Prefer short, semantic keys:\n"
        "  - hooks_add, hooks_manage, hooks_actions, hooks_delete_confirm_title\n"
        "  - vars_edit_title, vars_add, account_logout, settings_debug\n"
        "\n"
        "TEXT STYLE:\n"
        "- Keep strings concise and UI-friendly.\n"
        "- Preserve punctuation like '?'.\n"
        "- Try not to exceed max_chars by much.\n"
        "- If original is French/other, translate to natural English for 'en'.\n"
        "\n"
        "Output must match schema exactly."
    )

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps({"items": requests}, ensure_ascii=False)},
        ],
        text_format=Out,
    )
    out: Out = resp.output_parsed

    result: Dict[str, Dict[str, str]] = {}
    for it in out.items:
        # Postprocess per language (fixed terms, hook casing, punctuation)
        result[it.id] = {
            "key": slug_key(it.key),
            "en": postprocess_text("en", it.en),
            "fr": postprocess_text("fr", it.fr),
            "de": postprocess_text("de", it.de),
            "es": postprocess_text("es", it.es),
            "it": postprocess_text("it", it.it),
            "nl": postprocess_text("nl", it.nl),
        }
    return result

def ensure_unique_keys(items: List[Tuple[str, Dict[str, str]]]) -> List[Tuple[str, Dict[str, str]]]:
    """
    Ensure unique keys: if a collision happens with different EN, suffix _2, _3...
    """
    key_to_en: Dict[str, str] = {}
    out: List[Tuple[str, Dict[str, str]]] = []

    def unique_key(k: str) -> str:
        if k not in key_to_en:
            return k
        n = 2
        while f"{k}_{n}" in key_to_en:
            n += 1
        return f"{k}_{n}"

    for lit, data in items:
        k = data["key"]
        en = data.get("en", "")
        if k in key_to_en and key_to_en[k] != en:
            data = dict(data)
            data["key"] = unique_key(k)
        key_to_en[data["key"]] = en
        out.append((lit, data))
    return out

def build_comment(lit: str, info: Dict[str, str]) -> str:
    parts = [f"original: {lit}"]
    for lang in ["fr", "de", "es", "it", "nl"]:
        s = info.get(lang, "")
        if s:
            parts.append(f"{lang}: {s}")
    c = " | ".join(parts)
    return c[:260].replace("*/", "* /")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Root directory to scan (recursively)")
    ap.add_argument("--use-openai", action="store_true", help="Use OpenAI for keys/translations")
    ap.add_argument("--limit", type=int, default=10, help="Dry-run preview count (unique literals)")
    ap.add_argument("--apply", action="store_true", help="Write .strings + rewrite .swift files")
    ap.add_argument("--backup", action="store_true", help="Write .bak for rewritten .swift files")
    ap.add_argument("--out-l10n", default="L10n", help="Output folder containing <lang>.lproj/")
    ap.add_argument("--table", default="Localizable", help="Strings table name")
    ap.add_argument("--exclude-dir", action="append", default=[], help="Directory names to exclude (repeatable)")
    ap.add_argument("--fill-suggestions", action="store_true",
                    help="Fill non-EN .strings with generated suggestions (instead of leaving blank).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    exclude_dirs = set(EXCLUDE_DIRS_DEFAULT) | set(args.exclude_dir)

    swift_files = iter_swift_files(root, exclude_dirs)
    all_occs: List[Occurrence] = []
    for f in tqdm(swift_files, desc="Scanning .swift"):
        all_occs.extend(extract_occurrences(f))

    by_lit: Dict[str, List[Occurrence]] = {}
    for oc in all_occs:
        by_lit.setdefault(oc.literal, []).append(oc)

    literals = list(by_lit.keys())
    preview = literals[: args.limit]

    out_base = Path(args.out_l10n).resolve()
    existing: Dict[str, Dict[str, str]] = {}
    for lang in LANGS:
        path = out_base / f"{lang}.lproj" / f"{args.table}.strings"
        existing[lang] = load_existing_strings(path)

    # -------------------------
    # Build mapping (preview)
    # -------------------------
    mapping_preview: Dict[str, Dict[str, str]] = {}

    for lit in preview:
        if lit in COMMON_MAP:
            data = dict(COMMON_MAP[lit])
            # Postprocess common translations too (punct/terms)
            for lang in ["en", "fr", "de", "es", "it", "nl"]:
                data[lang] = postprocess_text(lang, data[lang])
            mapping_preview[lit] = data

    remaining = [lit for lit in preview if lit not in mapping_preview]

    if args.use_openai and remaining:
        reqs = []
        for idx, lit in enumerate(remaining):
            occ0 = by_lit[lit][0]
            ns = guess_namespace(occ0)
            # headroom for longer languages
            max_chars = max(12, int(len(lit) * 1.30))
            reqs.append({"id": f"p{idx:04d}", "original": lit, "namespace": ns, "max_chars": max_chars})

        resp = ai_normalize(reqs)
        tmp = [(r["original"], resp[r["id"]]) for r in reqs]
        tmp = ensure_unique_keys(tmp)
        for lit, data in tmp:
            mapping_preview[lit] = data
    else:
        for lit in remaining:
            occ0 = by_lit[lit][0]
            ns = guess_namespace(occ0)
            mapping_preview[lit] = {
                "key": f"{ns}_{slug_key(lit)}",
                "en": postprocess_text("en", lit),
                "fr": "", "de": "", "es": "", "it": "", "nl": "",
            }

    # -------------------------
    # DRY-RUN PLAN
    # -------------------------
    print("\n================= DRY-RUN PLAN =================")
    print(f"Root: {root}")
    print(f"Found occurrences: {len(all_occs)}")
    print(f"Unique literals: {len(literals)}")
    print(f"Previewing first {len(preview)} terms\n")

    for idx, lit in enumerate(preview, start=1):
        info = mapping_preview[lit]
        key = info["key"]

        print(f"--- Term #{idx} ---")
        print(f"Original: {lit!r}")
        print(f"Proposed key: {key}")

        en_val = existing["en"].get(key, info.get("en", ""))
        print(f"EN: {'(kept existing) ' if key in existing['en'] else ''}{en_val!r}")

        for lang in ["fr", "de", "es", "it", "nl"]:
            if key in existing[lang]:
                print(f"{lang.upper()}: (kept existing) {existing[lang][key]!r}")
            else:
                sug = info.get(lang, "")
                print(f"{lang.upper()}: (would write {'suggestion' if args.fill_suggestions else 'empty'})  (suggestion: {sug!r})")

        print("Replacements:")
        occs = by_lit[lit]
        for oc in occs[:3]:
            before = oc.line_text.rstrip("\n")
            if oc.mode == "swiftui_key":
                # replace inside quotes
                after = rewrite_line(before, [ (oc.span, key) ])
            else:
                # replace including quotes with NSLocalizedString(...)
                start, end = oc.span
                after = rewrite_line(before, [ ((start - 1, end + 1), f'NSLocalizedString("{key}", comment: "")') ])
            print(f"  {oc.file.name}:{oc.line_no} ({oc.mode})")
            print(f"    - {before.strip()}")
            print(f"    + {after.strip()}")
        if len(occs) > 3:
            print(f"  … plus {len(occs)-3} other occurrence(s)")
        print("")

    if not args.apply:
        print("NOTE: This was a dry-run. Use --apply to write files and rewrite Swift sources.")
        return

    # -------------------------
    # APPLY: build mapping for all literals
    # -------------------------
    mapping_all: Dict[str, Dict[str, str]] = {}

    for lit in literals:
        if lit in COMMON_MAP:
            data = dict(COMMON_MAP[lit])
            for lang in ["en", "fr", "de", "es", "it", "nl"]:
                data[lang] = postprocess_text(lang, data[lang])
            mapping_all[lit] = data

    remaining_all = [lit for lit in literals if lit not in mapping_all]

    if args.use_openai and remaining_all:
        chunk_size = 40
        for i in tqdm(range(0, len(remaining_all), chunk_size), desc="OpenAI normalize (all)"):
            chunk = remaining_all[i:i+chunk_size]
            reqs = []
            for j, lit in enumerate(chunk):
                occ0 = by_lit[lit][0]
                ns = guess_namespace(occ0)
                max_chars = max(12, int(len(lit) * 1.30))
                reqs.append({"id": f"c{i+j:06d}", "original": lit, "namespace": ns, "max_chars": max_chars})

            resp = ai_normalize(reqs)
            tmp = [(r["original"], resp[r["id"]]) for r in reqs]
            tmp = ensure_unique_keys(tmp)
            for lit, data in tmp:
                mapping_all[lit] = data
    else:
        for lit in remaining_all:
            occ0 = by_lit[lit][0]
            ns = guess_namespace(occ0)
            mapping_all[lit] = {
                "key": f"{ns}_{slug_key(lit)}",
                "en": postprocess_text("en", lit),
                "fr": "", "de": "", "es": "", "it": "", "nl": "",
            }

    # -------------------------
    # Write .strings
    # - EN filled (existing wins)
    # - other langs:
    #   - existing wins
    #   - else blank by default
    #   - or fill with suggestions if --fill-suggestions
    # -------------------------
    per_lang_lines: Dict[str, List[str]] = {lang: [] for lang in LANGS}

    for lit in literals:
        info = mapping_all[lit]
        key = info["key"]

        cmt = build_comment(lit, info)

        en_val = existing["en"].get(key, info.get("en", ""))

        vals: Dict[str, str] = {"en": en_val}
        for lang in ["fr", "de", "es", "it", "nl"]:
            if key in existing[lang]:
                vals[lang] = existing[lang][key]
            else:
                vals[lang] = info.get(lang, "") if args.fill_suggestions else ""

        for lang in LANGS:
            per_lang_lines[lang].append(f"/* {cmt} */")
            per_lang_lines[lang].append(f"\"{escape_strings(key)}\" = \"{escape_strings(vals[lang])}\";")
            per_lang_lines[lang].append("")

    for lang in LANGS:
        folder = out_base / f"{lang}.lproj"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{args.table}.strings"
        path.write_text("\n".join(per_lang_lines[lang]), encoding="utf-8")
        print(f"Wrote {path}")

    # -------------------------
    # Rewrite Swift files
    # -------------------------
    file_edits: Dict[Path, Dict[int, List[Tuple[Tuple[int, int], str]]]] = {}

    for lit, occs in by_lit.items():
        key = mapping_all[lit]["key"]
        for oc in occs:
            if oc.mode == "swiftui_key":
                # replace inside quotes
                file_edits.setdefault(oc.file, {}).setdefault(oc.line_no, []).append((oc.span, key))
            else:
                # replace including quotes with NSLocalizedString(...)
                start, end = oc.span
                file_edits.setdefault(oc.file, {}).setdefault(oc.line_no, []).append(
                    ((start - 1, end + 1), f'NSLocalizedString("{key}", comment: "")')
                )

    for f, edits_by_line in tqdm(file_edits.items(), desc="Rewriting .swift"):
        if args.backup:
            bak = f.with_suffix(f.suffix + ".bak")
            if not bak.exists():
                bak.write_text(f.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines(True)
        for line_no, edits in edits_by_line.items():
            idx = line_no - 1
            original = lines[idx].rstrip("\n")
            lines[idx] = rewrite_line(original, edits) + ("\n" if lines[idx].endswith("\n") else "")
        f.write_text("".join(lines), encoding="utf-8")
        print(f"Rewrote {f}")

if __name__ == "__main__":
    main()
