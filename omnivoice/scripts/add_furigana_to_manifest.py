#!/usr/bin/env python3
"""Add fugashi-generated Japanese furigana labels to WebDataset JSONL shards.

The script keeps existing audio tar shards untouched. It reads an OmniVoice
``data.lst`` manifest, rewrites each companion JSONL into a new ``txts``
directory, and writes a new manifest that points at the rewritten JSONL files.

By default the original ``text`` field is preserved and ``text_fugashi`` is
added. Use ``--replace_text`` when a downstream reader should consume the
furigana text through the existing ``text`` field.
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

KANJI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff々〆ヶ]")
WHITESPACE_RE = re.compile(r"(\s+)")


@dataclass(frozen=True)
class ManifestEntry:
    tar_path: str
    jsonl_path: str
    num_items: int
    num_seconds: float
    num_seconds_text: str


@dataclass
class ShardStats:
    total: int = 0
    processed: int = 0
    changed: int = 0
    skipped_language: int = 0
    missing_text: int = 0

    def add(self, other: "ShardStats") -> None:
        self.total += other.total
        self.processed += other.processed
        self.changed += other.changed
        self.skipped_language += other.skipped_language
        self.missing_text += other.missing_text


def _load_fugashi_tagger(dicdir: str | None, mecab_args: str | None):
    try:
        import fugashi
    except ImportError as exc:
        raise RuntimeError(
            "fugashi is required. Install fugashi with a dictionary, for example: "
            "`uv pip install fugashi unidic-lite`."
        ) from exc

    args = mecab_args or ""
    if dicdir:
        args = f"{args} -d {dicdir}".strip()
    return fugashi.Tagger(args)


def _read_manifest(path: Path) -> list[ManifestEntry]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid manifest line {line_no} in {path}: {line!r}. "
                    "Expected: <tar_path> <jsonl_path> <num_items> <num_seconds>."
                )
            tar_path, jsonl_path, num_items, num_seconds = parts
            entries.append(
                ManifestEntry(
                    tar_path=tar_path,
                    jsonl_path=jsonl_path,
                    num_items=int(num_items),
                    num_seconds=float(num_seconds),
                    num_seconds_text=num_seconds,
                )
            )
    return entries


def _katakana_to_hiragana(text: str) -> str:
    chars = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


def _feature_value(feature: Any, names: Iterable[str]) -> str | None:
    for name in names:
        value = getattr(feature, name, None)
        if value and value != "*":
            return str(value)

    if hasattr(feature, "_asdict"):
        data = feature._asdict()
        for name in names:
            value = data.get(name)
            if value and value != "*":
                return str(value)

    if isinstance(feature, (list, tuple)):
        # IPADIC-style features usually store reading and pronunciation at
        # indices 7 and 8. UniDic users normally hit the named fields above.
        for index in (7, 8):
            if len(feature) > index and feature[index] and feature[index] != "*":
                return str(feature[index])

    return None


def _node_reading(node: Any) -> str | None:
    reading = _feature_value(
        node.feature,
        (
            "kana",
            "kanaBase",
            "pron",
            "pronBase",
            "reading",
            "yomi",
        ),
    )
    if not reading:
        return None
    reading = _katakana_to_hiragana(reading.strip())
    return reading or None


def _furiganize_chunk(tagger: Any, text: str) -> str:
    pieces = []
    for node in tagger(text):
        surface = node.surface
        if not surface:
            continue
        reading = _node_reading(node)
        if KANJI_RE.search(surface) and reading:
            pieces.append(f"{surface}({reading})")
        else:
            pieces.append(surface)
    return "".join(pieces)


def furiganize_text(tagger: Any, text: str) -> str:
    """Annotate each MeCab morpheme containing Kanji as ``surface(reading)``."""
    if not text:
        return text

    parts = WHITESPACE_RE.split(text)
    return "".join(
        part
        if not part or WHITESPACE_RE.fullmatch(part)
        else _furiganize_chunk(tagger, part)
        for part in parts
    )


def _should_process_language(
    item: dict[str, Any],
    language_field: str,
    language_ids: set[str],
    require_language_match: bool,
) -> bool:
    if not language_ids:
        return True

    language = item.get(language_field)
    if language is None:
        return not require_language_match
    return str(language) in language_ids


def _output_jsonl_path(
    input_jsonl: Path,
    output_txt_dir: Path,
    used_names: set[str],
    shard_index: int,
) -> Path:
    name = input_jsonl.name
    if name in used_names:
        name = f"{shard_index:06d}-{name}"
    used_names.add(name)
    return output_txt_dir / name


def _rewrite_jsonl(
    input_path: Path,
    output_path: Path,
    tagger: Any,
    text_field: str,
    furigana_field: str,
    replace_text: bool,
    original_text_field: str,
    language_field: str,
    language_ids: set[str],
    require_language_match: bool,
) -> ShardStats:
    stats = ShardStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            stats.total += 1

            text = item.get(text_field)
            if text is None:
                stats.missing_text += 1
                print(json.dumps(item, ensure_ascii=False), file=fout)
                continue

            if not _should_process_language(
                item=item,
                language_field=language_field,
                language_ids=language_ids,
                require_language_match=require_language_match,
            ):
                stats.skipped_language += 1
                print(json.dumps(item, ensure_ascii=False), file=fout)
                continue

            furigana_text = furiganize_text(tagger, str(text))
            item[furigana_field] = furigana_text
            item["furigana_source"] = "fugashi"
            item["furigana_format"] = "morpheme_surface(reading_hiragana)"
            if replace_text:
                item.setdefault(original_text_field, text)
                item[text_field] = furigana_text

            stats.processed += 1
            if furigana_text != text:
                stats.changed += 1

            print(json.dumps(item, ensure_ascii=False), file=fout)

    logger.info(
        "Wrote %s from %s: total=%d processed=%d changed=%d "
        "skipped_language=%d missing_text=%d",
        output_path,
        input_path,
        stats.total,
        stats.processed,
        stats.changed,
        stats.skipped_language,
        stats.missing_text,
    )
    return stats


def add_furigana_to_manifest(args: argparse.Namespace) -> None:
    input_manifest = Path(args.input_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_txt_dir = output_dir / args.output_txt_subdir
    output_manifest = (
        Path(args.output_manifest).expanduser().resolve()
        if args.output_manifest
        else output_dir / input_manifest.name
    )

    if not input_manifest.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {input_manifest}")
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output manifest already exists: {output_manifest}. "
            "Pass --overwrite to replace it."
        )

    entries = _read_manifest(input_manifest)
    if not entries:
        raise ValueError(f"No entries found in manifest: {input_manifest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    tagger = _load_fugashi_tagger(args.dicdir, args.mecab_args)
    language_ids = {item for item in args.language_ids if item}

    total_stats = ShardStats()
    rewritten_entries = []
    used_names: set[str] = set()

    for shard_index, entry in enumerate(entries):
        input_jsonl = Path(entry.jsonl_path).expanduser()
        if not input_jsonl.is_absolute():
            input_jsonl = (input_manifest.parent / input_jsonl).resolve()
        if not input_jsonl.is_file():
            raise FileNotFoundError(f"Label JSONL does not exist: {input_jsonl}")

        output_jsonl = _output_jsonl_path(input_jsonl, output_txt_dir, used_names, shard_index)
        if output_jsonl.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output JSONL already exists: {output_jsonl}. "
                "Pass --overwrite to replace it."
            )

        shard_stats = _rewrite_jsonl(
            input_path=input_jsonl,
            output_path=output_jsonl,
            tagger=tagger,
            text_field=args.text_field,
            furigana_field=args.furigana_field,
            replace_text=args.replace_text,
            original_text_field=args.original_text_field,
            language_field=args.language_field,
            language_ids=language_ids,
            require_language_match=args.require_language_match,
        )
        total_stats.add(shard_stats)
        rewritten_entries.append((entry, output_jsonl))

    with output_manifest.open("w", encoding="utf-8") as fout:
        for entry, output_jsonl in rewritten_entries:
            print(
                entry.tar_path,
                str(output_jsonl),
                entry.num_items,
                entry.num_seconds_text,
                file=fout,
            )

    logger.info(
        "Wrote manifest %s: shards=%d total=%d processed=%d changed=%d "
        "skipped_language=%d missing_text=%d",
        output_manifest,
        len(rewritten_entries),
        total_stats.total,
        total_stats.processed,
        total_stats.changed,
        total_stats.skipped_language,
        total_stats.missing_text,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the JSONL sidecars referenced by an OmniVoice data.lst "
            "manifest with fugashi-generated Japanese furigana labels."
        )
    )
    parser.add_argument("--input_manifest", required=True, help="Input data.lst path.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory. Writes <output_dir>/txts/*.jsonl and data.lst.",
    )
    parser.add_argument(
        "--output_manifest",
        default=None,
        help="Optional output data.lst path. Defaults to <output_dir>/<input name>.",
    )
    parser.add_argument(
        "--output_txt_subdir",
        default="txts",
        help="Subdirectory under output_dir for rewritten JSONL shards.",
    )
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--furigana_field", default="text_fugashi")
    parser.add_argument(
        "--replace_text",
        action="store_true",
        help="Replace text_field with the furigana text for legacy readers.",
    )
    parser.add_argument(
        "--original_text_field",
        default="text_original",
        help="Field used to preserve original text when --replace_text is set.",
    )
    parser.add_argument(
        "--language_field",
        default="language_id",
        help="JSONL field used for optional language filtering.",
    )
    parser.add_argument(
        "--language_ids",
        nargs="*",
        default=["ja", "jpn", "jp", "ja-JP", "Japanese", "japanese"],
        help=(
            "Language ids to process when language_field exists. Empty list "
            "means process all rows."
        ),
    )
    parser.add_argument(
        "--process_all_languages",
        action="store_true",
        help="Ignore language filtering and process every row.",
    )
    parser.add_argument(
        "--require_language_match",
        action="store_true",
        help="Skip rows with no language_field instead of processing them.",
    )
    parser.add_argument(
        "--dicdir",
        default=None,
        help="Optional MeCab dictionary directory passed to fugashi.",
    )
    parser.add_argument(
        "--mecab_args",
        default=None,
        help="Optional raw MeCab arguments passed to fugashi.Tagger.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.process_all_languages:
        args.language_ids = []

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    add_furigana_to_manifest(args)


if __name__ == "__main__":
    main()
