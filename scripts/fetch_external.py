"""Download the OPEN external games used to train/validate the model, pinned to a source commit and verified by sha256.

    python scripts/fetch_external.py            # downloads missing files into ext_open/ (gitignored) and verifies all
    python scripts/fetch_external.py --check    # only verifies what is already there

Sources (other games, not the competition match):
  - Metrica Sports sample data, https://github.com/metrica-sports/sample-data  (no formal licence; asks for responsible
    use and credit), pinned to commit METRICA_SHA.
  - SkillCorner open data, https://github.com/SkillCorner/opendata (MIT), pinned to commit SKILLCORNER_SHA; the tracking
    files are Git LFS objects served by media.githubusercontent.com.
Standard library only. It never touches data/ or MTAH_DATA_DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "ext_open"

METRICA_SHA = "e706dd506b360d69d9d123d5b8026e7294b13996"       # master on 2026-11-12 (last commit 2021-04-15)
SKILLCORNER_SHA = "4340d274572876239c154c90bc507a9b3250a656"   # master on 2026-11-12 (last commit 2026-09-14)

# (folder under ext_open, file name) -> (path inside the source repo, sha256, size in bytes)
FILES = {
    ("metrica_game1", "Sample_Game_1_RawEventsData.csv"): ("data/Sample_Game_1", "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05", 139446),
    ("metrica_game1", "Sample_Game_1_RawTrackingData_Away_Team.csv"): ("data/Sample_Game_1", "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618", 32782073),
    ("metrica_game1", "Sample_Game_1_RawTrackingData_Home_Team.csv"): ("data/Sample_Game_1", "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364", 32782961),
    ("metrica_game2", "Sample_Game_2_RawEventsData.csv"): ("data/Sample_Game_2", "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba", 153897),
    ("metrica_game2", "Sample_Game_2_RawTrackingData_Away_Team.csv"): ("data/Sample_Game_2", "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b", 29627711),
    ("metrica_game2", "Sample_Game_2_RawTrackingData_Home_Team.csv"): ("data/Sample_Game_2", "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694", 31890338),
    ("metrica_game3", "Sample_Game_3_events.json"): ("data/Sample_Game_3", "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0", 1709486),
    ("metrica_game3", "Sample_Game_3_metadata.xml"): ("data/Sample_Game_3", "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1", 81806),
    ("metrica_game3", "Sample_Game_3_tracking.txt"): ("data/Sample_Game_3", "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658", 53443115),
}
SKILLCORNER = {
    "1874553": {"dynamic_events.csv": ("92507490abbbeec7697227581c864cfb0a96c29b214dd63dcaa48929bb6c6408", 4738858),
                "match.json": ("e6974e402779d8be44e75bf88ef2a3acfd4d025b87b3fc57f518fa826a645ad8", 33730),
                "phases_of_play.csv": ("e70054b64202a3f318b7208edec9a329da1f4cfcafeeb72ae197176c4a043a8c", 115978),
                "tracking_extrapolated.jsonl": ("ea97f58f8eaad925feaeacc6395ec24860dd80027af8a89450276adebd29d265", 90729279)},
    "1886347": {"dynamic_events.csv": ("b21368b9b94eb8f83032fe14a2a8da88ab7eda048b4f8078c06d0e6955d12bfb", 4918599),
                "match.json": ("30c608e78663e9f490b3545479a3ce67b171f418c5133d5609cb249a2466abb2", 27541),
                "phases_of_play.csv": ("d1c3066f7ed0598069d0d1b317bb30261eca3d7786fd9d540985f395e4c8aea9", 121864),
                "tracking_extrapolated.jsonl": ("3577e2803da95390f8b2f85d47829eb55fbbfee6211c94adc65db0dc78e46b41", 89280839)},
    "1899585": {"dynamic_events.csv": ("14c62ffc99aa5fd62998c238036873509a9d82ced44354697a1db4251424d557", 4375032),
                "match.json": ("41de8cb74a0512fcf0cbaaa5c615ae3e7dcaaf6745b9c78a050b59b906b1cb06", 27810),
                "phases_of_play.csv": ("e40562010c1b7c4aa6e0dbc17425dd236a1d82ce538ec03e77367a74e3f96b7e", 119735),
                "tracking_extrapolated.jsonl": ("90b6289ec2181192d1793ec6240fbbe8225ff0219c7956710c6278c31a661100", 85550410)},
    "1925299": {"dynamic_events.csv": ("2557acbdcba6547d764b9f7c277c3d7e81a4d58692d6c20c2bfa28625ae0cce3", 5043601),
                "match.json": ("f4345b32c0fdd7c9990df70050b3a74f57b1992cf762e20f2a4d309bd2fba7d2", 28396),
                "phases_of_play.csv": ("a1682a736a9cf1c311cb799f0a949f7e023dca4ebaa9dd294b138e8929271274", 131104),
                "tracking_extrapolated.jsonl": ("48528be40ad82fa3e2b232e1af8d311705f7d6d119925ef199d2b59ea36fd3c9", 97091519)},
}


# UNTOUCHED HOLD-OUT matches of EXP-014 (same source commit). Never used for training, tuning or selection.
HOLDOUT = {
    "1927964": {
        "dynamic_events.csv": ("638747f76b43690974f95958952dfe62c9e50d64648f9456f43f66d5d91201f3", 5509112),
        "match.json": ("f967f911c20f5c5f8c89bbe2020a30c92731173616d5a1e4bafeb65435278b76", 33400),
        "phases_of_play.csv": ("9216447a63a912edcad1a4d36370372210dcc800a7f48fc10127fd7de9eae503", 117263),
        "tracking_extrapolated.jsonl": ("858c7a756ca3d33017f864775af5999d2c0f364e58ab60ae040c8b9ebab17827", 91049828),
    },
    "1953632": {
        "dynamic_events.csv": ("d1f676ebb182da585da128fc80d468e27fb676e254abc47531daf920d977e51f", 4643533),
        "match.json": ("366c8dd6b504d4f4ace5685040648b0f30081043a44f01663b8b7e077e6bfde8", 27767),
        "phases_of_play.csv": ("9243e4be8efcc1922fe46f9cda56c781584329f1febe84a74ee808c8f4ff0b93", 114478),
        "tracking_extrapolated.jsonl": ("1c2014f97c3eff1ce2475203512c9d126930d7599c036105c9ac442d1de1667e", 90027965),
    },
    "1959846": {
        "dynamic_events.csv": ("aa3d08a927a84f2b42d976cc10401e9efc52468744e0a7a86cb0d3dc2e288058", 4499680),
        "match.json": ("4bbe2058af53edd42d5e3acd7fb28ecb8c79e4afb16cb9b77a6fe9b58f6e516b", 32946),
        "phases_of_play.csv": ("63b33ce9c020cfd54be6e957cd01a42cebec2acacd1345c338196fff5ecd9a9f", 128571),
        "tracking_extrapolated.jsonl": ("5ba2d87f6fd1ba3138301b369bd9266be69416a2d799a3fba5b5ec8a52e1284b", 87710167),
    },
    "1986691": {
        "dynamic_events.csv": ("c2f7102092970368cc487353d6e93e06522f8d3f184326cced63fa167ca0d9f5", 4290910),
        "match.json": ("754f6a6fd0d996236cf553ef5c3c24513274b0db024cb6833a7419b08d55dd68", 33792),
        "phases_of_play.csv": ("7589168484ae1305072afa397126a72f8925cf261b9eb121ecb0192083632cd3", 115267),
        "tracking_extrapolated.jsonl": ("630e946950fe459f5e48872b01a36e6dfc319cf9027e7b98156150bb6e2c30f6", 88531597),
    },
}
SKILLCORNER.update(HOLDOUT)


def _entries():
    for (folder, name), (srcdir, sha, size) in FILES.items():
        url = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/{}/{}".format(METRICA_SHA, srcdir, name)
        yield DEST / folder / name, url, sha, size
    for match, files in SKILLCORNER.items():
        for suffix, (sha, size) in files.items():
            name = "{}_{}".format(match, suffix)
            base = "media.githubusercontent.com/media" if suffix.endswith(".jsonl") else "raw.githubusercontent.com"
            url = "https://{}/SkillCorner/opendata/{}/data/matches/{}/{}".format(base, SKILLCORNER_SHA, match, name)
            yield DEST / "skillcorner_{}".format(match) / name, url, sha, size


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, never download")
    args = ap.parse_args(argv)
    bad = 0
    for path, url, sha, size in _entries():
        status = "ok"
        if not path.exists() or path.stat().st_size != size or _sha256(path) != sha:
            if args.check:
                status = "MISSING/MISMATCH"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, path)
                status = "downloaded" if (path.stat().st_size == size and _sha256(path) == sha) else "MISMATCH"
        bad += status in ("MISSING/MISMATCH", "MISMATCH")
        print("{:18s} {}".format(status, path.relative_to(ROOT)))
    print("{} file(s) failed verification".format(bad) if bad else "all external files verified (sha256)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
