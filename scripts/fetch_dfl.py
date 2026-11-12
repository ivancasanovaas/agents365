"""Download the Sportec Open DFL Dataset (7 Bundesliga matches), an OPEN dataset of OTHER games, and verify it by MD5.

    python scripts/fetch_dfl.py            # downloads missing files into ext_open/dfl/ (gitignored) and verifies them
    python scripts/fetch_dfl.py --check    # only verifies

Source: https://doi.org/10.6084/m9.figshare.28196177 (Bassek et al., "An integrated dataset of spatiotemporal and event data
in elite soccer", Scientific Data 2025), licence CC BY 4.0, published 2025-02-02. Size and MD5 are those listed by figshare.
Standard library only. Never touches the competition data.
"""
import argparse, hashlib, sys, urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "ext_open" / "dfl"
# name -> (download url, size in bytes, md5)
FILES = {
    'DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml': ('https://ndownloader.figshare.com/files/51643475', 11930, 'bc5d5b21dbcf36aa30f4cd0428309f6b'),
    'DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml': ('https://ndownloader.figshare.com/files/51643472', 11722, '37b0f85588c0acfb995fe633c7ac2e0d'),
    'DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml': ('https://ndownloader.figshare.com/files/51643478', 11745, 'b9710781d400dd812f62b0a6e3d7bce8'),
    'DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml': ('https://ndownloader.figshare.com/files/51643481', 11952, '7b021e577fca917a63111d55cc29ae5b'),
    'DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml': ('https://ndownloader.figshare.com/files/51643487', 11922, '65a4568e4f374af94d3c1015cbc7c4f5'),
    'DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml': ('https://ndownloader.figshare.com/files/51643484', 11932, '5979209af316a914ce57347d0633df6c'),
    'DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml': ('https://ndownloader.figshare.com/files/51643490', 11823, '5e2d580d3d8ed97e32959de90d04992b'),
    'DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml': ('https://ndownloader.figshare.com/files/51643493', 776578, '2d88e180a1f63c7e4157feb980fc7255'),
    'DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml': ('https://ndownloader.figshare.com/files/51643496', 612114, '83e8ed7ea8e8ebbbdc61f61ac3534008'),
    'DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml': ('https://ndownloader.figshare.com/files/51643499', 635747, 'ab66e51512f39d06f49f27dd3b4c2f40'),
    'DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml': ('https://ndownloader.figshare.com/files/51643502', 683018, '64b1be01d6ff6848168efebd56f344b9'),
    'DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml': ('https://ndownloader.figshare.com/files/51643505', 698898, 'f68482441d38723d69f244f896482803'),
    'DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml': ('https://ndownloader.figshare.com/files/51643508', 732186, 'b7e9d82af4034539d802319be6c57776'),
    'DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml': ('https://ndownloader.figshare.com/files/51643511', 681050, 'c13bc5efa5eca4a022cc678210b87d5f'),
    'DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml': ('https://ndownloader.figshare.com/files/51643514', 418524833, 'e4273a2adf5dbd9d4ec42def2f01905c'),
    'DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml': ('https://ndownloader.figshare.com/files/51643517', 391722803, 'bacb2f609c1d1b0bfc906f6b9d4109f2'),
    'DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml': ('https://ndownloader.figshare.com/files/51643520', 348945715, '3efff25a86a61768e7d253b4bba137a3'),
    'DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml': ('https://ndownloader.figshare.com/files/51643523', 362567859, '4e1063e0a0ae499920cf68ac1aa3ab4e'),
    'DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml': ('https://ndownloader.figshare.com/files/51643526', 371688499, 'e56ff1336a334f8da51110fbe7cab764'),
    'DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml': ('https://ndownloader.figshare.com/files/51643529', 356219423, '9105bad987191d3c3c362c1fce9a4fcd'),
    'DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml': ('https://ndownloader.figshare.com/files/51643532', 372934140, 'e060bb7d7a5a3bfbd15bbf9bc631024d'),
}

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    DEST.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, (url, size, md5) in FILES.items():
        path = DEST / name
        status = "ok"
        if not path.exists() or path.stat().st_size != size or _md5(path) != md5:
            if args.check:
                status = "MISSING/MISMATCH"
            else:
                urllib.request.urlretrieve(url, path)
                status = "downloaded" if (path.stat().st_size == size and _md5(path) == md5) else "MISMATCH"
        bad += status in ("MISSING/MISMATCH", "MISMATCH")
        print("{:18s} {} ({:.0f} MB)".format(status, name, size / 1e6), flush=True)
    print("{} file(s) failed verification".format(bad) if bad else "all DFL files verified (md5)")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
