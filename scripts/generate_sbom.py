#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Write an SPDX 2.3 JSON SBOM for an hva release.

The only third-party code in the repository is the two pruned header sets
under vendor/. They have no package manifest, so generic scanners (syft,
trivy) do not see them; this script describes them explicitly. Their
provenance is checked against vendor/*/README.md so a re-vendor that forgets
to update the table below fails instead of producing a stale SBOM.

Usage: generate_sbom.py --version 0.1.0 --output hva-0.1.0.spdx.json
"""

import argparse
import datetime
import json
import pathlib
import re
import sys
import uuid

REPO_URL = "https://github.com/ROCm/hip-verbs-acceptance"
ROOT = pathlib.Path(__file__).resolve().parent.parent

ROCM_HEADERS_COMMIT = "37723244ca8cd75d5e55d30f44c77af1bd49218b"
RDMA_CORE_VERSION = "50.0"

VENDORED = [
    {
        "SPDXID": "SPDXRef-Package-rocm-headers",
        "name": "rocm-headers",
        "versionInfo": ROCM_HEADERS_COMMIT,
        "downloadLocation":
            "git+https://github.com/ROCm/rocm-systems@" + ROCM_HEADERS_COMMIT,
        "licenseDeclared": "MIT AND NCSA",
        "copyrightText": "Copyright (c) Advanced Micro Devices, Inc.",
        "supplier": "Organization: Advanced Micro Devices, Inc.",
        "comment": "Pruned HIP/HSA headers vendored under vendor/rocm-headers/.",
        "externalRefs": [{
            "referenceCategory": "PACKAGE-MANAGER",
            "referenceType": "purl",
            "referenceLocator":
                "pkg:github/ROCm/rocm-systems@" + ROCM_HEADERS_COMMIT,
        }],
        "readme": "vendor/rocm-headers/README.md",
        "provenance": ROCM_HEADERS_COMMIT,
    },
    {
        "SPDXID": "SPDXRef-Package-rdma-core-headers",
        "name": "rdma-core-headers",
        "versionInfo": RDMA_CORE_VERSION,
        "downloadLocation":
            "git+https://github.com/linux-rdma/rdma-core@v" + RDMA_CORE_VERSION,
        "licenseDeclared":
            "(GPL-2.0-only WITH Linux-syscall-note) OR Linux-OpenIB",
        "licenseConcluded": "Linux-OpenIB",
        "copyrightText": "NOASSERTION",
        "supplier": "Organization: linux-rdma",
        "comment": "libibverbs headers vendored under vendor/rdma-core-headers/; "
                   "used under the Linux-OpenIB option.",
        "externalRefs": [{
            "referenceCategory": "PACKAGE-MANAGER",
            "referenceType": "purl",
            "referenceLocator":
                "pkg:github/linux-rdma/rdma-core@v" + RDMA_CORE_VERSION,
        }],
        "readme": "vendor/rdma-core-headers/README.md",
        "provenance": "rdma-core v" + RDMA_CORE_VERSION,
    },
]


def check_provenance():
    errors = []
    for pkg in VENDORED:
        text = (ROOT / pkg["readme"]).read_text(encoding="utf-8")
        if pkg["provenance"] not in text:
            errors.append("%s does not mention %r; update VENDORED in %s"
                          % (pkg["readme"], pkg["provenance"],
                             pathlib.Path(__file__).name))
    return errors


def harness_version():
    text = (ROOT / "common" / "Version.hpp").read_text(encoding="utf-8")
    m = re.search(r'kHarnessVersion\s*=\s*"([^"]+)"', text)
    if not m:
        sys.exit("error: kHarnessVersion not found in common/Version.hpp")
    return m.group(1)


def build_document(version):
    main_pkg = {
        "SPDXID": "SPDXRef-Package-hva",
        "name": "hip-verbs-acceptance",
        "versionInfo": version,
        "downloadLocation": "git+%s@v%s" % (REPO_URL, version),
        "homepage": REPO_URL,
        "licenseDeclared": "MIT",
        "licenseConcluded": "MIT",
        "copyrightText": "Copyright (c) 2026 Advanced Micro Devices, Inc.",
        "supplier": "Organization: Advanced Micro Devices, Inc.",
        "filesAnalyzed": False,
        "externalRefs": [{
            "referenceCategory": "PACKAGE-MANAGER",
            "referenceType": "purl",
            "referenceLocator":
                "pkg:github/ROCm/hip-verbs-acceptance@v" + version,
        }],
    }
    packages = [main_pkg]
    relationships = [{
        "spdxElementId": "SPDXRef-DOCUMENT",
        "relationshipType": "DESCRIBES",
        "relatedSpdxElement": main_pkg["SPDXID"],
    }]
    for pkg in VENDORED:
        entry = {k: v for k, v in pkg.items()
                 if k not in ("readme", "provenance")}
        entry.setdefault("licenseConcluded", entry["licenseDeclared"])
        entry["filesAnalyzed"] = False
        packages.append(entry)
        relationships.append({
            "spdxElementId": main_pkg["SPDXID"],
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": entry["SPDXID"],
        })

    created = datetime.datetime.now(datetime.timezone.utc)
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "hip-verbs-acceptance-%s" % version,
        "documentNamespace": "%s/releases/v%s/sbom-%s"
                             % (REPO_URL, version, uuid.uuid4()),
        "creationInfo": {
            "created": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "creators": ["Organization: Advanced Micro Devices, Inc.",
                         "Tool: hip-verbs-acceptance-generate_sbom.py"],
        },
        "packages": packages,
        "relationships": relationships,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True,
                    help="release version without the leading 'v'")
    ap.add_argument("--output", required=True, type=pathlib.Path)
    args = ap.parse_args()

    errors = check_provenance()
    declared = harness_version()
    if declared != args.version:
        errors.append("common/Version.hpp says %r but the release is %r"
                      % (declared, args.version))
    if errors:
        for e in errors:
            print("error: " + e, file=sys.stderr)
        return 1

    doc = build_document(args.version)
    args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
