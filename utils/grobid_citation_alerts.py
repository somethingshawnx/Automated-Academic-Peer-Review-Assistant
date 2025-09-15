import os
import sys
import json
import argparse
import requests
import xml.etree.ElementTree as ET
from collections import Counter

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

def call_grobid(pdf_path: str, out_xml: str) -> None:
    """Send PDF to GROBID and save XML response."""
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            GROBID_URL,
            files={"input": f},
            data={"consolidateHeader": "1", "consolidateCitations": "1"},
            timeout=60,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"GROBID error {resp.status_code}: {resp.text[:200]}")
    with open(out_xml, "w", encoding="utf-8") as outf:
        outf.write(resp.text)


def parse_references_from_xml(xml_path: str):
    """Parse GROBID TEI XML and extract structured references."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    refs = []

    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", ns):
        title_el = bibl.find(".//tei:title", ns)
        year_el = bibl.find(".//tei:date", ns)
        doi_el = bibl.find(".//tei:idno[@type='DOI']", ns)
        authors = [
            pers.find("tei:surname", ns).text if pers.find("tei:surname", ns) is not None else ""
            for pers in bibl.findall(".//tei:author/tei:persName", ns)
        ]

        refs.append({
            "title": title_el.text if title_el is not None else None,
            "year": year_el.attrib.get("when") if year_el is not None else None,
            "authors": authors,
            "doi": doi_el.text if doi_el is not None else None
        })

    return refs


def analyze_citations(refs, year_threshold: int = 2015):
    """Compute citation quality metrics."""
    total = len(refs)
    if total == 0:
        return {
            "total_references": 0,
            "recent_percentage": 0,
            "outdated_percentage": 0,
            "missing_dois": 0,
            "diversity_score": 0,
            "citation_quality_score": 0,
            "outdated_references": [],
        }

    years = [int(r["year"]) for r in refs if r.get("year") and r["year"].isdigit()]
    outdated = [r for r in refs if r.get("year") and r["year"].isdigit() and int(r["year"]) <= year_threshold]
    recent = [r for r in refs if r.get("year") and r["year"].isdigit() and int(r["year"]) > year_threshold]

    missing_dois = [r for r in refs if not r.get("doi")]
    venues = [r.get("title", "").split(":")[0] for r in refs if r.get("title")]
    diversity_score = len(set(venues)) / total if total else 0

    # Simple quality scoring (out of 10)
    citation_quality_score = (
        (len(recent) / total) * 4  # recency
        + (diversity_score * 3)    # diversity
        + ((1 - len(missing_dois) / total) * 3)  # completeness of DOIs
    )

    return {
        "total_references": total,
        "recent_percentage": round(len(recent) / total * 100, 2),
        "outdated_percentage": round(len(outdated) / total * 100, 2),
        "missing_dois": len(missing_dois),
        "diversity_score": round(diversity_score, 2),
        "citation_quality_score": round(citation_quality_score, 2),
        "outdated_references": outdated,
    }


def main():
    parser = argparse.ArgumentParser(description="Run GROBID + citation analysis")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument("--year_threshold", type=int, default=2015, help="References before this year are outdated")
    parser.add_argument("--output", type=str, required=True, help="Path to save JSON report")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.pdf_path))[0]
    refs_xml = f"data/references/{base}_refs.xml"
    os.makedirs("data/references", exist_ok=True)

    print(f"Sending {args.pdf_path} to GROBID at {GROBID_URL} ...")
    call_grobid(args.pdf_path, refs_xml)
    print(f"References saved to {refs_xml}")

    refs = parse_references_from_xml(refs_xml)
    analysis = analyze_citations(refs, args.year_threshold)

    report = {
        "pdf": args.pdf_path,
        "analysis": analysis,
        "references": refs
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
