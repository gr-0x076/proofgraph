import argparse
import sys
import os
from proofgraph.pipeline import ProofGraphPipeline

def main():
    parser = argparse.ArgumentParser(
        description="ProofGraph: AI Citation & Source Attribution Auditor"
    )
    parser.add_argument(
        "--audit-citation",
        action="store_true",
        help="Audit an AI assertion or citation against a specific source URL"
    )
    parser.add_argument(
        "--claim", "-c",
        type=str,
        help="Target claim or assertion to audit (used with --audit-citation)"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        help="Source URL to audit against (required with --audit-citation)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Input document or claim text to verify"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to text file containing claims or document"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory to save report.md and graph.html (default: outputs/)"
    )
    parser.add_argument(
        "--no-live-search",
        action="store_true",
        help="Disable live web/Wikipedia fallback search"
    )

    args = parser.parse_args()

    # Mode 1: AI Citation & Source Attribution Audit
    if args.audit_citation:
        if not args.url:
            print("Error: --audit-citation requires a source URL via --url <https://...>", file=sys.stderr)
            sys.exit(1)
        target_text = args.claim or args.text
        if not target_text and args.file:
            if not os.path.exists(args.file):
                print(f"Error: File '{args.file}' not found.", file=sys.stderr)
                sys.exit(1)
            with open(args.file, "r") as fp:
                target_text = fp.read()
        if not target_text:
            print("Error: --audit-citation requires a target claim via --claim or --text.", file=sys.stderr)
            sys.exit(1)

        print("🛡️  Initializing ProofGraph AI Citation Auditor...")
        pipeline = ProofGraphPipeline()

        print(f"\n🔍 Auditing Cited Source: {args.url}")
        print(f"   Target Assertion: \"{target_text.strip()}\"")
        result = pipeline.verify_citation(
            claim_or_text=target_text,
            cited_url=args.url,
            output_dir=args.output
        )

        print("\n" + "=" * 70)
        print(" 🛡️  ProofGraph: AI Citation & Attribution Audit Result")
        print("=" * 70)
        print(f"Audited URL: {args.url}")
        print(f"Target Text: \"{target_text.strip()}\"")
        print("-" * 70)

        for idx, c in enumerate(result.claims, 1):
            v = result.verdicts.get(c.id)
            label = v.label.value if v else "UNVERIFIED"
            tier = v.evidence_tier.value if v else "UNKNOWN"

            if label == "SUPPORTED":
                status_tag = "🟢 SUPPORTED"
                audit_status = "PASSAGE VERIFIED (Cited source confirms assertion)"
            elif label == "CONTRADICTED":
                status_tag = "🔴 CONTRADICTED"
                audit_status = "CITATION REFUTATION (Cited source contradicts assertion)"
            elif label == "ENTITY_UNDERSPECIFIED":
                status_tag = "⚠️  ENTITY UNDERSPECIFIED"
                audit_status = "AMBIGUOUS SUBJECT (Cannot verify without entity disambiguation)"
            else:
                status_tag = "⚪ INSUFFICIENT_EVIDENCE"
                audit_status = "CITATION HALLUCINATION / UNMENTIONED (Claim not established in source)"

            print(f"\n[{idx}] Claim: \"{c.text}\"")
            print(f"    Verdict: {status_tag} [Evidence Tier: {tier}]")
            print(f"    Attribution: {audit_status}")
            if v:
                print(f"    Justification: {v.justification}")
            if c.is_underspecified and c.ambiguity_reason:
                print(f"    ⚠️  Ambiguity Notice: {c.ambiguity_reason}")

            passages = result.evidence_map.get(c.id, [])
            if passages:
                print(f"    Evidence Excerpts from Cited URL:")
                for p in passages[:2]:
                    print(f"      • \"{p.text[:220]}...\"")
            else:
                print(f"    Evidence: (No relevant passage found in cited source)")

        print("=" * 70)
        print(f"\n📁 Outputs saved to '{os.path.abspath(args.output)}':")
        print(f"   • Audit Report: {os.path.abspath(os.path.join(args.output, 'report.md'))}")
        print(f"   • Interactive Graph: {os.path.abspath(os.path.join(args.output, 'graph.html'))}")
        return

    # Mode 2: Multi-Claim Document Verification
    input_text = ""
    if args.text or args.claim:
        input_text = args.text or args.claim
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r") as fp:
            input_text = fp.read()
    else:
        print("Error: Please provide --text, --claim, or --audit-citation.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    print("🛡️  Initializing ProofGraph Pipeline...")
    pipeline = ProofGraphPipeline()

    print(f"\n🔍 Analyzing Document ({len(input_text)} characters)...")
    result = pipeline.verify(
        document_text=input_text,
        output_dir=args.output,
        use_live_web_fallback=not args.no_live_search
    )

    print(f"\n✅ Verification Complete!")
    print(f"   • Atomic Claims Extracted: {len(result.claims)}")
    for idx, c in enumerate(result.claims, 1):
        v = result.verdicts.get(c.id)
        label = v.label.value if v else "UNVERIFIED"
        tier = v.evidence_tier.value if v else "UNKNOWN"
        print(f"\n     [{idx}] [{label}] [TIER: {tier}] {c.text}")
        if c.is_underspecified and c.ambiguity_reason:
            print(f"         ⚠️  Ambiguity Notice: {c.ambiguity_reason}")
            print(f"         Action Required: Provide subject full name or context to verify.")
        if v and v.justification:
            print(f"         Justification: {v.justification}")
        for passage in result.evidence_map.get(c.id, []):
            method = passage.metadata.get("retrieval_method", "local")
            print(f"         Evidence: {passage.source_title} | via {method} | {passage.source_url or '(no URL)'}")

    print(f"\n📁 Outputs saved to '{os.path.abspath(args.output)}':")
    print(f"   • Audit Report: {os.path.abspath(os.path.join(args.output, 'report.md'))}")
    print(f"   • Interactive Graph: {os.path.abspath(os.path.join(args.output, 'graph.html'))}")

if __name__ == "__main__":
    main()
