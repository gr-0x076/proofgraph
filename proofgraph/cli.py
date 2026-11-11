import argparse
import sys
import os
from proofgraph.pipeline import ProofGraphPipeline

def main():
    parser = argparse.ArgumentParser(
        description="ProofGraph: evidence-backed claim review prototype"
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

    input_text = ""
    if args.text:
        input_text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r") as fp:
            input_text = fp.read()
    else:
        print("Error: Please provide either --text or --file.", file=sys.stderr)
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
        conf = f"{v.confidence * 100:.1f}% (uncalibrated rule score)" if v else "N/A"
        print(f"     [{idx}] [{label}] ({conf}) {c.text}")
        for passage in result.evidence_map.get(c.id, []):
            method = passage.metadata.get("retrieval_method", "local")
            print(f"         Evidence: {passage.source_title} | via {method} | {passage.source_url or '(no URL)'}")

    print(f"\n📁 Outputs saved to '{os.path.abspath(args.output)}':")
    print(f"   • Audit Report: {os.path.abspath(os.path.join(args.output, 'report.md'))}")
    print(f"   • Interactive Graph: {os.path.abspath(os.path.join(args.output, 'graph.html'))}")

if __name__ == "__main__":
    main()
