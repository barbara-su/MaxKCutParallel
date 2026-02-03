import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Extract (graph_id, total_time) from gen_V_from_Q JSON logs"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output JSON file"
    )
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    out_records = []

    for rec in data["records"]:
        # Prefer numeric ID for downstream analysis
        graph_id = rec.get("gset_id", rec.get("graph_name"))

        total_time = rec["timing_seconds"]["total"]

        out_records.append(
            {
                "graph_id": graph_id,
                "total_time_seconds": total_time,
            }
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(
            {
                "num_records": len(out_records),
                "records": out_records,
            },
            f,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
