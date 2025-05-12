import json
import argparse

def convert_queries(input_file, output_file):
    """
    Convert queries from input JSONL format to MomentDETR format
    
    Input format:
    {
        "qid": int,
        "query": str,
        "duration": int,
        "vid": str
    }
    
    Output format:
    {
        "vid": str,
        "query": str
    }
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            # Extract just the video ID without timestamps if present
            vid = data['vid']
            qid = data['qid']
            
            # Create new entry in MomentDETR format
            new_entry = {
                "vid": vid,
                "query": data['query'],
                "qid": qid
            }
            
            # Write to output file
            f_out.write(json.dumps(new_entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Input JSONL file with queries")
    parser.add_argument("--output_file", required=True, help="Output JSONL file in MomentDETR format")
    
    args = parser.parse_args()
    convert_queries(args.input_file, args.output_file)
    print(f"Converted queries from {args.input_file} to {args.output_file}") 