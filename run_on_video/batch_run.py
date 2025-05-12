import os
import json
from tqdm import tqdm
from run import MomentDETRPredictor
from utils.basic_utils import load_jsonl

def run_batch_predictions(
    video_dir,
    query_file,
    output_file,
    ckpt_path="run_on_video/moment_detr_ckpt/model_best.ckpt",
    clip_model_name_or_path="ViT-B/32",
    device="cuda"
):
    """
    Run MomentDETR predictions on multiple videos and queries
    
    Args:
        video_dir: Directory containing video files
        query_file: JSONL file containing queries for each video
        output_file: Where to save the predictions
        ckpt_path: Path to the model checkpoint
        clip_model_name_or_path: CLIP model to use
        device: Device to run on ("cuda" or "cpu")
    """
    # Load queries
    queries = load_jsonl(query_file)
    
    # Initialize model
    print("Initializing model...")
    predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device=device
    )
    
    # Run predictions
    all_predictions = []
    for query in tqdm(queries, desc="Processing queries"):
        video_id = query.get('vid', query.get('video_id'))
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found, skipping...")
            continue
            
        try:
            predictions = predictor.localize_moment(
                video_path=video_path,
                query_list=[query['query']]
            )
            
            # Add video_id and qid to predictions
            for pred in predictions:
                pred['video_id'] = video_id
                pred['qid'] = query['qid']
                all_predictions.append(pred)
                
        except Exception as e:
            print(f"Error processing query {query['qid']} for video {video_id}: {str(e)}")
            continue
    
    # Save predictions
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"Processed {len(queries)} queries")
    print(f"Generated {len(all_predictions)} predictions")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Directory containing video files")
    parser.add_argument("--query_file", required=True, help="JSONL file containing queries")
    parser.add_argument("--output_file", required=True, help="Where to save predictions")
    parser.add_argument("--ckpt_path", default="run_on_video/moment_detr_ckpt/model_best.ckpt",
                      help="Path to model checkpoint")
    parser.add_argument("--clip_model", default="ViT-B/32",
                      help="CLIP model to use")
    parser.add_argument("--device", default="cuda",
                      help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    run_batch_predictions(
        video_dir=args.video_dir,
        query_file=args.query_file,
        output_file=args.output_file,
        ckpt_path=args.ckpt_path,
        clip_model_name_or_path=args.clip_model,
        device=args.device
    ) 