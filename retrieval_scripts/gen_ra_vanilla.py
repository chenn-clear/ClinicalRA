import os
import argparse
import numpy as np
import faiss
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Vanilla RA Splits")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to original split folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save RA splits")
    parser.add_argument("--k", type=int, default=3, help="Number of items to retrieve")
    return parser.parse_args()

def load_data(path):
    """Read the file and return (lines_content, paths, labels, attrs_matrix)."""
    lines_content = []
    paths = []
    labels = []
    attrs = []
    
    if not os.path.exists(path):
        return [], [], [], np.array([])

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            lines_content.append(line)
            parts = line.split('\t')
            paths.append(parts[0])
            labels.append(int(parts[1]))
            attrs.append([float(x) for x in parts[2:]])
            
    return lines_content, paths, np.array(labels), np.array(attrs)

def write_ra_file(output_path, query_lines, retrieved_indices, db_lines, k, is_train_on_train):
    """Write RA formatted file."""
    with open(output_path, 'w') as f:
        for i, query_line in enumerate(query_lines):
            # Build the output line
            out_str = query_line
            
            # Get retrieval results
            # If Train retrieves Train, retrieved_indices[i][0] is itself, skip it
            start_idx = 1 if is_train_on_train else 0
            end_idx = start_idx + k
            
            current_indices = retrieved_indices[i][start_idx:end_idx]
            
            for idx in current_indices:
                # db_lines[idx] is a string like "path \t label \t attr..."
                # Simply append it
                out_str += "\t" + db_lines[idx]
            
            f.write(out_str + "\n")

def process_fold(fold_idx, args):
    train_file = os.path.join(args.input_dir, f"train_fold{fold_idx}.txt")
    test_file = os.path.join(args.input_dir, f"test_fold{fold_idx}.txt")
    
    # 1. Read data
    train_lines, train_paths, train_y, train_raw = load_data(train_file)
    test_lines, test_paths, test_y, test_raw = load_data(test_file)
    
    if len(train_lines) == 0:
        print(f"Skipping Fold {fold_idx}: No training data.")
        return

    # 2. Min-Max normalization
    scaler = MinMaxScaler()
    train_feats = scaler.fit_transform(train_raw).astype(np.float32)
    # Test data can only be transformed
    test_feats = scaler.transform(test_raw).astype(np.float32) if len(test_raw) > 0 else np.array([])

    # 3. Build Faiss index (L2 Normalized Inner Product)
    d = train_feats.shape[1]
    index = faiss.IndexFlatIP(d)
    
    # Copy and apply L2 normalization (required by Faiss)
    db_feats = train_feats.copy()
    faiss.normalize_L2(db_feats)
    index.add(db_feats)
    
    # === Process Train Set (Self-Retrieval) ===
    # Retrieve k+1 items since the first is itself; skip via logic
    D_train, I_train = index.search(db_feats, args.k + 1)
    
    out_train_path = os.path.join(args.output_dir, f"train_fold{fold_idx}.txt")
    write_ra_file(out_train_path, train_lines, I_train, train_lines, args.k, is_train_on_train=True)
    
    # === Process Test Set ===
    if len(test_lines) > 0:
        query_feats = test_feats.copy()
        faiss.normalize_L2(query_feats)
        D_test, I_test = index.search(query_feats, args.k)
        
        out_test_path = os.path.join(args.output_dir, f"test_fold{fold_idx}.txt")
        write_ra_file(out_test_path, test_lines, I_test, train_lines, args.k, is_train_on_train=False)

    print(f"Fold {fold_idx} processed.")

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Assume standard 5 folds
    for fold in range(5):
        process_fold(fold, args)

if __name__ == "__main__":
    main()
