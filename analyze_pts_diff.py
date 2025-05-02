def load_pts(file):
    with open(file, "r") as f:
        return [
            float(line.strip().rstrip(',')) 
            for line in f 
            if line.strip().rstrip(',').replace(".", "", 1).isdigit()
        ]


def analyze_diff(pts):
    diffs = [pts[i+1] - pts[i] for i in range(len(pts)-1)]
    print(f"平均間隔: {sum(diffs)/len(diffs):.6f} 秒")
    print(f"最小間隔: {min(diffs):.6f}，最大間隔: {max(diffs):.6f}")
    print(f"有異常間隔 (>1.5x 平均) 的幀數: {sum(d > 1.5 * diffs[0] for d in diffs)}")

pts1 = load_pts("pts_1.txt")
pts2 = load_pts("pts_2.txt")
print("=== Camera 1 ===")
analyze_diff(pts1)
print("=== Camera 2 ===")
analyze_diff(pts2)
