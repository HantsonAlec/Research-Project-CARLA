import json

lanes_data = []
for line in open('Carla-Lane-Detection-Dataset-Generation/src/data/dataset/Town01_Opt/train_gt_tmp.json', 'r'):
    lanes_data.append(json.loads(line))

for lane_data in lanes_data:
    title = lane_data['raw_file'].split("/")[-1][:4]
    f = open(f"labels/{title}.txt", "w")
    for i in range(len(lane_data['lanes'])):
        line = ""
        for j in range(len(lane_data['lanes'][i])):
            x = lane_data['lanes'][i][j]
            y = lane_data['h_samples'][j]
            coordinate = f"{x} {y}"
            line = line+coordinate+" "
        f.write(f"{line}\n")
    f.close()
