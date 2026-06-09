import os
import sys
import pandas as pd
import json
import csv
def process_json_list_to_csv(json_path, csv_output_path):
    with open(json_path, "r") as F_:
        json_dict = json.load(F_)
    raw_data_to_write = []
    N = len(json_dict["video_paths"])
    '''
    self.path = metadata["file_name"].to_list()
    self.mask_video_path = metadata["mask_name"].to_list()
    self.text = metadata["text"].to_list()
    '''
    for i in range(N):
        raw_data_to_write.append({
            "video_paths":json_dict["video_paths"][i],
            "mask_paths":json_dict["mask_paths"][i],
            "prompts":json_dict["prompts"][i].replace("\n","")
        })
    print(f"info number: {len(raw_data_to_write)}")
    fieldnames = ["video_paths", "mask_paths", "prompts"]
    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write header row
        writer.writerows(raw_data_to_write)  # Write data rows
#anyway;
if __name__ == "__main__":
    #import pandas as pd
    #process_json_list_to_csv("/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/dataprocess_minimal/datalists/split_0529_blurred_all.json", "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/DiffSynth-Studio/dataset/csv_blur/metadata.csv")
    #process_json_list_to_csv("/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/dataprocess_minimal/datalists/split_0603_26000.json", "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/DiffSynth-Studio/dataset/csv_100k_new/metadata.csv")
    #process_json_list_to_csv("/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/dataprocess_minimal/datalists/split_0522_98frames_rest_test.json", "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/DiffSynth-Studio/dataset/csv_rest_old_data/metadata_test.csv")
    #process_json_list_to_csv("/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/TrajectoryCrafter_friday/assets/inference_far_nvd/video/meta.json", "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/DiffSynth-Studio/dataset/csv_test/metadata.csv")
    process_json_list_to_csv("/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/dataprocess_minimal/datalists/split_0606_78000.json", "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/DiffSynth-Studio/dataset/csv_78k_0606/metadata.csv")
    
    