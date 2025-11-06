import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

INFO_DATASET = {
    "annotation_path": "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/dataset_final/info/pinpoint_info_train_val_llava.json",
    "data_path": "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/infographic/images" 
}

MPDOC_DATASET = {
    "annotation_path": "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/mpdoc/combined_mpdoc/train_llava.json",
    "data_path": "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/mpdoc/combined_mpdoc/images" 
}

SPDOC_DATASET = {
    "annotation_path": "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/dataset_final/spdoc/pinpoint_spdoc_train_llava.json",
    "data_path": "/root/Desktop/workspace/kwon/pinpoint/pinpoint_dataset/spdoc" 
}

# 2. data_dict에 "등록" (이 이름을 .sh 파일에서 사용)
data_dict = {
    "info_dataset": INFO_DATASET,
    "mpdoc_dataset": MPDOC_DATASET,
    "spdoc_dataset": SPDOC_DATASET,
}

# data_dict = {
#     "cambrian_737k": CAMBRIAN_737K,
#     "cambrian_737k_pack": CAMBRIAN_737K_PACK,
#     "mp_doc": MP_DOC,
#     "clevr_mc": CLEVR_MC,
#     "videochatgpt": VIDEOCHATGPT,
# }


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
