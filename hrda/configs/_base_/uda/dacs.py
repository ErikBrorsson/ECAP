# ---------------------------------------------------------------
# Copyright (c) 2023-2024 Volvo Group, Erik Brorsson. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------

# The code is based on: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
img_norm_cfg_dacs = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
uda = dict(
    type='DACS',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,
    ecap=False,
    ecap_config=dict(
        type='ECAP',
        tmp_dir=None,
        mean=img_norm_cfg_dacs["mean"],
        std=img_norm_cfg_dacs["std"],
        p_ea = 2.5,
        start=10000,
        rampup=10000,
        ea_sigmoid=True,
        ea_beta=0.96,
        ea_gamma=0.005,
        n_rounds=1,
        max_bank_size=50,
        rot_degree=20,
        min_scale=0.1,
        synthia=False,
        crop_margins=[30, 240]
    )
)
use_ddp_wrapper = True
