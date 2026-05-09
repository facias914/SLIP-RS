# model settings
pretrained = './model_weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'  # noqa
text_encoder_checkpoint = './model_weights/remoteclip_ft.pth'

model = dict(
    type='SLIP_RS',
    backbone=dict(
        type='ConvNeXt',
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        pretrain=pretrained),
    text_encoder=dict(
        type='CLIP_Text_Encoder',
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        position = "all",
        params = ['q', 'k', 'v'],
        r = 2,
        alpha = 1,
        dropout_rate = 0.0,
        checkpoint=text_encoder_checkpoint,
        freeze=True),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='SLIP_RS_RoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='SLIP_RS_FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            embed_dim=512,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='HieAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=1800,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kl',
                topk=[2,1],
                ratio=0.9), 
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                gpu_assign_thr=1800),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))


# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsCustom', with_bbox=True, with_text=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleCustom'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'text_prompts']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsCustom', with_bbox=True, with_text=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundleCustom'),
            dict(type='Collect', keys=['img', 'text_prompts']),
        ])
]


dataset_type = 'RS_Dataset'
attri_dataset_type = 'RS_Dataset_Attri'

# RS-O
dota_cls_dict = ("plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "ship", 
                "tennis-court", "basketball-court", "storage-tank", "soccer-ball-field", "roundabout", "harbor", 
                "swimming-pool", "helicopter", "container-crane", "airport", "helipad")
dior_cls_dict = ("airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam", 
                "Expressway-Service-area", "Expressway-toll-station", "golffield", "groundtrackfield", 
                "harbor", "overpass", "ship", "stadium", "storagetank", "tenniscourt", "trainstation", 
                "vehicle", "windmill")
aitodv2_cls_dict = ("plane", "bridge", "storage-tank", "ship", "swimming-pool", "vehicle", "person", "wind-mill")
dronevehicle_cls_dict = ("van", "car", "truck", "freight-car", "bus")
nwpuvhr10_cls_dict = ("plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", 
                        "ground-track-field", "harbor", "bridge", "vehicle")
rsdgod_cls_dict = ("airport", "helicopter", "oiltank", "plane", "warship")
rsod_cls_dict = ("aircraft", "playground", "oiltank", "overpass")
soda_cls_dict = ("plane", "helicopter", "small-vehicle", "large-vehicle", "ship", "container", "storage-tank", 
                "swimming-pool", "wind-mill")
bridge_cls_dict = ("bridge", )
ghbridge_cls_dict = ("bridge", )
hrplane_cls_dict = ("plane", )
fair1m_cls_dict = ("small-vehicle", "large-vehicle", "plane", "ship", "tennis-court", "intersection", "baseball-diamond", 
                    "bridge", "soccer-ball-field", "roundabout", "basketball-court")
mar20_cls_dict = ("plane", )
shiprsimagenet_cls_dict = ("ship", )
simd_cls_dict = ("small-vehicle", "large-vehicle", "plane", "helicopter", "ship")

# RS-C
rs_c_cls_dict = ("plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "ship", 
                    "tennis-court", "basketball-court", "storage-tank", "soccer-ball-field", "roundabout", "harbor", 
                    "swimming-pool", "helicopter", "container-crane", "airport", "helipad")
rs_c2_cls_dict = ("windmill", "chimney", "container-crane", "helipad", "helicopter")

# RS-Attri-O
attri_dict = {"Plane" : {'Engine position': ['At wing roots and lower fuselage', 'Beneath the wings',
                                            'On the nose', 'Rear fuselage', 'Above the wings', 'Embedded within wing'],
                        'Number of engines': ['Eight-engine', 'Four-engine', 'One-engine', 'Twin-engine', 'Ten-engine'],
                        'Propulsion type': ['Jet', 'Propeller'],
                        'Purpose': ['AerialSupport Aircraft', 'Airborne Early Warning Aircraft', 'Airline Aircraft',
                                    'Anti-Submarine Warfare Aircraft', 'Bomber', 'Chartered aircraft', 'Fighter',
                                    'Propeller', 'Trainer', 'Transport Aircraft', 'Attack aircraft'],
                        'Usage': ['Civilian Aircraft', 'Commercial Aircraft', 'Military Aircraft'],
                        'Wing configuration': ['Straight wing', 'Swept delta wing', 'Swept diamond-like wing',
                                                'Swept wing', 'Swept, variable-sweep wing', 'Flying wing']},
            "Ship" : {'Usage': ['Civilian Ship', 'Commercial Ship', 'Engineering Ship', 'Military Ship'],
                        'Subcat': ['Barge', 'Container Ship', 'Dry Cargo Ship',
                                    'Cruise Ship', 'Liquid Cargo Ship', 'RoRo', 'Yacht'],
                        'Purpose': ['Aircraft Carrier', 'Amphibious Ship', 'Auxiliary Ship', 'Cargo Ship', 'Commander', 
                                    'Cruiser', 'Destroyer', 'Frigate', 'Landing', 'Medical Ship', 'Military Transport Ship', 
                                    'Passenger Ship', 'Patrol', 'Submarine', 'Test ship', 'Training ship', 'Tugboat',
                                    'Fishing Vessel', 'Motorboat']},
            "Vehicle" : {'Purpose': ['Bus', 'Cargo Truck', 'Dump Truck', 'Excavator', 'Pick-up', 'Small Passenger Car',
                                     'Tractor', 'Truck Tractor', 'Van'],
                         'Usage': ['Engineering Vehicle', 'Large Civilian Vehicle', 'Small Civilian Vehicle', 'Truck']}}


# train data path
## RS-O
train_rs_o_data_root = '/path/to/RS_O'
## RS-Attri-O
train_rs_o_attri_data_root = '/path/to/RS_Attri_O'
## RS-C
train_rs_c_data_root = '/path/to/RS_C'
## RS-Attri-C
train_rs_c_attri_data_root = '/path/to/RS_C'

# test data path
test_attri_data_root = '/path/to/Attribute_test'


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[
            # RS-O
            dict(
                type=dataset_type,
                classes=dota_cls_dict,
                ann_file=train_rs_o_data_root + '/dota/dota2_train_label.json',
                img_prefix=train_rs_o_data_root + '/dota/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=dior_cls_dict,
                ann_file=train_rs_o_data_root + '/dior/dior_train_label.json',
                img_prefix=train_rs_o_data_root + '/dior/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=aitodv2_cls_dict,
                ann_file=train_rs_o_data_root + '/aitod2/annotations.json',
                img_prefix=train_rs_o_data_root + '/aitod2/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=dronevehicle_cls_dict,
                ann_file=train_rs_o_data_root + '/dronevehicle/annotations.json',
                img_prefix=train_rs_o_data_root + '/dronevehicle/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=nwpuvhr10_cls_dict,
                ann_file=train_rs_o_data_root + '/nwpuvhr10/annotations.json',
                img_prefix=train_rs_o_data_root + '/nwpuvhr10/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=rsdgod_cls_dict,
                ann_file=train_rs_o_data_root + '/rsgod/annotations.json',
                img_prefix=train_rs_o_data_root + '/rsgod/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=rsod_cls_dict,
                ann_file=train_rs_o_data_root + '/rsod/annotations.json',
                img_prefix=train_rs_o_data_root + '/rsod/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=soda_cls_dict,
                ann_file=train_rs_o_data_root + '/soda/annotations.json',
                img_prefix=train_rs_o_data_root + '/soda/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=bridge_cls_dict,
                ann_file=train_rs_o_data_root + '/bridge/annotations.json',
                img_prefix=train_rs_o_data_root + '/bridge/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=ghbridge_cls_dict,
                ann_file=train_rs_o_data_root + '/ghbridge/annotations.json',
                img_prefix=train_rs_o_data_root + '/ghbridge/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=hrplane_cls_dict,
                ann_file=train_rs_o_data_root + '/hrplane/annotations.json',
                img_prefix=train_rs_o_data_root + '/hrplane/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=fair1m_cls_dict,
                ann_file=train_rs_o_data_root + '/fair1m/annotations_one.json',
                img_prefix=train_rs_o_data_root + '/fair1m/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=mar20_cls_dict,
                ann_file=train_rs_o_data_root + '/mar20/annotations.json',
                img_prefix=train_rs_o_data_root + '/mar20/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=shiprsimagenet_cls_dict,
                ann_file=train_rs_o_data_root + '/shiprsimagenet/annotations.json',
                img_prefix=train_rs_o_data_root + '/shiprsimagenet/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=simd_cls_dict,
                ann_file=train_rs_o_data_root + '/simd/annotations_one.json',
                img_prefix=train_rs_o_data_root + '/simd/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            # RS-C
            dict(
                type=dataset_type,
                classes=rs_c_cls_dict,
                ann_file=train_rs_c_data_root + '/Asia/annotations.json',
                img_prefix=train_rs_c_data_root + '/Asia/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=rs_c_cls_dict,
                ann_file=train_rs_c_data_root + '/Europe/annotations.json',
                img_prefix=train_rs_c_data_root + '/Europe/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=rs_c_cls_dict,
                ann_file=train_rs_c_data_root + '/North_America/annotations.json',
                img_prefix=train_rs_c_data_root + '/North_America/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=rs_c_cls_dict,
                ann_file=train_rs_c_data_root + '/Others/annotations.json',
                img_prefix=train_rs_c_data_root + '/Others/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            dict(
                type=dataset_type,
                classes=rs_c2_cls_dict,
                ann_file=train_rs_c_data_root + '/Others1/annotations.json',
                img_prefix=train_rs_c_data_root + '/Others1/images/',
                pipeline=train_pipeline,
                text_template='{}'),
            # RS-Attri-O
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                ann_file=train_rs_o_attri_data_root + '/annotations.json',
                img_prefix=train_rs_o_attri_data_root + '/images/',
                pipeline=train_pipeline),
            # RS-Attri-C
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                ann_file=train_rs_c_attri_data_root + '/Asia/annotations_attribute.json',
                img_prefix=train_rs_c_attri_data_root + '/Asia/images/',
                pipeline=train_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                ann_file=train_rs_c_attri_data_root + '/Europe/annotations_attribute.json',
                img_prefix=train_rs_c_attri_data_root + '/Europe/images/',
                pipeline=train_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                ann_file=train_rs_c_attri_data_root + '/North_America/annotations_attribute.json',
                img_prefix=train_rs_c_attri_data_root + '/North_America/images/',
                pipeline=train_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                ann_file=train_rs_c_attri_data_root + '/Others_Attri/annotations.json',
                img_prefix=train_rs_c_attri_data_root + '/Others_Attri/images/',
                pipeline=train_pipeline)
        ],
        separate_eval=True  
),
    val=dict(
        type='ConcatDataset',
        datasets=[
            # DOTA2
            dict(
                type=dataset_type,
                classes=dota_cls_dict,
                ann_file='/path/to/dota2_val_label.json',
                img_prefix='/path/to/val_images/',
                test_mode=True,
                pipeline=test_pipeline,
                text_template='{}'),
            # DIOR
            dict(
                type=dataset_type,
                classes=dior_cls_dict,
                ann_file='/path/to/test.json',
                img_prefix='/path/to/test_images/',
                test_mode=True,
                pipeline=test_pipeline,
                text_template='{}'),
            # Attribute
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Vehicle",
                test_attri="Usage",
                test_mode=True,
                ann_file=test_attri_data_root + '/vehicle/annotations.json',
                img_prefix=test_attri_data_root + '/vehicle/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Vehicle",
                test_attri="Purpose",
                test_mode=True,
                ann_file=test_attri_data_root + '/vehicle/annotations.json',
                img_prefix=test_attri_data_root + '/vehicle/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Propulsion type",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Number of engines",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Wing configuration",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Engine position",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Purpose",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Usage",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Ship",
                test_attri="Usage",
                test_mode=True,
                ann_file=test_attri_data_root + '/ship/annotations.json',
                img_prefix=test_attri_data_root + '/ship/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Ship",
                test_attri="Purpose",
                test_mode=True,
                ann_file=test_attri_data_root + '/ship/annotations.json',
                img_prefix=test_attri_data_root + '/ship/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Ship",
                test_attri="Subcat",
                test_mode=True,
                ann_file=test_attri_data_root + '/ship/annotations.json',
                img_prefix=test_attri_data_root + '/ship/images/',
                pipeline=test_pipeline)
        ],
        separate_eval=True  # 是否分开评测
    ),
    test=dict(
        type='ConcatDataset',
        datasets=[
            # DOTA2
            dict(
                type=dataset_type,
                classes=dota_cls_dict,
                ann_file='/path/to/dota2_val_label.json',
                img_prefix='/path/to/val_images/',
                test_mode=True,
                pipeline=test_pipeline,
                text_template='{}'),
            # DIOR
            dict(
                type=dataset_type,
                classes=dior_cls_dict,
                ann_file='/path/to/test.json',
                img_prefix='/path/to/test_images/',
                test_mode=True,
                pipeline=test_pipeline,
                text_template='{}'),
            # Attribute
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Vehicle",
                test_attri="Usage",
                test_mode=True,
                ann_file=test_attri_data_root + '/vehicle/annotations.json',
                img_prefix=test_attri_data_root + '/vehicle/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Vehicle",
                test_attri="Purpose",
                test_mode=True,
                ann_file=test_attri_data_root + '/vehicle/annotations.json',
                img_prefix=test_attri_data_root + '/vehicle/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Propulsion type",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Number of engines",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Wing configuration",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Engine position",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Purpose",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri="Usage",
                test_mode=True,
                ann_file=test_attri_data_root + '/plane/annotations.json',
                img_prefix=test_attri_data_root + '/plane/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Ship",
                test_attri="Usage",
                test_mode=True,
                ann_file=test_attri_data_root + '/ship/annotations.json',
                img_prefix=test_attri_data_root + '/ship/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Ship",
                test_attri="Purpose",
                test_mode=True,
                ann_file=test_attri_data_root + '/ship/annotations.json',
                img_prefix=test_attri_data_root + '/ship/images/',
                pipeline=test_pipeline),
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Ship",
                test_attri="Subcat",
                test_mode=True,
                ann_file=test_attri_data_root + '/ship/annotations.json',
                img_prefix=test_attri_data_root + '/ship/images/',
                pipeline=test_pipeline),
            # You can use any number and order of attributes to test the recognition capability of attribute permutations and combinations.
            dict(
                type=attri_dataset_type,
                attri_dict=attri_dict,
                test_cls="Plane",
                test_attri=["Usage", "Purpose", "Propulsion type", "Engine position", "Number of engines", "Wing configuration"],
                test_mode=True,
                ann_file=test_plane_data_root + '/annotations.json',
                img_prefix=test_plane_data_root + '/images/',
                pipeline=test_pipeline)
        ],
        separate_eval=True 
    ))
evaluation = dict(interval=1, metric='bbox', proposal_nums=(300, 1000, 3000))


# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00001,  ################################
    betas=(0.9, 0.999),
    weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# # yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)