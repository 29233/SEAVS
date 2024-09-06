model = dict(
    type='AVS_Model',
    backbone=dict(
        type='pvt_v2_b5',
        init_weights_path='pretrained/pvt_v2_b5.pth'),
    vggish=dict(
        freeze_audio_extractor=True,
        pretrained_vggish_model_path='pretrained/vggish-10086976.pth',
        preprocess_audio_to_log_mel=True,
        postprocess_log_mel_with_pca=False,
        pretrained_pca_params_path=None),
    bridger=dict(
        type='TemporalFusers',
        nhead=8,
        t=5,
        num_queries=5,
        dropout=0.1,
        feature_channel=[64, 128, 320, 512],
    ),
    head=dict(
        type='RefFormerHead',
        scale_factors=[8, 4, 2, 1],
        d_models=[64, 128, 320, 512],
        nhead=8,
        pos_emb=None,
        drop_out=0.1,
        conv_dim=256,
        mask_dim=71,
        use_bias=False,
        interpolate_scale=4
    ),
    projector=dict(
        type='None',
    ),
    freeze=dict(
        audio_backbone=True,
        visual_backbone=True
    )
)
dataset = dict(
    train=dict(
        type='V2Dataset',
        split='train',
        num_class=71,
        mask_num=10,
        crop_img_and_mask=True,
        crop_size=224,
        meta_csv_path='/data/AVSS/metadata.csv',
        label_idx_path='/data/AVSS/label2idx.json',
        dir_base='/data/AVSS',
        img_size=(224, 224),
        batch_size=4),
    val=dict(
        type='V2Dataset',
        split='val',
        num_class=71,
        mask_num=10,
        crop_img_and_mask=True,
        crop_size=224,
        meta_csv_path='/data/AVSS/metadata.csv',
        label_idx_path='/data/AVSS/label2idx.json',
        dir_base='/data/AVSS',
        img_size=(224, 224),
        resize_pred_mask=True,
        save_pred_mask_img_size=(360, 240),
        batch_size=4),
    test=dict(
        type='V2Dataset',
        split='test',
        num_class=71,
        mask_num=10,
        crop_img_and_mask=True,
        crop_size=224,
        meta_csv_path='/data/AVSS/metadata.csv',
        label_idx_path='/data/AVSS/label2idx.json',
        dir_base='/data/AVSS',
        img_size=(224, 224),
        resize_pred_mask=True,
        save_pred_mask_img_size=(360, 240),
        batch_size=4))
optimizer = dict(
    type='AdamW',
    lr=6e-5)
loss = dict(
    weight_dict=dict(
        focal_loss=1.0),
    loss_type='dice')
process = dict(
    num_works=8,
    train_epochs=30,
    start_eval_epoch=10,
    eval_interval=1,
    freeze_epochs=-1)
discribe = dict(
    session_name='AVSS',
    info='ETAVS_res50_AVSS'
)