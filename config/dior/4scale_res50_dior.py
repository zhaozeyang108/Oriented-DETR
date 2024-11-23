num_classes = 91
ignore_cls = [] # these classes are 0 degree default
inf_conf = 0.005 # for inference, set 0.005 for accurate coco metrics
#lr_params
lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 2
weight_decay = 0.0001
epochs = 36
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = True
lr_drop_list = [27, 33]
modelname = 'odetr'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = False

#model params
dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
num_points = 13
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_angle_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 900
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
embed_init_tgt = True

# mathching cost
set_cost_class = 2.0
set_cost_bbox = 15.0
set_cost_angle = 2.0

# loss coef
cls_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 15.0
angle_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
no_interm_angle_loss = False
focal_alpha = 0.25

# decoder config
decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
matcher_type = 'AngleHungarianMatcher'
decoder_module_seq = ["intra_sa", "sa", 'ca', 'ffn']
nms_iou_threshold = -1
dec_pred_bbox_embed_share = True
dec_pred_angle_embed_share = True
dec_pred_class_embed_share = True


# aug coef
aug_ratio = 0.5
augment = True
degrees = 180.0
translate = 0.1
scale = 0.25
shear = 0.0
perspective = 0.0
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
flipud = 0.5
fliplr = 0.5
mosaic_border = [-512, -512]
mosaic_p = 0.5
img_size = 800
mosaic = True
