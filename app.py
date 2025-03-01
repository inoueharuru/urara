import os
import SimpleITK as sitk
import h5py
import pandas as pd
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from flask import Flask, request, jsonify
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score
from torch.cuda.amp import GradScaler, autocast
from docx import Document
import hashlib
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import torch.backends.cudnn as cudnn

# 设置随机种子以保证结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
cudnn.deterministic = True
cudnn.benchmark = False

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    
# 模型定义
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=0, keepdim=True)
        excitation = self.fc1(avg_pool)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        return x * excitation

class CrossModalAttention(nn.Module):
    def __init__(self, ct_dim=1280, pathology_dim=1280):
        super(CrossModalAttention, self).__init__()
        self.query_ct = nn.Linear(ct_dim, 1280)
        self.key_pathology = nn.Linear(pathology_dim, 1280)
        self.value_pathology = nn.Linear(pathology_dim, 1280)
        self.attn = nn.MultiheadAttention(1280, num_heads=1)

    def forward(self, ct_features, pathology_features):
        ct_features_proj = self.query_ct(ct_features)
        ct_features_transformed = ct_features_proj.unsqueeze(0)
        pathology_key = self.key_pathology(pathology_features).unsqueeze(0)
        pathology_value = self.value_pathology(pathology_features).unsqueeze(0)
        attn_output, _ = self.attn(ct_features_transformed, pathology_key, pathology_value)
        return attn_output.squeeze(0) + ct_features_proj

class MultiModalDiseaseClassifier(nn.Module):
    def __init__(self, dropout_p=0.8):
        super(MultiModalDiseaseClassifier, self).__init__()
        self.ct_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        for param in self.ct_model.parameters():
            param.requires_grad = False
        for param in self.ct_model.layer4.parameters():
            param.requires_grad = True
        self.pathology_feature_adjuster = nn.Linear(1024, 1280)
        self.pathology_fc = nn.Linear(1280, 1280)
        self.ct_feature_transform = nn.Linear(1000, 1280)
        self.pathology_transform = nn.Linear(1280, 1280)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.cross_modal_attention = CrossModalAttention(1280)
        self.fusion_fc = nn.Linear(2560, 1280)
        self.se_block = SEBlock(1280)
        self.extra_fc = nn.Linear(1280, 1280)
        self.fc = nn.Linear(1280, 2)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, masked_images, pathology_features):
        batch_size, num_slices, channels, height, width = masked_images.shape
        masked_images = masked_images.view(batch_size * num_slices, channels, height, width)
        ct_features = self.ct_model(masked_images)
        ct_features = ct_features.view(batch_size, num_slices, -1)
        ct_features = ct_features.mean(dim=1)
        ct_features = self.ct_feature_transform(ct_features)
        pathology_features = self.pathology_feature_adjuster(pathology_features)
        fusion_input = torch.cat([ct_features, pathology_features], dim=1)
        fusion_features = self.cross_modal_attention(ct_features, pathology_features)
        fusion_features = self.fusion_fc(fusion_input)
        alpha = self.alpha.view(-1, 1)
        fused_features = alpha * ct_features + (1 - alpha) * pathology_features
        fused_features = self.se_block(fused_features)
        fused_features = self.dropout(fused_features)
        fused_features = self.extra_fc(fused_features)
        outputs = self.fc(fused_features)
        return outputs

# 加载模型
def load_model(model_path, dropout_p=0.2):
    model = MultiModalDiseaseClassifier(dropout_p=dropout_p).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 处理 CT 影像
def process_ct_image(image_path, mask_path, num_slices=25):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    lesion_slices = np.any(mask_array > 0, axis=(1, 2))
    slice_indices = np.where(lesion_slices)[0]
    if len(slice_indices) == 0:
        raise ValueError(f"No lesion found in mask: {mask_path}")
    num_slices_to_keep = max(1, len(slice_indices) // 2)
    selected_slices = np.random.choice(slice_indices, num_slices_to_keep, replace=True)
    masked_slices = []
    for idx in selected_slices:
        masked_slice = np.where(mask_array[idx] > 0, image_array[idx], 0)
        masked_slices.append(masked_slice)
    num_available_slices = len(masked_slices)
    if num_available_slices >= num_slices:
        selected_indices = np.random.choice(num_available_slices, num_slices, replace=False)
    else:
        selected_indices = np.random.choice(num_available_slices, num_slices, replace=True)
    processed_slices = [masked_slices[i] for i in selected_indices]
    masked_images = np.stack(processed_slices, axis=0)
    masked_images = np.expand_dims(masked_images, axis=1)
    masked_images = np.repeat(masked_images, 3, axis=1)
    masked_images = torch.tensor(masked_images, dtype=torch.float32).to(device)
    return masked_images

# 处理病理特征
def process_pathology_features(pathology_path):
    with h5py.File(pathology_path, 'r') as h5_file:
        features = h5_file['features'][:]
    avg_features = features.mean(axis=0)
    pathology_features = torch.tensor(avg_features, dtype=torch.float32).to(device)
    return pathology_features

# 模型推理
def predict(model, ct_image_path, mask_path, pathology_path, num_slices=25):
    ct_images = process_ct_image(ct_image_path, mask_path, num_slices)
    pathology_features = process_pathology_features(pathology_path)
    with torch.no_grad():
        outputs = model(ct_images.unsqueeze(0), pathology_features.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

app = Flask(__name__)


# 上传文件目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        ct_file = request.files['ctFile']
        pathology_file = request.files['pathologyFile']

        if ct_file and pathology_file:
            ct_filename = ct_file.filename
            pathology_filename = pathology_file.filename

            ct_file_path = os.path.join(UPLOAD_FOLDER, ct_filename)
            pathology_file_path = os.path.join(UPLOAD_FOLDER, pathology_filename)

            ct_file.save(ct_file_path)
            pathology_file.save(pathology_file_path)

            # 这里假设 mask 文件路径是固定的，实际应用中需要根据具体情况处理
            mask_file_path = 'path/to/mask/file'

            return jsonify({
                'ctFilePath': ct_file_path,
                'pathologyFilePath': pathology_file_path,
                'maskFilePath': mask_file_path
            })
        else:
            return jsonify({"error": "缺少必要的文件"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 加载训练好的模型
model_path = r"D:\python-learning\learning\joint\best_model_fold_para43_663340de.pth"  # 请替换为实际的模型路径
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        ct_image_path = data.get('ct_image_path')
        mask_path = data.get('mask_path')
        pathology_path = data.get('pathology_path')

        if not ct_image_path or not mask_path or not pathology_path:
            return jsonify({"error": "Missing required fields"}), 400

        result = predict(model, ct_image_path, mask_path, pathology_path)
        label_dict = {0: 'OF', 1: 'FD'}
        predicted_label = label_dict[result]

        return jsonify({"prediction": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    # from waitress import serve
    # serve(app, host='127.0.0.1', port=5000)