# outfit.py — Aesthetic Dash UI with GPT Suggestions + Compatibility Check

import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import time
import os
import json
import sys
import torch
import re
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mcn.model import CompatModel
from mcn.utils import prepare_dataloaders
from mcn.hf_gpt_utils import generate_hf_suggestion as generate_gpt_suggestion

with open("../data/train_no_dup_with_category_3more_name.json") as f:
    name_map = json.load(f)

data_root = "../data"
img_root = os.path.join(data_root, "images")

_, _, _, _, test_dataset, _ = prepare_dataloaders(root_dir=img_root, num_workers=1)
device = torch.device("cpu")
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=2757).to(device)
model.load_state_dict(torch.load("../mcn/model_train_relation_vse_type_cond_scales.pth", map_location="cpu"))
model.eval()
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

def extract_item_names_from_paths(paths_dict):
    item_names = {}
    for part, full_path in paths_dict.items():
        try:
            parts = full_path.replace("\\", "/").split("/")
            outfit_id = parts[-2]
            index = int(parts[-1].split(".")[0])
            if outfit_id in name_map and part in name_map[outfit_id]:
                item_names[part] = name_map[outfit_id][part]["name"]
        except:
            item_names[part] = ""
    return item_names

def base64_to_tensor(image_bytes_dict):
    import torchvision.transforms as transforms
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    outfit_tensor = []
    for k, v in image_bytes_dict.items():
        img = base64_to_image(v)
        tensor = my_transforms(img)
        outfit_tensor.append(tensor.squeeze())
    outfit_tensor = torch.stack(outfit_tensor)
    outfit_tensor = outfit_tensor.to(device)
    return outfit_tensor

def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data).convert("RGB")
    return img

def defect_detect(img, model, normalize=True):
    relation = None
    def func_r(module, grad_in, grad_out):
        nonlocal relation
        relation = grad_in[1].detach()
    for name, module in model.named_modules():
        if name == 'predictor.0':
            module.register_backward_hook(func_r)
    out = model._compute_score(img)
    out = out[0]
    one_hot = torch.FloatTensor([[-1]]).to(device)
    model.zero_grad()
    out.backward(gradient=one_hot, retain_graph=True)
    if normalize:
        relation = relation / (relation.max() - relation.min())
    relation += 1e-3
    return relation, out.item()

def item_diagnosis(relation, select):
    mats = vec2mat(relation, select)
    for m in mats:
        mask = torch.eye(*m.shape, dtype=torch.bool)
        m.masked_fill_(mask, 0)
    result = torch.cat(mats).sum(dim=0)
    order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
    return result, order

def vec2mat(relation, select):
    mats = []
    for idx in range(4):
        mat = torch.zeros(5, 5)
        mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
        mat += torch.triu(mat, 1).transpose(0, 1)
        mat = mat[select, :]
        mat = mat[:, select]
        mats.append(mat)
    return mats

external_stylesheets = [dbc.themes.SLATE]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

categories = ["top", "bottom", "shoe", "bag", "accessory"]
app.layout = html.Div([
    dbc.Container([
        html.H2("👗 StyleSync: Aesthetic Outfit Advisor", style={"color": "#0ff", "textAlign": "center"}),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Label(f"Upload {cat.title()}", style={"color": "#17a2b8"}),
                dcc.Upload(id=f"upload-{cat}", children=html.Div("Click or Drag Here"), style={"height": "70px", "lineHeight": "70px", "border": "2px dashed #0ff", "textAlign": "center", "marginBottom": "10px"}),
                html.Img(id=f"preview-{cat}", style={"width": "100%", "height": "150px", "objectFit": "cover", "borderRadius": "10px"})
            ], md=2) for cat in categories
        ]),
        html.Br(),
        dcc.Input(id="occasion", type="text", placeholder="e.g. party, casual", style={"width": "100%", "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(html.Button("Check Compatibility", id="check-compat", style={"width": "100%"}), md=6),
            dbc.Col(html.Button("Get GPT Suggestion", id="get-gpt", style={"width": "100%"}), md=6)
        ]),
        html.Br(),
        html.Div(id="output-area", style={"color": "#eee"})
    ])
])

for cat in categories:
    @app.callback(
        Output(f"preview-{cat}", "src"),
        Input(f"upload-{cat}", "contents")
    )
    def update_preview(contents):
        return contents

@app.callback(
    Output("output-area", "children"),
    Input("get-gpt", "n_clicks"),
    [State(f"upload-{cat}", "filename") for cat in categories] + [State("occasion", "value")]
)
def generate_gpt(n, top, bottom, shoe, bag, accessory, occasion):
    if not n:
        return
    filenames = {"top": top, "bottom": bottom, "shoe": shoe, "bag": bag, "accessory": accessory}
    paths = {cat: os.path.join(img_root, fname) for cat, fname in filenames.items() if fname}
    item_names = extract_item_names_from_paths(paths)
    gpt_tip = generate_gpt_suggestion(item_names, occasion or "casual")
    return html.Div([
        html.H5("GPT Style Suggestion", style={"color": "#ffd700"}),
        html.P(gpt_tip)
    ])

@app.callback(
    Output("output-area", "children"),
    Input("check-compat", "n_clicks"),
    [State(f"preview-{cat}", "src") for cat in categories]
)
def check_compat(n_clicks, top, bottom, shoe, bag, accessory):
    if not n_clicks:
        return dash.no_update
    img_dict = {"top": top, "bottom": bottom, "shoe": shoe, "bag": bag, "accessory": accessory}
    if not all(img_dict.values()):
        return "Please upload all category images."
    tensor = base64_to_tensor(img_dict)
    tensor.unsqueeze_(0)
    relation, score = defect_detect(tensor, model)
    if score > 0.9:
        return html.H4(f"✅ Compatible! Score: {score:.4f}", style={"color": "lightgreen"})
    result, order = item_diagnosis(relation.squeeze(), [0, 1, 2, 3, 4])
    incompatible = categories[order[0]]
    return html.Div([
        html.H4(f"❌ Not Compatible (Score: {score:.4f})", style={"color": "salmon"}),
        html.P(f"Least compatible item: {incompatible.title()}")
    ])

if __name__ == "__main__":
    app.run(debug=True, port=8055, host="0.0.0.0")
