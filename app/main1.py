import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import base64
import numpy as np
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
# from text_suggestions import generate_outfit_suggestion_by_id


sys.path.insert(0, "../mcn")
import torchvision.transforms as transforms
from model import CompatModel
from utils import prepare_dataloaders
from PIL import Image
from dash import Dash, dcc, html
data_root = "../data"
img_root = os.path.join(data_root, "images")

_, _, _, _, test_dataset, _ = prepare_dataloaders(root_dir=img_root, num_workers=1)
device = torch.device('cpu')
model = CompatModel(embed_size=1000, need_rep=True, vocabulary=2757).to(device)
# Load pretrained weights
model.load_state_dict(torch.load("../mcn/model_train_relation_vse_type_cond_scales.pth", map_location="cpu"))
model.eval()
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

def defect_detect(img, model, normalize=True):
    # Register hook for comparison matrix
    relation = None

    def func_r(module, grad_in, grad_out):
        nonlocal relation
        relation = grad_in[1].detach()

    for name, module in model.named_modules():
        if name == 'predictor.0':
            module.register_backward_hook(func_r)
    # Forward
    out  = model._compute_score(img)
    out = out[0]

    # Backward
    one_hot = torch.FloatTensor([[-1]]).to(device)
    model.zero_grad()
    out.backward(gradient=one_hot, retain_graph=True)

    if normalize:
        relation = relation / (relation.max() - relation.min())
    relation += 1e-3
    return relation, out.item()

def item_diagnosis(relation, select):
    """ Output the most incompatible item in the outfit
    
    Return:
        result (list): Diagnosis value of each item 
        order (list): The indices of items ordered by its importance
    """
    mats = vec2mat(relation, select)
    for m in mats:
        mask = torch.eye(*m.shape, dtype=torch.bool)
        m.masked_fill_(mask, 0)
    result = torch.cat(mats).sum(dim=0)
    order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
    return result, order

def vec2mat(relation, select):
    """ Convert relation vector to 4 matrix, which is corresponding to 4 layers
    in the backend CNN.
    
    Args:
        relation: (np.array | torch.tensor) of shpae (60,)
        select: List of select item indices, e.g. (0, 2, 3) means select 3 items
            in total 5 items in the outfit.
        
    Return:
        mats: List of matrix
    """
    mats = []
    for idx in range(4):
        mat = torch.zeros(5, 5)
        mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
        mat += torch.triu(mat, 1).transpose(0, 1)
        mat = mat[select, :]
        mat = mat[:, select]
        mats.append(mat)
    return mats

def retrieve_sub(x, select, order, try_most=5):
    """ Retrieve the datset to substitute the worst item for the best choice.
    """
    all_names = {0:'upper', 1:'bottom', 2:'shoe', 3:'bag', 4:'accessory'}
   
    best_score = -1
    best_img_path = dict()

    for o in order:
        if best_score > 0.9:
            break
        problem_part_idx = select[o]
        problem_part = all_names[problem_part_idx]
        for outfit in random.sample(test_dataset.data, try_most):
            if best_score > 0.9:
                break
            if problem_part in outfit[1]:
                img_path = os.path.join(test_dataset.root_dir, outfit[0], str(outfit[1][problem_part]['index'])) + '.jpg'
                img = Image.open(img_path).convert('RGB')
                img = test_dataset.transform(img).to(device)
                x[0][problem_part_idx] = img
                with torch.no_grad():
                    out = model._compute_score(x)
                    score = out[0]
                if score.item() > best_score:
                    best_score = score.item()
                    best_img_path[problem_part] = img_path
        if problem_part in best_img_path:
            x[0][problem_part_idx] = test_dataset.transform(Image.open(best_img_path[problem_part]).convert('RGB')).to(device)
    
            print('problem_part: {}'.format(problem_part))
            print('best substitution: {} {}'.format(problem_part, best_img_path[problem_part]))
            print('After substitution the score is {:.4f}'.format(best_score))
    return best_score, best_img_path

def base64_to_tensor(image_bytes_dict):
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

################################## LAYOUT ###################################

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "StyleSync Compatibility"
server = app.server

json_file = os.path.join(data_root, "test_no_dup_with_category_3more_name.json")
json_data = json.load(open(json_file))
json_data = {k:v for k, v in json_data.items() if os.path.exists(os.path.join(img_root, k))}

top_options, bottom_options, shoe_options, bag_options, accessory_options = [], [], [], [], []
for cnt, (iid, outfit) in enumerate(json_data.items()):
    if cnt > 10:
        break
    if "upper" in outfit:
        label = os.path.join(iid, str(outfit['upper']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        top_options.append({'label': label, 'value': value})
    if "bottom" in outfit:
        label = os.path.join(iid, str(outfit['bottom']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        bottom_options.append({'label': label, 'value': value})
    if "shoe" in outfit:
        label = os.path.join(iid, str(outfit['shoe']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        shoe_options.append({'label': label, 'value': value})
    if "bag" in outfit:
        label = os.path.join(iid, str(outfit['bag']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        bag_options.append({'label': label, 'value': value})
    if "accessory" in outfit:
        label = os.path.join(iid, str(outfit['accessory']['index']))
        value = os.path.join(img_root, label) + ".jpg"
        accessory_options.append({'label': label, 'value': value})

def category_card(label, dropdown_id, upload_id, preview_id, options):
    return dbc.Card([
        dbc.CardBody([
            html.H4(label, style={"color": "#00BFFF", "fontWeight": "bold", "fontFamily": "Segoe UI"}),
            dcc.Dropdown(id=dropdown_id, options=options, placeholder=f"Select {label}"),
            dcc.Upload(
                id=upload_id,
                children=html.Div([f"Upload {label}"], style={"fontSize": "15px"}),
                style={
                    'border': '2px dashed #00BFFF',
                    'padding': '12px',
                    'borderRadius': '8px',
                    'textAlign': 'center',
                    'color': 'white',
                    'backgroundColor': '#111'
                }),
            html.Img(id=preview_id, style={'maxHeight': '120px', 'marginTop': '10px', 'borderRadius': '8px'})
        ])
    ], className='mb-4', style={"backgroundColor": "#1c1c1c", "border": "1px solid #00BFFF", "borderRadius": "12px"})

body = dbc.Row([
    dbc.Col(html.Div([
        category_card("Top", 'top', 'upload-top', 'top-img-preview', top_options),
        category_card("Bottom", 'bottom', 'upload-bottom', 'bottom-img-preview', bottom_options),
        category_card("Shoes", 'shoe', 'upload-shoe', 'shoe-img-preview', shoe_options),
        category_card("Bag", 'bag', 'upload-bag', 'bag-img-preview', bag_options),
        category_card("Accessory", 'accessory', 'upload-accessory', 'accessory-img-preview', accessory_options),
        dbc.Card([
            dbc.CardBody([
                html.H5("Try count", style={"color": "#00BFFF"}),
                dcc.Slider(min=3, max=20, value=6, id="try-most-slider", marks={k:str(k) for k in range(3, 21)})
            ])
        ], style={"backgroundColor": "#1c1c1c", "border": "1px solid #00BFFF", "borderRadius": "12px"})
    ]), width=4),
    dbc.Col(html.Div([
        html.H3("Current Outfit", style={"color": "#00BFFF", "fontWeight": "bold"}),
        html.Div(id='original-score'),
        html.Div([
            html.Img(id='top-img', style={"maxHeight":"130px", "maxWidth":"130px", "margin":"5px", "borderRadius": "8px"}),
            html.Img(id='bottom-img', style={"maxHeight":"130px", "maxWidth":"130px", "margin":"5px", "borderRadius": "8px"}),
            html.Img(id='shoe-img', style={"maxHeight":"130px", "maxWidth":"130px", "margin":"5px", "borderRadius": "8px"}),
            html.Img(id='bag-img', style={"maxHeight":"130px", "maxWidth":"130px", "margin":"5px", "borderRadius": "8px"}),
            html.Img(id='accessory-img', style={"maxHeight":"130px", "maxWidth":"130px", "margin":"5px", "borderRadius": "8px"})
        ], style={"display": "flex", "flexWrap": "wrap"}),
        html.Br(),
        dbc.Button("Check Compatibility", id="submit-button", color="info", outline=True, style={'width': '100%', 'marginBottom': '10px', 'fontWeight': 'bold'}),
        dcc.Loading(id="loading-output", children=html.Div(id="output-state"), type="circle")
        # dbc.Button("Give me Suggestions", id="suggest-button", color="secondary", outline=True, style={'width': '100%', 'marginTop': '10px'}),
        # html.Div(id="suggestion-output", style={'marginTop': '20px', 'color': '#00BFFF', 'fontStyle': 'italic'})

    ]), width=8)
])

app.layout = html.Div([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("StyleSync Compatibility", style={"color": "white", "fontSize": "22px", "fontWeight": "bold", "letterSpacing": "1px"})
        ]),
        color="#000",
        dark=True,
        className="mb-4"
    ),
    dbc.Container([
        body
    ], style={"backgroundColor": "#000", "color": "white", "padding": "30px", "border": "2px solid #00BFFF", "borderRadius": "12px"}),
    html.Footer(
        html.Center("© 2020 Parth Gawande. All Rights Reserved."),
        style={"padding": "20px", "backgroundColor": "#000", "color": "#aaa"}
    )
])

################################## Callback ###################################

@app.callback(
    Output('top-img', 'src'),
    [Input('top', 'value'), Input('upload-top', 'contents')],
    [State('upload-top', 'filename'),
     State('upload-top', 'last_modified')])
def update_top(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(encoded_img.decode())

@app.callback(
    Output('bottom-img', 'src'),
    [Input('bottom', 'value'), Input('upload-bottom', 'contents')],
    [State('upload-bottom', 'filename'),
     State('upload-bottom', 'last_modified')])
def update_bottom(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback(
    Output('shoe-img', 'src'),
    [Input('shoe', 'value'), Input('upload-shoe', 'contents')],
    [State('upload-shoe', 'filename'),
     State('upload-shoe', 'last_modified')])
def update_shoe(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback(
    Output('bag-img', 'src'),
    [Input('bag', 'value'), Input('upload-bag', 'contents')],
    [State('upload-bag', 'filename'),
     State('upload-bag', 'last_modified')])
def update_bag(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback(
    Output('accessory-img', 'src'),
    [Input('accessory', 'value'), Input('upload-accessory', 'contents')],
    [State('upload-accessory', 'filename'),
     State('upload-accessory', 'last_modified')])
def update_accessory(fname, content, name, date):
    ctx = dash.callback_context
    triggered  = ctx.triggered[0]['prop_id']
    if 'upload' in triggered and content is not None:
            content_type, content_string = content.split(',')
            return 'data:image/png;base64,{}'.format(
                content_string)
    elif fname is not None and os.path.exists(fname):
        encoded_img = base64.b64encode(open(fname, "rb").read())
        return 'data:image/png;base64,{}'.format(
            encoded_img.decode())

@app.callback([Output('output-state', 'children'),
               Output('original-score', 'children'),
               Output('top-img', 'style'),
               Output('bottom-img', 'style'),
               Output('shoe-img', 'style'),
               Output('bag-img', 'style'),
               Output('accessory-img', 'style')],
              [Input('submit-button', 'n_clicks')],
              [State('try-most-slider', 'value'),
               State('top-img', 'src'),
               State('bottom-img', 'src'),
               State('shoe-img', 'src'),
               State('bag-img', 'src'),
               State('accessory-img', 'src'),
               State('top-img', 'style'),
               State('bottom-img', 'style'),
               State('shoe-img', 'style'),
               State('bag-img', 'style'),
               State('accessory-img', 'style')])

def update_output(n_clicks, try_most, top, bottom, shoe, bag, accessory, top_style, bottom_style, shoe_style, bag_style, accessory_style):
    out = [html.H5(children="Revised Outfit")]
    style={"max-height":"130px", "max-width":"130px", "margin":"5px"}
    redbox_style={"max-height":"130px", "max-width":"130px", "margin":"5px", "border": "5px solid red"}
    greenbox_style={"max-height":"130px", "max-width":"130px", "margin":"5px", "border": "5px solid green"}
    if n_clicks:
        start_time = time.time()
        img_dict = {
            "top": top.split(',')[1],
            "bottom": bottom.split(',')[1],
            "shoe": shoe.split(',')[1],
            "bag": bag.split(',')[1],
            "accessory": accessory.split(',')[1]
        }
        img_tensor = base64_to_tensor(img_dict)
        img_tensor.unsqueeze_(0)
        relation, score = defect_detect(img_tensor, model)
        if score > 0.9:
            original_score = html.H5(children="Original Score: {:.4f}. This outfit is compatible.".format(score), style={"color":"green"})
            return [None, original_score, top_style, bottom_style, shoe_style, bag_style, accessory_style]

        relation = relation.squeeze()
        result, order = item_diagnosis(relation, select=[0, 1, 2, 3, 4])
        best_score, best_img_path = retrieve_sub(img_tensor, [0, 1, 2, 3, 4], order, try_most)

        original_score = html.H5(children="Original Score: {:.4f}".format(score), style={"color":"red"})
        out.append(html.H5(children="Revised Score: {:.4f}".format(best_score), style={"color": "green"}))
        end_time = time.time()
        out.append(html.H5(children="Execute time for diagnosis: {:.4f}s".format(end_time - start_time), style={"color": "black"}))

        for part in ["top", "bottom", "shoe", "bag", "accessory"]:
            if part in best_img_path.keys():
                fname = best_img_path[part]
                encoded_img = base64.b64encode(open(fname, "rb").read())
                src= 'data:image/png;base64,{}'.format(encoded_img.decode())
                out.append(html.Img(id='{}-img-new'.format(part), style=greenbox_style, src=src))
            else:
                src = locals()[part]
                style={"max-height": "150px", "max-width": "150px", "margin": "5px"}
                out.append(html.Img(id='{}-img-new'.format(part), style=style, src=src))

        if "top" in best_img_path.keys():
            top_style['border'] = "5px solid red"
        else:
            top_style.pop('border', None)

        if "bottom" in best_img_path.keys():
            bottom_style['border'] = "5px solid red"
        else:
            bottom_style.pop('border', None)

        if "bag" in best_img_path.keys():
            bag_style['border'] = "5px solid red"
        else:
            bag_style.pop('border', None)

        if "shoe" in best_img_path.keys():
            shoe_style['border'] = "5px solid red"
        else:
            shoe_style.pop('border', None)

        if "accessory" in best_img_path.keys():
            accessory_style['border'] = "5px solid red"
        else:
            accessory_style.pop('border', None)

        return [out, original_score, top_style, bottom_style, shoe_style, bag_style, accessory_style]
    else:
        return [html.H5("Output after submisson"), None, top_style, bottom_style, shoe_style, bag_style, accessory_style]


@app.callback(
    Output("readme", "is_open"),
    [Input("readme-button", "n_clicks")],
    [State("readme", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run(debug=True, port=8050, host='0.0.0.0')

