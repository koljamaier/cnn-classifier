#!/usr/bin/env python
import argparse
from style_transfer.function import adaptive_instance_normalization
import style_transfer.net as net
from pathlib import Path
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.utils import save_image
from tqdm import tqdm

def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop:
        transform_list.append(torchvision.transforms.CenterCrop(size))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def transfer(content_dir, style_dir, output_dir, num_styles, extensions=['png', 'jpeg', 'jpg'], content_size=0, crop=False, style_size=512, alpha=1.0):
    style_dir = Path(style_dir)
    style_dir = style_dir.resolve()
    output_dir = Path(output_dir)
    output_dir = output_dir.resolve()
    print(style_dir)
    assert style_dir.is_dir(), 'Style directory not found'

    # collect content files
    assert len(extensions) > 0, 'No file extensions specified'
    content_dir = Path(content_dir)
    content_dir = content_dir.resolve()
    assert content_dir.is_dir(), 'Content directory not found'
    dataset = []
    for ext in extensions:
        dataset += list(content_dir.rglob('*.' + ext))

    assert len(dataset) > 0, 'No images with specified extensions found in content directory' + content_dir
    content_paths = sorted(dataset)
    print('Found %d content images in %s' % (len(content_paths), content_dir))

    # collect style files
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    print('Found %d style images in %s' % (len(styles), style_dir))

    decoder = net.decoder
    vgg = net.vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = input_transform(content_size, crop)
    style_tf = input_transform(style_size, crop)


    # actual style transfer as in AdaIN
    with tqdm(total=len(content_paths) * num_styles) as pbar:
        for content_path in content_paths:
            try:
                content_img = Image.open(content_path).convert('RGB')
            except OSError as e:
                print('Skipping stylization of %s due to error below' %(content_path))
                print(e)
                continue
            for style_path in random.sample(styles, num_styles):
                try:
                    style_img = Image.open(style_path).convert('RGB')
                except OSError as e:
                    print('Skipping stylization of %s with %s due to error below' %(content_path, style_path))
                    print(e)
                    continue

                content = content_tf(content_img)
                style = style_tf(style_img)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style,
                                            alpha)
                output = output.cpu()

                rel_path = content_path.relative_to(content_dir)
                out_dir = output_dir.joinpath(rel_path.parent)

                # create directory structure if it does not exist
                if not out_dir.is_dir():
                    out_dir.mkdir(parents=True)

                content_name = content_path.stem
                style_name = style_path.stem
                out_filename = content_name + '-stylized-' + style_name + content_path.suffix
                output_name = out_dir.joinpath(out_filename)

                save_image(output, output_name, padding=0) #default image padding is 2.
                style_img.close()
                pbar.update(1)
            content_img.close()
