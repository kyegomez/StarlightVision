import torch 
import torch.nn as nn
from  torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.utils.data import DataLoader
from transformers import DiffusionModel, ClipModel, DiffusionConfig, DPTImageProcessor, DPTForDepthEstimation
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import math

#spatial transformer

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(SpatialTransformer, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.query_proj = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x, content_embeddings):
        #compute keys and values from content embedding
        keys =  self.key_proj(content_embeddings)
        values = self.value_proj(content_embeddings)

        #compute queries from input
        queries = self.query_proj(x)


        #conpute attention scores
        attention_scores = torch.matmul(queries, keys.tranpose(-1. -2)) / (self.in_channels ** 0.5)
        attention_scores = self.softmax(attention_scores)


        #compute attented scores
        attented_values = torch.matmul(attention_scores, values)

        #add residual connection and apply output projection
        out = x + attented_values
        out = self.output_proj(out)

        return out
    

class Midas(nn.Module):
    def __init__(self):
        super(Midas, self).__init__()
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    
    def forward(self, x):
        #prepare images for the model
        inputs = self.processor(images=x, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        #interplate to original size
        prediction = torch.nn.function.interpolate(
            predicted_depth.unsqueeze(1),
            size=x.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        return prediction.squeeze()
    


#sinusodial embeddings
class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_channels):
        super(SinusoidalEmbedding, self).__init__()
        self.num_channels = num_channels

    def forward(self, ts, T):
        ts = torch.Tensor([ts])
        T = torch.Tensor([T])
        ts_embed = torch.zeros(self.num_channels // 2)
        div_term = torch.exp(torch.arange(0, self.num_channels // 2, 2) * -(math.log(10000.0) / self.num_channels))
        ts_embed[0::2] = torch.sin(ts * div_term)
        ts_embed[1::2] = torch.cos(ts * div_term)
        return ts_embed


#unet

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.temporal = temporal
        if temporal:
            self.temporal_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn_temporal = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        if self.temporal:
            out = self.bn_temporal(self.temporal_conv(out))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, temporal=False):
        super(TransformerBlock, self).__init__()
        self.spatial_transformer = nn.TransformerEncoderLayr(d_model=in_channels, nhead=8)
        self.temporal = temporal
        if temporal:
            self.temporal_transformer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)

    def forward(self, x):
        out = self.spatial_transformer(x)
        if self.temporal:
            out = self.temporal_transformer(out)
        return out
    
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64, temporal=False),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128, temporal=False),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256, temporal=False),
            nn.MaxPool2d(2),
            ResidualBlock(256, 512, temporal=False),
        )
        
        # Center part of the UNet
        self.middle = nn.Sequential(
            TransformerBlock(512, temporal=False),
        )

        # Decoder
        self.decoder = nn.Sequential(
            ResidualBlock(512, 256, temporal=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResidualBlock(256, 128, temporal=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResidualBlock(128, 64, temporal=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # Pass input through the encoder
        z = self.encoder(x)

        # Pass the latent representations through the middle UNet layers
        z = self.middle(z)
        
        # Pass output to decoder
        out = self.decoder(z)

        return out



#extended unet
class ExtendedUNet(Unet):
    def __init__(self, in_channels, out_channels, blur_kernel_size=3, blur_sigma=1.0):
        super(ExtendedUNet, self).__init__(in_channels, out_channels)
        self.midas_dpt_large = Midas() # Load the MiDaS DPT-Large model
        self.clip_model = ClipModel.from_pretrained("openai/clip-vit-base-patch32")
        self.blur = GaussianBlur(blur_kernel_size, sigma=blur_sigma)

    def process_structure(self, x, ts, Ts):
        depth_maps = self.midas_dpt_large(x)
        for _ in range(ts):
            depth_maps = self.blur(depth_maps)
            depth_maps = F.interpolate(depth_maps, scale_factor=0.5)
        depth_maps = F.interpolate(depth_maps, size=x.shape[-2:])
        z_structure = self.encoder(depth_maps)
        return z_structure

    def process_content(self, x):
        content_repr = self.clip_model.encode_image(x)
        return content_repr

    def sample(self, x, t, c, ts, guidance_scale=1.0, temporal_scale=1.0):
        zt = self.encoder(x)
        z_structure = self.process_structure(x, ts, Ts)
        zt = torch.cat([zt, z_structure], dim=1)
        content_repr = self.process_content(c)

        # Apply the spatial transformer and cross-attention conditioning
        out = self.layers(zt, content_repr)

        # Compute the adjusted predictions
        unconditional_pred = self.decoder(out)
        conditional_pred = self.decoder(out)
        adjusted_pred = unconditional_pred + guidance_scale * (conditional_pred - unconditional_pred)

        # Control temporal consistency
        image_model_pred = ... # Compute the prediction of the image model applied to each frame individually
        adjusted_pred = image_model_pred + temporal_scale * (adjusted_pred - image_model_pred)

        return adjusted_pred



class Starlight(nn.Module):
    def __init__(self, Ts):
        super(Starlight, self).__init__()

        self.unet_lowres = ExtendedUNet()
        self.unet_midres = ExtendedUNet()
        self.unet_highres = ExtendedUNet()
        self.sinusoidal_embedding = SinusoidalEmbedding(4)
        self.Ts = Ts
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1)


    def forward(self, x, s, c, ts):
        #apply multi scale architecture
        x_lowres = F.interpolate(x, scale_factor=0.25)
        x_midres = F.interpolate(x, scale_factor=0.5)
        x_highres = x

        means_lowres = self.process_scale(x_lowres, s, c, ts)
        means_midres = self.process_scale(x_midres, s, c, ts)
        means_highres = self.process_scale(x_highres, s, c, ts)


        return means_lowres, means_midres, means_highres
    
    def process_structure(self, depth_maps, ts):
        for _ in range(ts):
            depth_maps = self.unet.blur(depth_maps)
            depth_maps = F.interpolate(depth_maps, scale_factor=0.5)
        depth_maps = F.interpolate(depth_maps, size=depth_maps.shape[-2:])
        z_structure = self.unet.encoder(depth_maps)
        return z_structure
    
    def process_scale(self, x, s, c, ts):
        # Compute content representation
        content_embedding = self.unet.process_content(x)

        # Compute structure representation
        depth_maps = self.unet.midas_dpt_large(x)
        structure_embedding = self.process_structure(depth_maps, ts)

        # Condition on structure
        zt = torch.cat([x, structure_embedding], dim=1)

        # Add sinusoidal embedding of ts
        ts_embed = self.sinusoidal_embedding(ts, self.Ts).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        ts_embed = ts_embed.repeat(x.size(0), 1, x.size(2), x.size(3))
        zt = torch.cat([zt, ts_embed], dim=1)

        # Apply cross-attention
        zt = self.unet.spatial_transformer_blocks(zt, content_embedding, c)

        # Process LSTM temporal context
        lstm_input = zt.view(zt.size(0), 1, -1)
        _, (hidden_state, _) = self.lstm(lstm_input)
        hidden_state = hidden_state.view(hidden_state.size(1), -1, 1, 1)

        # Concatenate hidden_state with zt
        zt = torch.cat([zt, hidden_state], dim=1)

        # Pass through UNet
        means = self.unet(zt)

        return means
    
model = Starlight()

def train_starlight(model, train_loader, optimizer, device, perceptual_loss_fn, style_loss_fn, alpha, beta):
    model.train()

    for inputs, content, structure, ts in train_loader:
        inputs = inputs.to(device)
        content = content.to(device)
        structure = structure.to(device)
        ts = ts.to(device)

        optimizer.zero_grad()

        means_lowres, means_midres, means_highres = model(inputs, structure, content, ts)

        loss_lowres = F.mse_loss(inputs, means_lowres)
        loss_midres = F.mse_loss(inputs, means_midres)
        loss_highres = F.mse_loss(inputs, means_highres)

        # Perceptual and style losses
        perceptual_loss = perceptual_loss_fn(inputs, means_highres)
        style_loss = style_loss_fn(inputs, means_highres)

        loss = loss_lowres + loss_midres + loss_highres + alpha * perceptual_loss + beta * style_loss
        loss.backward()

        optimizer.step()
