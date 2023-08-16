import torch
from starlight_vision import Starlight

# Example of usage:
model = Starlight()

texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles',
    'dust motes swirling in the morning sunshine on the windowsill'
]

videos = torch.randn(4, 3, 10, 32, 32).cuda()

model.train(videos, texts=texts)
sampled_videos = model.sample(texts=texts, video_frames=20)
