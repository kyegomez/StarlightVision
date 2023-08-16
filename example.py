import torch
from starlight_vision import Unet3D, ElucidatedStarlight, StarlightTrainer

unet1 = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()

unet2 = Unet3D(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()

# elucidated starlight, which contains the unets above (base unet and super resoluting ones)

starlight = ElucidatedStarlight(
    unets = (unet1, unet2),
    image_sizes = (16, 32),
    random_crop_sizes = (None, 16),
    temporal_downsample_factor = (2, 1),        # in this example, the first unet would receive the video temporally downsampled by 2x
    num_sample_steps = 10,
    cond_drop_prob = 0.1,
    sigma_min = 0.002,                          # min noise level
    sigma_max = (80, 160),                      # max noise level, double the max noise level for upsampler
    sigma_data = 0.5,                           # standard deviation of data distribution
    rho = 7,                                    # controls the sampling schedule
    P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
    P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn = 80,                               # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
).cuda()

texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles',
    'dust motes swirling in the morning sunshine on the windowsill'
]

videos = torch.randn(4, 3, 10, 32, 32).cuda() # (batch, channels, time / video frames, height, width)

# feed images into starlight, training each unet in the cascade
# for this example, only training unet 1

trainer = StarlightTrainer(starlight)

# you can also ignore time when training on video initially, shown to improve results in video-ddpm paper. eventually will make the 3d unet trainable with either images or video. research shows it is essential (with current data regimes) to train first on text-to-image. probably won't be true in another decade. all big data becomes small data

trainer(videos, texts = texts, unet_number = 1, ignore_time = False)
trainer.update(unet_number = 1)

videos = trainer.sample(texts = texts, video_frames = 20) # extrapolating to 20 frames from training on 10 frames

videos.shape # (4, 3, 20, 32, 32)
