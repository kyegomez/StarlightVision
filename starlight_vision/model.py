from starlight_vision import Unet3D, ElucidatedStarlight, StarlightTrainer

class Starlight:
    def __init__(self, 
                 dim=64, 
                 dim_mults=(1, 2, 4, 8), 
                 image_sizes=(16, 32),
                 random_crop_sizes=(None, 16),
                 temporal_downsample_factor=(2, 1),
                 num_sample_steps=10,
                 cond_drop_prob=0.1,
                 sigma_min=0.002,
                 sigma_max=(80, 160),
                 sigma_data=0.5,
                 rho=7,
                 P_mean=-1.2,
                 P_std=1.2,
                 S_churn=80,
                 S_tmin=0.05,
                 S_tmax=50,
                 S_noise=1.003):

        # Initialize the Unet models
        self.unet1 = Unet3D(dim=dim, dim_mults=dim_mults).cuda()
        self.unet2 = Unet3D(dim=dim, dim_mults=dim_mults).cuda()

        # Initialize the Starlight model
        self.starlight = ElucidatedStarlight(
            unets=(self.unet1, self.unet2),
            image_sizes=image_sizes,
            random_crop_sizes=random_crop_sizes,
            temporal_downsample_factor=temporal_downsample_factor,
            num_sample_steps=num_sample_steps,
            cond_drop_prob=cond_drop_prob,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
            P_mean=P_mean,
            P_std=P_std,
            S_churn=S_churn,
            S_tmin=S_tmin,
            S_tmax=S_tmax,
            S_noise=S_noise,
        ).cuda()

        # Initialize the trainer
        self.trainer = StarlightTrainer(self.starlight)

    def train(self, videos, texts, unet_number=1, ignore_time=False):
        self.trainer(videos, texts=texts, unet_number=unet_number, ignore_time=ignore_time)
        self.trainer.update(unet_number=unet_number)

    def sample(self, texts, video_frames=20):
        return self.trainer.sample(texts=texts, video_frames=video_frames)

