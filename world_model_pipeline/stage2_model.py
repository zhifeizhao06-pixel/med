from guided_diffusion.script_util import create_model_and_diffusion


def stage2_model_args(image_size=128, num_channels=64, num_res_blocks=2, diffusion_steps=1000):
    return dict(
        image_size=image_size,
        class_cond=False,
        learn_sigma=True,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult="",
        in_ch=7,  # 4 MRI + coarse + uncertainty + noisy mask
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions="16",
        dropout=0.0,
        diffusion_steps=diffusion_steps,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        dpm_solver=False,
        version="new",
    )


def build_stage2(**overrides):
    args = stage2_model_args()
    args.update(overrides)
    return (*create_model_and_diffusion(**args), args)

