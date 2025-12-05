from timm import create_model

def get_autoencoder(vae_path, encoder_type="vq_gan_taming"):
    vae = create_model(model_name=encoder_type,  path=vae_path, pretrained=True)
    return vae
