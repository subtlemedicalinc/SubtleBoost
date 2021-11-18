import ml_collections


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.encoder_num_layers = 3
    config.transformer.decoder_num_layers = 3
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.encoder_num_layers = 3
    config.transformer.decoder_num_layers = 3
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    config.representation_size = None
    config.patch_size = 8
    config.window_size = 3

    return config


def get_l8_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.encoder_num_layers = 3
    config.transformer.decoder_num_layers = 3

    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0

    config.patch_size = 8
    config.window_size = 3
    config.input_channels = 1
    config.output_patches = 1
    config.n_contrast = 4

    config.discriminator = ml_collections.ConfigDict()
    config.discriminator.n_layer = 4
    config.discriminator.gan_type = 'lsgan'
    config.discriminator.dim = 64
    config.discriminator.norm = 'bn'
    config.discriminator.activ = 'lrelu'
    config.discriminator.num_scales = 3
    config.discriminator.pad_type = 'zero'
    config.discriminator.input_dim = 1
    return config

    return config


def get_unet_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.encoder = ml_collections.ConfigDict()
    config.encoder.in_channels = 1
    config.encoder.base_filter = 16
    config.encoder.out_channels = 16
    config.decoder = ml_collections.ConfigDict()
    config.decoder.in_channels = 16
    config.decoder.base_filter = 16
    config.decoder.out_channels = 1
    config.output_patches = 1
    config.n_contrast = 4
    return config


def get_unet_l16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_l16_config()
    config.encoder = ml_collections.ConfigDict()
    config.encoder.in_channels = 1
    config.encoder.base_filter = 16
    config.encoder.out_channels = 16
    config.decoder = ml_collections.ConfigDict()
    config.decoder.in_channels = 16
    config.decoder.base_filter = 16
    config.decoder.out_channels = 1
    config.output_patches = 1
    config.n_contrast = 4
    return config

def get_unet_l8_config():
    """Returns the Resnet50 + ViT-L/8 configuration."""
    config = get_l8_config()
    config.encoder = ml_collections.ConfigDict()
    config.encoder.in_channels = 1
    config.encoder.base_filter = 16
    config.encoder.out_channels = 16
    config.decoder = ml_collections.ConfigDict()
    config.decoder.in_channels = 16
    config.decoder.base_filter = 16
    config.decoder.out_channels = 1
    config.output_patches = 1
    config.n_contrast = 4
    config.discriminator = ml_collections.ConfigDict()
    config.discriminator.n_layer = 4
    config.discriminator.gan_type = 'lsgan'
    config.discriminator.dim = 64
    config.discriminator.norm = 'bn'
    config.discriminator.activ = 'lrelu'
    config.discriminator.num_scales = 3
    config.discriminator.pad_type = 'zero'
    config.discriminator.input_dim = 1
    return config

def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_l16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = 'pretrained/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_contrast = 4
    config.n_skip = 0
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config
