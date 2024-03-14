# MAE

Group channels here in

class MaskedAutoencoderGroupChannelViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=96, patch_size=8, in_chans=10, spatial_mask=False,
                 channel_groups=((0, 1, 2, 6), (3, 4, 5, 7), (8, 9)),
                 channel_embed=256, embed_dim=1024, depth=24, num_heads=16,
                 decoder_channel_embed=128, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 proj_ratio=4):
        super().__init__()

    - need to change channels and groups to adapt

# Dataset Type Creation
 elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size*4, mean, std) # input_size*2 = 96*2 = 192
        dataset = SentinelIndividualImageDataset(file_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)

# Dataset Class
    class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

# Classes
- don't need if have generic training need

# Example data formats
- https://purl.stanford.edu/vg497cb6002

README.md
File 1

2.85 kB	 DownloadREADME.md
fmow-sentinel.tar.gz
File 2

77.50 GB	 Downloadfmow-sentinel.tar.gz
test_gt.csv
File 3

21.16 MB	 Downloadtest_gt.csv
train.csv
File 4

178.78 MB	 Downloadtrain.csv
val.csv
File 5

21.15 MB	 Download